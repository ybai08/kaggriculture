# SPDX-License-Identifier: Apache-2.0
# Derived work. Base: "Farming Score V4: A Better Shop" by lynnsakurai (ARLENE),
# https://www.kaggle.com/code/lynnsakurai/farming-score-v4-a-better-shop, version 7,
# released under the Apache License 2.0. Copied 2026-09-14 into this repository.
# This file is the Kaggle submission (main.py). Modifications (farmbot project, 2026-09-14): appended layer "F1" at the end of the
# file that enables the chassis front_run hook with our own route as the opponent's
# expected plan, so copies of the public tape are out-sold by one step. No other
# lines were changed. See docs/BUILD_NOTES.md for measurements.
# 2026-09-14 (tape search t1): the compressed route payload was re-encoded with a few
# premium-product SELL orders moved by one step (research/tape_search.py); no other change.
"""Kaggriculture route-replay chassis (pure Python, stdlib only).

A *route* is a pre-computed tape of 719 Kaggle-format actions
``{"farmer": [op, ...], "hands": [[op, ...], ...], "market": [[order, item, qty], ...]}``.
The chassis replays the tape chosen by a caller-supplied ``router`` and wraps it
in small reactive layers (each independently switchable via ``settings``):

    hand_align            pad/truncate hands to the real hand count      (fieldbook_logic)
    weed_repair           DIG a weed that blocks PLANT/BUILD, replay      (tetsutani + task spec)
    sell_lead             sell next step's lots one step early            (fieldbook _lead_sale)
    front_run             sell before the opponent's scheduled SELL       (hook; opponent_plan)
    budget_guard          fund each 72-step block's purchases             (six_day_budget_guard.hpp)
    room_guard            keep shed <= 99 at hour 23                      (tetsutani)
    clamp_sells           trim SELL orders to the projected shed          (tetsutani)
    dead_stock            sell stock the route will never sell            (tetsutani)
    terminal_liquidation  step >= 718: sell the whole projected shed      (fieldbook _terminal_sale)

Engine facts (verified against kaggle_environments 1.32.7, env_1_32_7.py):
  observation["farms"][p] = {"money", "tiles"[y][x], "farmer"[x,y], "hands"[[x,y]..],
                             "unlocked_quadrants", "hires_today"}
  tiles: None (empty) | "LOCKED" | {"kind": WEED|COOP|PASTURE|PLANT, "crop"/"animal", ...}
  observation["private"] = {"shed": {item: n}, "seeds": {crop: n}, "inventories": [{}...]}
  observation["market"] = {"inventory": {...}, "prices": {...}}
  observation["town"] = {"unlocked_shops": [...]}
  Agents act on steps 0..718 (interpreter marks DONE once step >= episodeSteps-2).
"""
from __future__ import annotations

import copy

PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
SEED_PRICE = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
ANIMAL_STRUCTURE = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}
LAND_PRICES = (1000, 2000, 4000)
MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
FRONT_RUN_ITEMS = ("MILK", "WOOL", "STRAWBERRY", "MELON")
LAST_ACT_STEP = 718
PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}

DEFAULT_SETTINGS = {
    "hand_align": True,
    "weed_repair": True,
    "sell_lead": True,
    "front_run": True,
    "budget_guard": True,
    "room_guard": True,
    "clamp_sells": True,
    "dead_stock": True,
    "terminal_liquidation": True,
    # tunables
    "block_turns": 72,
    "shed_capacity": 100,
    "board_size": 10,
    "max_orders": 10,
    "turns_per_day": 24,
    "min_sell_price": 2,
}


# --------------------------------------------------------------------------- helpers
def _get(value, key, default=None):
    """Field access that works for dicts and Kaggle Struct/attribute objects."""
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def _int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _fib(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _step_of(observation):
    raw = _get(observation, "step")
    if raw is not None:
        return _int(raw)
    return _int(_get(observation, "day", 0)) * 24 + _int(_get(observation, "hour", 0))


def _shed_adjacent(pos, board):
    if not isinstance(pos, (list, tuple)) or len(pos) < 2:
        return False
    half = board // 2
    return pos[0] in (half - 1, half) and pos[1] in (half - 1, half)


def _tile_at(tiles, pos):
    try:
        x, y = int(pos[0]), int(pos[1])
        return tiles[y][x]
    except (TypeError, ValueError, IndexError):
        return "LOCKED"


def _is_noop(act, tile, inv, seeds, pos, board):
    """True when the engine will certainly ignore ``act`` (mirrors _apply_unit_action)."""
    if not act:
        return True
    op = act[0]
    x, y = pos[0], pos[1]
    if op in MOVES:
        dx, dy = MOVES[op]
        return not (0 <= x + dx < board and 0 <= y + dy < board)
    if op == "PASS":
        return True
    adjacent = _shed_adjacent(pos, board)
    if op == "DROP":
        return (not adjacent) or (not inv)
    if op == "PICKUP":
        return not adjacent
    if op == "PLACE":
        item = act[1] if len(act) > 1 else None
        if item in ANIMAL_STRUCTURE and isinstance(tile, dict) \
                and _get(tile, "kind") == ANIMAL_STRUCTURE[item] and _get(tile, "animal") is None:
            return _int(_get(inv, item, 0)) <= 0
        return (not adjacent) or _int(_get(inv, item, 0)) <= 0
    if tile == "LOCKED":
        return True
    is_dict = isinstance(tile, dict)
    kind = _get(tile, "kind") if is_dict else None
    animal = is_dict and _get(tile, "animal") is not None
    if op == "PLANT":
        return tile is not None or _int(_get(seeds, act[1] if len(act) > 1 else None, 0)) <= 0
    if op == "WATER":
        return kind != "PLANT" or bool(_get(tile, "watered_today"))
    if op == "HARVEST":
        return (not is_dict) or _int(_get(tile, "yield_units", 0)) <= 0
    if op == "FERTILIZE":
        return kind != "PLANT" or _int(_get(inv, "FERTILIZER", 0)) <= 0
    if op == "DIG":
        return tile is None or animal
    if op in ("BUILD_COOP", "BUILD_PASTURE"):
        return tile is not None
    if op == "FEED":
        return (not animal) or bool(_get(tile, "fed_today")) or _int(_get(inv, "WHEAT", 0)) <= 0
    if op == "COLLECT_FERTILIZER":
        return (not animal) or (not _get(tile, "fertilizer_available"))
    if op == "CARE":
        return (not animal) or bool(_get(tile, "cared_today"))
    return True


class _View:
    """Cheap per-step snapshot of everything the layers read from the observation."""

    def __init__(self, observation, player, cfg):
        farms = list(_get(observation, "farms", []) or [])
        self.farm = farms[player] if player < len(farms) else {}
        self.rival = farms[1 - player] if len(farms) >= 2 and 1 - player < len(farms) else {}
        private = _get(observation, "private", {}) or {}
        self.shed = {k: max(0, _int(v)) for k, v in dict(_get(private, "shed", {}) or {}).items()}
        self.seeds = dict(_get(private, "seeds", {}) or {})
        self.invs = [dict(i or {}) for i in (_get(private, "inventories", []) or [])]
        market = _get(observation, "market", {}) or {}
        self.prices = {k: _int(v) for k, v in dict(_get(market, "prices", {}) or {}).items()}
        self.money = float(_get(self.farm, "money", 0.0) or 0.0)
        self.tiles = _get(self.farm, "tiles", []) or []
        self.board = len(self.tiles) or cfg["board_size"]
        self.positions = [_get(self.farm, "farmer", None)] + [list(p) for p in (_get(self.farm, "hands", []) or [])]
        self.hires_today = _int(_get(self.farm, "hires_today", 0))
        self.quadrants = len(list(_get(self.farm, "unlocked_quadrants", []) or []))

    def inv(self, idx):
        return self.invs[idx] if idx < len(self.invs) else {}

    def in_hands(self, item):
        return sum(max(0, _int(_get(inv, item, 0))) for inv in self.invs)


# --------------------------------------------------------------------------- chassis
class Chassis:
    """Replays ``routes[router(...)]`` with reactive safety/market layers.

    routes         : {route_id: list of >= 719 Kaggle action dicts}
    router         : callable(observation, step, state_dict) -> route_id, called every
                     step; ``state_dict`` is per-player and persists across the game.
    settings       : overrides for DEFAULT_SETTINGS (layer switches + tunables)
    opponent_plan  : optional list of the opponent's expected actions (front_run hook)
    """

    def __init__(self, routes, router=None, settings=None, opponent_plan=None):
        self.routes = {rid: list(tape) for rid, tape in routes.items()}
        self.router = router or (lambda observation, step, state: next(iter(self.routes)))
        self.cfg = dict(DEFAULT_SETTINGS)
        self.cfg.update(settings or {})
        self.opponent_plan = opponent_plan
        self.players = {}
        self.diagnostics = {"layer_fallbacks": 0, "entry_fallbacks": 0}
        self._future_sells = {}   # route id -> {item: [remaining planned SELL qty from step t]}

    # ---- state -----------------------------------------------------------------
    def _state(self, player, step):
        st = self.players.get(player)
        if st is None or step == 0 or step <= st["last_step"]:
            st = {"last_step": -1, "route": None, "router_state": {},
                  "pending": {}, "sell_state": {"due_step": -1, "suppress": {}}}
            self.players[player] = st
        st["last_step"] = step
        return st

    def _route_action(self, route, step):
        tape = self.routes[route]
        if 0 <= step < len(tape) and isinstance(tape[step], dict):
            return copy.deepcopy(tape[step])
        return copy.deepcopy(PASS_ACTION)

    def future_sells(self, route, item, step):
        """Planned SELL quantity of ``item`` in route steps >= ``step`` (suffix sums)."""
        table = self._future_sells.get(route)
        if table is None:
            tape = self.routes[route]
            n = len(tape)
            table = {p: [0] * (n + 1) for p in PRODUCTS}
            for t in range(n - 1, -1, -1):
                for p in PRODUCTS:
                    table[p][t] = table[p][t + 1]
                for o in (tape[t].get("market") or []) if isinstance(tape[t], dict) else []:
                    if o and o[0] == "SELL" and len(o) >= 3 and o[1] in table:
                        table[o[1]][t] += max(0, _int(o[2]))
            self._future_sells[route] = table
        col = table.get(item)
        return col[step] if col and 0 <= step < len(col) else 0

    # ---- main entry -----------------------------------------------------------
    def act(self, observation, configuration=None):
        if len(_get(observation, "farms", []) or []) < 2:
            raise ValueError("incomplete observation")  # factory falls back to tape
        step = _step_of(observation)
        player = _int(_get(observation, "player", 0))
        st = self._state(player, step)
        cfg = self.cfg
        view = _View(observation, player, cfg)

        route = self.router(observation, step, st["router_state"])
        if route not in self.routes:
            route = st["route"] if st["route"] in self.routes else next(iter(self.routes))
        st["route"] = route
        action = self._route_action(route, step)
        raw = copy.deepcopy(action)
        try:
            if cfg["hand_align"]:
                self._hand_align(action, view)
            if cfg["weed_repair"]:
                self._weed_repair(action, view, st, route, step)
            if cfg["sell_lead"] or cfg["front_run"]:
                self._apply_suppression(action, st["sell_state"], step)
            projected = self._projected_shed(action, view)
            lead_available = dict(projected)
            next_sup = {"due_step": -1, "suppress": {}, "r36_debts": st["sell_state"].get("r36_debts", {})}
            if cfg["sell_lead"]:
                self._sell_lead(action, view, lead_available, route, step, next_sup)
            if cfg["front_run"] and self.opponent_plan:
                self._front_run(action, view, lead_available, route, step, next_sup)
            st["sell_state"] = next_sup
            if cfg["budget_guard"]:
                self._budget_guard(action, view, route, step)
            if cfg["room_guard"]:
                self._room_guard(action, view, route, step)
            if cfg["clamp_sells"]:
                self._clamp_sells(action, projected)
            if cfg["dead_stock"]:
                self._dead_stock(action, view, projected, route, step)
            if cfg["terminal_liquidation"]:
                self._terminal_liquidation(action, projected, step)
            action["market"] = action["market"][: cfg["max_orders"]]
            return action
        except Exception:
            self.diagnostics["layer_fallbacks"] += 1
            return raw

    # ---- layer: hand_align ----------------------------------------------------
    def _hand_align(self, action, view):
        """Pad with PASS / truncate the tape's hand list to the real number of hands
        (fieldbook_logic.act). Extra hands would be ignored by the engine anyway;
        missing ones just idle, so alignment only tidies the action."""
        expected = max(0, len(view.positions) - 1)
        hands = list(action.get("hands") or [])
        hands.extend([["PASS"] for _ in range(max(0, expected - len(hands)))])
        action["hands"] = hands[:expected]

    # ---- layer: weed_repair ---------------------------------------------------
    def _weed_repair(self, action, view, st, route, step):
        """If a PLANT/BUILD_* target tile is a WEED, DIG now and queue the intended
        action for that unit; the queue replays on a later step when the unit still
        stands there and its tape action would be a no-op (the displaced no-op is
        queued behind it, so PLANT -> WATER chains survive). A PLANT is only replayed
        when the unit's next tape action is not a move, so the mandatory same-day
        WATER can follow; otherwise the seed is kept. A no-op turn spent on a weed
        is also converted to DIG (tetsutani weed_dig)."""
        units = [action.get("farmer") or ["PASS"]] + list(action.get("hands") or [])
        pending = st["pending"]
        tape = self.routes[route]
        nxt = tape[step + 1] if step + 1 < len(tape) and isinstance(tape[step + 1], dict) else {}
        next_units = [nxt.get("farmer") or ["PASS"]] + list(nxt.get("hands") or [])
        for i in range(min(len(units), len(view.positions))):
            pos = view.positions[i]
            if not isinstance(pos, (list, tuple)):
                continue
            pos = (int(pos[0]), int(pos[1]))
            tile = _tile_at(view.tiles, pos)
            act = list(units[i])
            queue = pending.get(i)
            if queue and queue[0][0] != pos:
                pending.pop(i, None)
                queue = None
            is_weed = isinstance(tile, dict) and _get(tile, "kind") == "WEED"
            noop = _is_noop(act, tile, view.inv(i), view.seeds, pos, view.board)
            next_op = next_units[i][0] if i < len(next_units) and next_units[i] else "PASS"
            if act and act[0] in ("PLANT", "BUILD_COOP", "BUILD_PASTURE") and is_weed:
                pending.setdefault(i, []).append((pos, act))
                act = ["DIG"]
            elif queue and noop:
                _, replay = queue[0]
                if replay[0] == "PLANT" and next_op in MOVES:
                    pending.pop(i, None)          # WATER could never follow: keep the seed
                else:
                    queue.pop(0)
                    if act and act[0] != "PASS" and act[0] not in MOVES:
                        queue.append((pos, act))
                    act = replay
                    if not queue:
                        pending.pop(i, None)
            elif is_weed and noop:
                act = ["DIG"]
            units[i] = act
        action["farmer"] = units[0]
        action["hands"] = units[1:]

    # ---- projected shed -------------------------------------------------------
    def _projected_shed(self, action, view):
        """Shed contents after this step's unit actions but before the market runs:
        PICKUP removes, DROP/PLACE(non-animal) near the shed adds up to capacity
        (fieldbook _projected_shed / tetsutani projected shed)."""
        cap = self.cfg["shed_capacity"]
        proj = {p: view.shed.get(p, 0) for p in PRODUCTS}
        for k, v in view.shed.items():
            proj.setdefault(k, v)
        total = sum(proj.values())
        units = [action.get("farmer") or ["PASS"]] + list(action.get("hands") or [])
        for i in range(min(len(units), len(view.positions))):
            if not _shed_adjacent(view.positions[i], view.board):
                continue
            act = units[i]
            op = act[0] if act else "PASS"
            inv = view.inv(i)
            if op == "PICKUP" and len(act) >= 2 and act[1] in proj:
                qty = min(proj[act[1]], max(0, _int(act[2]) if len(act) >= 3 else 1))
                proj[act[1]] -= qty
                total -= qty
            elif op == "DROP":
                for item, held in inv.items():
                    take = min(max(0, _int(held)), max(0, cap - total))
                    if take > 0:
                        proj[item] = proj.get(item, 0) + take
                        total += take
            elif op == "PLACE" and len(act) >= 2 and act[1] not in ANIMAL_STRUCTURE:
                item = act[1]
                take = min(max(0, _int(act[2]) if len(act) >= 3 else 1),
                           max(0, _int(_get(inv, item, 0))), max(0, cap - total))
                if take > 0:
                    proj[item] = proj.get(item, 0) + take
                    total += take
        return proj

    # ---- layer: sell_lead / front_run suppression ------------------------------
    @staticmethod
    def _apply_suppression(action, sell_state, step):
        """Remove from this step's SELLs the quantities already sold a step early."""
        if sell_state.get("due_step") != step:
            return
        remaining = dict(sell_state.get("suppress", {}))
        kept = []
        for order in action.get("market") or []:
            order = list(order)
            if order and order[0] == "SELL" and len(order) >= 3 and remaining.get(order[1], 0) > 0:
                removed = min(max(0, _int(order[2])), remaining[order[1]])
                order[2] = _int(order[2]) - removed
                remaining[order[1]] -= removed
                # A zero-quantity order keeps later market race slots intact.
            kept.append(order)
        action["market"] = kept

    @staticmethod
    def _add_sell(action, item, qty, max_orders, merge=True):
        market = action.setdefault("market", [])
        if merge:
            for order in market:
                if order and order[0] == "SELL" and order[1] == item:
                    order[2] = _int(order[2]) + qty
                    return True
        if len(market) >= max_orders:
            return False
        market.append(["SELL", item, qty])
        return True

    def _sell_lead(self, action, view, projected, route, step, next_sup):
        """fieldbook _lead_sale: when step % 4 != 0 (no town consumption between the
        two steps) sell the lots the tape plans to SELL next step now, for products
        other than WHEAT/FERTILIZER we already hold, and suppress them next step.
        Skipped at the last step, at shop-unlock boundaries and if a SELL for that
        product is already queued this step."""
        cfg = self.cfg
        nxt = step + 1
        unlock_period = 3 * cfg["turns_per_day"]
        if nxt > LAST_ACT_STEP or nxt % unlock_period == 0 or step % 4 == 0:
            return
        tape = self.routes[route]
        future = tape[nxt] if nxt < len(tape) and isinstance(tape[nxt], dict) else {}
        planned = {}
        for o in future.get("market") or []:
            if o and o[0] == "SELL" and len(o) >= 3 and o[1] in PRODUCTS:
                planned[o[1]] = planned.get(o[1], 0) + max(0, _int(o[2]))
        already = {o[1] for o in action.get("market") or [] if o and o[0] == "SELL" and len(o) > 1}
        for item in PRODUCTS:
            if item in ("WHEAT", "FERTILIZER") or planned.get(item, 0) <= 0 or item in already:
                continue
            qty = min(projected.get(item, 0), planned[item])
            if qty <= 0 or view.prices.get(item, 0) < cfg["min_sell_price"]:
                continue
            if not self._add_sell(action, item, qty, cfg["max_orders"], merge=False):
                break
            projected[item] -= qty
            next_sup["suppress"][item] = next_sup["suppress"].get(item, 0) + qty
        if next_sup["suppress"]:
            next_sup["due_step"] = nxt

    def _front_run(self, action, view, projected, route, step, next_sup):
        """Hook: if ``opponent_plan`` (their expected tape) schedules a SELL of
        MILK/WOOL/STRAWBERRY/MELON next step, sell what we hold of it now (before
        their supply depresses the price) and suppress our own SELL of that quantity
        next step. Bounded by our own remaining planned sales so it never dumps."""
        cfg = self.cfg
        nxt = step + 1
        plan = self.opponent_plan
        if nxt > LAST_ACT_STEP or nxt >= len(plan) or not isinstance(plan[nxt], dict):
            return
        already = {o[1] for o in action.get("market") or [] if o and o[0] == "SELL" and len(o) > 1}
        for o in plan[nxt].get("market") or []:
            if not (o and o[0] == "SELL" and len(o) >= 3 and o[1] in FRONT_RUN_ITEMS):
                continue
            item = o[1]
            if item in already or view.prices.get(item, 0) < cfg["min_sell_price"]:
                continue
            own_next = sum(max(0, _int(x[2])) for x in self.routes[route][nxt].get("market", [])
                           if len(x) >= 3 and x[0] == "SELL" and x[1] == item)
            qty = min(projected.get(item, 0), max(0, _int(o[2])), own_next)
            if qty <= 0:
                continue
            if not self._add_sell(action, item, qty, cfg["max_orders"], merge=False):
                break
            projected[item] -= qty
            already.add(item)
            next_sup["suppress"][item] = next_sup["suppress"].get(item, 0) + qty
        if next_sup["suppress"]:
            next_sup["due_step"] = nxt

    # ---- layer: budget_guard --------------------------------------------------
    def _block_requirements(self, view, route, start, end):
        """Planned purchase cost and item reserves for tape steps [start, end)
        (six_day_budget_guard.hpp calculate_six_day_requirements)."""
        tape = self.routes[route]
        budget = 0.0
        seed_bal, item_bal = {}, {}
        seed_need, item_need = {}, {}
        hires_by_day = {}
        quadrants = view.quadrants
        for t in range(start, min(end, len(tape))):
            a = tape[t] if isinstance(tape[t], dict) else {}
            for u in [a.get("farmer") or ["PASS"]] + list(a.get("hands") or []):
                if not u:
                    continue
                op = u[0]
                arg = u[1] if len(u) > 1 else None
                qty = max(1, _int(u[2]) if len(u) > 2 else 1)
                if op == "PLANT" and arg in SEED_PRICE:
                    seed_bal[arg] = seed_bal.get(arg, 0) - 1
                    seed_need[arg] = max(seed_need.get(arg, 0), -seed_bal[arg])
                elif op == "FEED":
                    item_bal["WHEAT"] = item_bal.get("WHEAT", 0) - 1
                    item_need["WHEAT"] = max(item_need.get("WHEAT", 0), -item_bal["WHEAT"])
                elif op == "FERTILIZE":
                    item_bal["FERTILIZER"] = item_bal.get("FERTILIZER", 0) - 1
                    item_need["FERTILIZER"] = max(item_need.get("FERTILIZER", 0), -item_bal["FERTILIZER"])
                elif op == "PLACE" and arg is not None:
                    item_bal[arg] = item_bal.get(arg, 0) - qty
                    item_need[arg] = max(item_need.get(arg, 0), -item_bal[arg])
            for o in a.get("market") or []:
                if not o:
                    continue
                op = o[0]
                item = o[1] if len(o) > 1 else None
                qty = max(1, _int(o[2]) if len(o) > 2 else 1)
                if op == "HIRE":
                    day = (t - start) // self.cfg["turns_per_day"]
                    hires_by_day[day] = hires_by_day.get(day, 0) + 1
                elif op == "BUY_LAND":
                    extra = quadrants - 1
                    if 0 <= extra < len(LAND_PRICES):
                        budget += LAND_PRICES[extra]
                        quadrants += 1
                elif op == "BUY_SEED" and item in SEED_PRICE:
                    budget += SEED_PRICE[item] * qty
                    seed_bal[item] = seed_bal.get(item, 0) + qty
                elif op == "BUY_PRODUCT" and item in ("WHEAT", "FERTILIZER"):
                    budget += view.prices.get(item, 0) * qty
                    item_bal[item] = item_bal.get(item, 0) + qty
                elif op == "BUY_ANIMAL" and item in ANIMAL_COST:
                    budget += ANIMAL_COST[item] * qty
                    item_bal[item] = item_bal.get(item, 0) + qty
        for day, n in hires_by_day.items():
            first = view.hires_today if day == 0 else 0
            for k in range(n):
                budget += _fib(first + k)
        return budget, item_need

    def _budget_guard(self, action, view, route, step):
        """At every block boundary (step % 72 == 0) make sure cash + the value of
        stock the block already plans to sell covers the block's purchases (hires,
        land, seeds, animals, products). A shortfall is covered by extra SELLs of
        unprotected shed stock, highest price first; SELLs are moved in front of
        the buys so the money is there when they execute."""
        cfg = self.cfg
        block = cfg["block_turns"]
        if block <= 0 or step % block != 0:
            return
        budget, item_need = self._block_requirements(view, route, step, step + block)
        market = action.setdefault("market", [])
        existing = {}
        for o in market:
            if o and o[0] == "SELL" and len(o) >= 3:
                existing[o[1]] = existing.get(o[1], 0) + max(0, _int(o[2]))
        cash = view.money
        for item in PRODUCTS:
            planned = max(existing.get(item, 0), self.future_sells(route, item, step)
                          - self.future_sells(route, item, step + block))
            cash += min(view.shed.get(item, 0), planned) * view.prices.get(item, 0)
        shortfall = budget - cash
        if shortfall <= 0:
            return
        candidates = []
        for item in PRODUCTS:
            price = view.prices.get(item, 0)
            if price < cfg["min_sell_price"]:
                continue
            protected = max(0, item_need.get(item, 0) - view.in_hands(item))
            avail = view.shed.get(item, 0) - protected - existing.get(item, 0)
            if avail > 0:
                candidates.append((-price, item, avail, price))
        candidates.sort()
        added = False
        for _, item, avail, price in candidates:
            if shortfall <= 0:
                break
            qty = min(avail, -(-int(shortfall) // price))
            if self._add_sell(action, item, qty, cfg["max_orders"]):
                shortfall -= qty * price
                added = True
        if added:
            sells = [o for o in market if o and o[0] == "SELL"]
            others = [o for o in market if not (o and o[0] == "SELL")]
            action["market"] = sells + others

    # ---- layer: room_guard ----------------------------------------------------
    def _room_guard(self, action, view, route, step):
        """tetsutani room_guard: at hour 23 the end-of-day drop pushes every unit's
        inventory into the shed and overflow is destroyed. Estimate the shed after
        this step (stock + carried + harvest/collect - feed/fertilize/place + buys -
        sells) and, if it exceeds capacity-1, add SELLs preferring products with no
        future planned sale, then highest price."""
        cfg = self.cfg
        if step % cfg["turns_per_day"] != cfg["turns_per_day"] - 1:
            return
        cap = cfg["shed_capacity"]
        units = [action.get("farmer") or ["PASS"]] + list(action.get("hands") or [])
        carried = sum(max(0, _int(n)) for inv in view.invs for n in inv.values())
        produced = consumed = 0
        for i in range(min(len(units), len(view.positions))):
            tile = _tile_at(view.tiles, view.positions[i])
            a = units[i]
            if not a:
                continue
            op = a[0]
            if op == "HARVEST" and isinstance(tile, dict):
                produced += max(0, _int(_get(tile, "yield_units", 0)))
            elif op == "COLLECT_FERTILIZER" and isinstance(tile, dict) and _get(tile, "fertilizer_available"):
                produced += 1
            elif op in ("FEED", "FERTILIZE"):
                consumed += 1
            elif op == "PLACE" and len(a) > 1 and a[1] in ANIMAL_STRUCTURE:
                consumed += 1
        market = action.setdefault("market", [])
        planned_sells, planned_buys = {}, 0
        for o in market:
            if not o:
                continue
            if o[0] == "SELL" and len(o) >= 3:
                planned_sells[o[1]] = planned_sells.get(o[1], 0) + max(0, _int(o[2]))
            elif o[0] in ("BUY_PRODUCT", "BUY_ANIMAL") and len(o) >= 3:
                planned_buys += max(0, _int(o[2]))
        shed_total = sum(view.shed.values())
        fillable = sum(min(view.shed.get(it, 0), n) for it, n in planned_sells.items())
        needed = shed_total + carried + produced - consumed + planned_buys - fillable - (cap - 1)
        if needed <= 0:
            return
        priority = sorted(PRODUCTS, key=lambda it: (self.future_sells(route, it, step + 1) > 0,
                                                   -view.prices.get(it, 0), it))
        for item in priority:
            avail = max(0, view.shed.get(item, 0) - planned_sells.get(item, 0))
            qty = min(needed, avail)
            if qty <= 0 or view.prices.get(item, 0) < 1:
                continue
            if not self._add_sell(action, item, qty, cfg["max_orders"]):
                continue
            planned_sells[item] = planned_sells.get(item, 0) + qty
            needed -= qty
            if needed <= 0:
                break

    # ---- layer: clamp_sells ---------------------------------------------------
    @staticmethod
    def _clamp_sells(action, projected):
        """Clamp against a sequential stock upper bound, retaining market slots.

        Earlier BUY_PRODUCT orders can fund a wheat wash's sell leg. Their full
        quantity is an upper bound; the engine enforces actual cash/capacity.
        Removing empty orders would change the later lockstep market races.
        """
        avail = dict(projected)
        kept = []
        for o in action.get("market") or []:
            if o and o[0] == "SELL" and len(o) >= 3:
                have = avail.get(o[1], 0)
                n = min(_int(o[2]), have)
                n = max(0, n)
                avail[o[1]] = have - n
                kept.append(["SELL", o[1], n])
            else:
                kept.append(o)
                if o and o[0] in ("BUY_PRODUCT", "BUY_ANIMAL") and len(o) >= 3:
                    avail[o[1]] = avail.get(o[1], 0) + max(0, _int(o[2]))
        action["market"] = kept

    # ---- layer: dead_stock ----------------------------------------------------
    def _dead_stock(self, action, view, projected, route, step):
        """tetsutani dead_stock: stock beyond everything the rest of the route still
        plans to SELL is dead; sell it now when price > 1 (on day 29 everything not
        already in this step's orders is dead). Highest value lots first."""
        planned = {}
        for o in action.get("market") or []:
            if o and o[0] == "SELL" and len(o) >= 3:
                planned[o[1]] = planned.get(o[1], 0) + _int(o[2])
        day = step // self.cfg["turns_per_day"]
        extra = []
        for item in PRODUCTS:
            have = projected.get(item, 0) - planned.get(item, 0)
            if have <= 0:
                continue
            surplus = have if day >= 29 else have - self.future_sells(route, item, step + 1)
            if surplus > 0 and view.prices.get(item, 0) > 1:
                extra.append(["SELL", item, surplus])
        extra.sort(key=lambda o: -view.prices.get(o[1], 0) * o[2])
        action["market"] = (action.get("market") or []) + extra

    # ---- layer: terminal_liquidation -----------------------------------------
    def _terminal_liquidation(self, action, projected, step):
        """fieldbook _terminal_sale: on the final acting step (>= 718) replace the
        market orders with a SELL of the whole projected shed."""
        if step < LAST_ACT_STEP:
            return
        action["market"] = [["SELL", item, qty] for item, qty in projected.items()
                            if qty > 0 and item in PRODUCTS][: self.cfg["max_orders"]]


# --------------------------------------------------------------------------- factory
def make_agent(routes, router=None, opponent_plan=None, **settings):
    """Build a Kaggle ``agent(observation, configuration)`` closure that never raises:
    a failure inside a layer falls back to the raw tape action, and a failure even
    before that falls back to PASS (with hands padded when possible)."""
    chassis = Chassis(routes, router, settings, opponent_plan)

    def agent(observation, configuration=None):
        try:
            return chassis.act(observation, configuration)
        except Exception:
            chassis.diagnostics["entry_fallbacks"] += 1
            try:
                step = _step_of(observation)
                player = _int(_get(observation, "player", 0))
                tape = chassis.routes.get(chassis.players.get(player, {}).get("route"),
                                          next(iter(chassis.routes.values())))
                if 0 <= step < len(tape):
                    return copy.deepcopy(tape[step])
            except Exception:
                pass
            try:
                farms = _get(observation, "farms", []) or []
                hands = _get(farms[_int(_get(observation, "player", 0))], "hands", []) or []
                return {"farmer": ["PASS"], "hands": [["PASS"] for _ in hands], "market": []}
            except Exception:
                return copy.deepcopy(PASS_ACTION)

    agent.chassis = chassis
    return agent


import base64
import json
import zlib
_PAYLOAD=json.loads(zlib.decompress(base64.b85decode('c-ri}U9TiZvLyCj_<SBJKPyvjJ;T}=VoCOZY;Iw7K@6^>D=Z+aAnfkjLjPUeeKPAK!raU}vYImsEieFUs_RrmM0!NHo12^e_wW9<fBmoj_TB&TFaQ0!|MOq}%YXj2|ML0ecYpiv>#yJa{`TE}`q%&a|M;(;-}wCUKmY50`M>_{zkPoFFW>#?xBvL}zy0#@```Zj>ASb@u7ACJ`262)`2Fo)zWep^<B#Qg(Z_%Pf8QQ|^XHGheEMtrqxLsXU;gE<Km2m}&VBLaYd-w^!%rW+{J@_-UcY<mtG9ps>*eydFaJ=C`qz)YJ&*eH;{EXFfB)Oxwy*l~McZoLr}+5l`O}YcE`I6u!S|s&{p7>fvETaBZ$JF_{og--<kN3ortbXM`=UPmSn(}#f<J!v^Tn87_56!J#lLgC{QBwZi$8x5Q`$c1uG;O3%XP2!!mpRh_n-g$r^}BYe+DK}t`{G{=Q;oU>Go~GJH^vcL+Z*Q?F$P`3H;hVwCjgYmtV?PUzdq2`@j1#lBZv||MC5s_r<ns)tchFJv{yH?TN-~egDk!3Z)-<ZEe=V)*pIbzm_Nb^hM_S-~X-NPfx$QpTpI>w=a13z5R#;3%{Q(k0YoI%KP1Fz3azaFLU|(YMF<`GPCDhFL&|jSFAgD&TQS^Kb`*J(pC623*YOXCF`v>8oLVX34(<M7sxkUQ1U>x0jQr%Ea?5aodpF~(xC+{e=7CG<m-wTO}^0cB1ayk_H)4X2OguoVakERw;FEj`>?iuGrpz1e{X-g{gd-0AAkJu^5;)~|96*PKK=0H5C3a<#NECLelX+k0zW?Z&fPO8-iG^b3tApN`gy-cflGC{Tt1Xf_<og^j;a%(MP6XEx_qX=PlG8tXU)DF>);|hM~w-_zf-Mb+qrW8nr|~quzwGY_jb~}+t!Omdp{kHoYld;J7DMJw?4#H)^NKVdNwb<>FOu^KZlaP?cYX)NL7csgq-AkOYszKWA)X`0&RA2Mb4tSm)Y;hOCJG2?E9o$kOeNn&cn-#5E3~)do?$QAhnvnO*}SQ%Unq~e{eOAx{P$Na`Fi71;gFH@UI_!`44S)23h5s6|@9j)^gv5eISgrpTh$@<OH+QM?PrY!0COu17c??e%|1w7;)m^79A*%;5o;@kG2Lgp=U6_59PA5F9ay~W)#MAA`4KDQ#vAKd;CF&M8Y*0(olr-MOc!=dX>urgx-j_k*joI0{zeaR`OhX)$bsfsJ>vL7j2IL92uFLw_fvl!1uLyx@6ieN%btM=v-mZ=&pOVwBh&6bE@Zv@7Xk3TVH#RnJYZ4GAnCd1uTJge?^wM?YONR{9-x54e?sP1kKGdnS{&84OCx(-giUPu*ne`{GVb62lR-B$Rj%LSfaS8W&Xk+FTo^urw6!!zcFq<*Wf`gFYNxkA%8`Fzx{0X#RZ8-@@oy(KJeO%A8K*qBCbl>t(y+Hb&hqh49}||C+Y`%V>heGySnXO3&oHSS-r+_@}|wZD$;1Lg-K$cwGDIv-&f)oDIWxYAPs(1`f|CgIdDa|SM_3tw)RA#v`LGDg069zF>_NfB>1>JQJPb0JthDcK&%8z<n6Wg^YkQ9mX(^4bI8|wFt8B7&pMFIB)}#3Ap_@elCSLh+=1s&_{1y-Oj&>3fz>C!^ZhR$|CaSkKnO6k!&w{%J(eeL{du`<Y!A%&a`1Qx=dV}!<ohrFF8s(Ivu_H<bYlh(oG$y<!6^l2&x=sQL)p9AAJ<EC+)s^2PuA^578$`Khn>FLQB<!)6T+X1j5VY6K51@F!S;W+zkmJo%ZKZqF2DTpAAxP7;6VM}v55hILge5QRIwWIe3-rG&etQp-6k<+oIiWL?CRb1=RW@9GtRgv?u^`@vQn<#!M_A&iHW+8M$qQDFm$^Sw2Um&KjSvTJec5y_+$gkE|||T4qsDVTa?$O3g#^5$0><Ho&vkxLppQd#gs$1KX;fpk^Jdmgtu$bU|y%T*zJo$r05O$0_a@G?Zn5O5&A=Kjg<9zZ`K|Hn0ozhYMyuZ_2b9S|CtRxd<UerKfnWcD38JpK^fnZLB)a3t@`xy;qkqyP`&?kGV-`DeJ~Z}^4z{4$Q-m#;aj7p9d!peAK*9;J_vgxjtc{P$C}a(Mp=jHpv)G8<Na$dt~*E&pi(jS$t*nT-0K1sBt4~AVg}#iTlG37%Al(;faA#_dayd^upH#g4_WGgg4>~_=NpbnU41Hl7;HS6sgM<?x=;oVp6)<(_@QT7oOfn0@MiHjSWL)$NSoZ^TYY`bhE+MCSB*y3PzC@3W4h}Ftvz^3gUh-Ub{0G-#>gRC@xhnOg>1AIn1-A_k*5R**WdkT2~G&aAZgiez567?YFGK}Xy);cdUklz#UWyXe$c--Mr_MfMM2ryIgn3zbkxKta0Ad|3igwy-v8c$NmssU9ZoUdNvHq)n+s(f$-1Qw=^8VBI6*G_mbK$K4c=jQ^<E!bc_8X9bWX2+Bod!)4Yg`7QbxYdzO@LnnzyMtL>TH=2`COJu^hq8y6TH3zHaJUM$i%2n%q-jMdWf1vqG*N8jkghzy%_<A=n&738xQ$gP2&5Q}=Pi>m+?idXb_ZrD{;>7S9F%&Kl7j^}eT3DOgxBp1`{Ml)Mc0&C3I~h~ASW5G=kxBf4)vKOxvPJCSGvVbP`vzuPt&k&n@dKrsAc8Q?_h8k4j$;8Nld!P$cMEP9_t)=>hJt@j}@1qO(LxA;w*h>U?;pcg?^H&3j-Vm^X&BxPAf!Z&_tg6&{91r9_iM)|}>qB*d`q?4xrmR(-pA8&gkhGiQ`6CB=s95YzZdc{(sxOqM3r|&t(?LPhKhadm_ma)<-m6*r={xw}6W~F8nG2diSbD5kv5m@K=0DN3YpoYQ5{g07;qLn0}Y&zvhD}1?5ncb&U?$s7V99_lr_TiLq2FA%;XwFSwNky<53I<RnkvFv0$kg#9OD&cXs~iYF!X^UQX|Pmv)U89`R8A_-h4(pi^(??KA(T?-SP8C$Pkm22xP1egMH$kOr>vpFu-kn|fKjDZsl*9emZ{zij~VQk?1<OMBh`8lbE3csJeja40D)JpGWohB0Y6nt0KEP4`xAH>4*>Y_Cadv{Y7a+tq5VcmU}XEv_{YOaNuVf=$VcfK+y3`PLMYCTg?AzwlRp9}EDfEs#BW>lQuGRuFAkT9f9u>`8_^S_XEbj!EF57^s}jU{W`73JN|tw4Z#-ixJtMA`X-ZG#^QcQ}*Qx;bv5vIsV?~pNgrowju2rFy(g1D>r<#El5GP&CP_Rb+lKz=98W3+{xAx{o6&vu|##n*q3e6|_L{$?D$Q%G5I(?=Rk6gAgFv}<Y!D7%Qn9`DqYGuuhC0_wYUL@Q-%_$X_!n3}Sf<ka|wW;Hj*eg0aP<CRduizzAWXzw)V0u1aW)6qVQQa14MWzFY9CZ_s9765e%PSYG3O?`Iou6Ur=B>n|N6O=)UL0K|;)kWVhAWh|DpA)2V-^-Pxw&zoBB-glal+5cYBIElR?qkpY88*(jOTWXGhayh4PKd1lnmu|Yc+jotUl&}YX;i*ht(gxoOViHunoj-d(nFEFCRbtdXdSrAu)@jcq<KPrw09asP6|lHvTa)0%wm8`Y~ReI{<=3g#@nmbw41vglk?|0Rq22WDO)KA#JZkRr!VJv-m&<!W3$Jq<}KWf?NpMwDjPNc(z3Zf)ZZM>I(+IIa&qDQQ=yY8psgcnYO-hGh(J={=>r`;Egr%b9su|E8V|2sP3u;Ocx0K0sRsn8*UaV@6Go{-fR~f_S5fvch|>3gDuKu?w#{I*kg0=`V6%K1+{3@l=RFB-^=xBGv&#;BRn$JyVD{kWpmJLe$zf1IoBs9lao=`If&rgd#{K5^T$#HNBZzJm1$c)<7S^sAcl)v6hr5B8WWWP2Tw-0cH818#AA<8@HOJXJOn$`*2v4;U%T;0x$;$)`Z-S`JXnn6&7CL+&hq;KcpGwIaJ07-Kv2{S==1nG)U}z;UqAhQp7&FarJiAtCa9C6-7DW$?gh+u5Np)Qyr8hOR-3JGw`h%cH-)#C^~5bIJ={m5N)wA0FL(}4UtKETRs9>tg?>mYTUbqaofo+xM+qENhrV#v0_<nVAxMc*B!u9xEJHi+#dU!;_^30AJ#ASLN`eMin(7^<s-9KzYT=|xx3JDZKOTJG;?*KyLtIKB%b{YzNQxOoON<CB;C`>k6S9y3dbOgBtyLPSIyuQ%uw%LQZaca1E|^)Lv$aU<V-&<KhL88j_Gi+1`t1;LRIr=y9FTq;qq5r)MZS@-URXZhN%=Z5r<`LYwU-HiI1NSYP$|JYv{DJzW3>YSH&;}6+E-fWxk$y_*2~H8ApO?#ypFXF^$~UhmP3*ZSX|Cqgw-dXByyK$b~}Zo&+0p2RX4slSYeww<>h{AR~O^7D}T?BL{@uM2P4?y{D4JjFb{y&Y6J+As<G<kbBr}Mj01VI;`_Kwk$PFH8at<xuTzt8tGYKO6{0Jb+RNpyva1k2uEW-jEj=&WXo*%gNcpevGf3&rnFVxp7(um{<2e}T@-fu}W)HN{LgqBks#gbO6^u=J^d$IaU>TkB0n?<2ADWqiK1eGCQ&}S>1x_V#GtIofmaY9mOz5Dh7MW$J*$Rp=3=&WNC)rIB<X>)O1(|rjt4W|zMN9(}mnqOfe^5<d8C(5NK0y`QGYB_m*;dVcxxS|1g>6*Qc_lr;8#lc)CUzL$o*=G=O$45wwNB_6MH5{{9=PRNw9YQ$0yN{Y*6!D&qm)!CaXU`(^W=5zrW!B^xS5k2s#2u_4?ygz4-qi3rWVy;QEa1&D3HlYte4vxNO*cuU`l}e6=Sy>XsSSy*1Wu|P7D;`8HMsxa-vBE5L~<UZ`5Oq8@T88gl%#x9c0EM4Z7&E+{51yaZ<iPdmrSOWD|8Z?w7F<T9^v$F(V4`=9k5gTH~Y6H^=(|Lqjk>EtS#h)L~=)8>@l_B_N{lkLd)tDwCr!4fF{Z@o>&-q=aBmauDT_8~s*Rq=1-u;+U13B%%rksF`3bY}Gxxe^@{o0Xs?`=)Md$1)rID9haI$zH%M!-N+2J#Wl2n3Kp4GOcU+|%Q?^teoFcb_T5mwW9OupXo<cu@^nWg$(z?DvCL4V`1vlN`GTnFa(awjVw);E+cwy`ojs{(bLCuOL8D3_C#%zvNxza1boeIe9y@ta94{|YJoXziMX%{mlfo`o&IxrIXoAw1qDNy`E*{|}`J^@2$7o}$>XRoY_{xZ|k{Nye6#%G=emEJkcgpK&ddC-n_j&*vOky7CxHbGi`ATWU;MfhTNPCeMfuD!<!Z5HrhXLoE29F8uy;Wlcjgift>Xi_IkUf5;8I3{Smr8~H$~aMV!Z;J`5RKg{x{zv+hyga5wmf87oWxEmFwrW8ka2)?S{YNMmYV?(8!6#YU9HVNOLt{It4wpt7T1q3QRhQuF*NH{u<{XrVL)tNLo}}Fb)DQnUH9tZ$HuZUcHcYvpafODn9yr?hvLqA8|oziDRV4Ywv0=Z(11IaMnMlpx+5g>4N%MmAcS^{xF+-3o~m50cLTpeZ>#0)1+$~&k)2^TQ8vSwR)T<ZmYDb$fD9R~FO4lJ78pKXW_#$PsmP1Ml%n*<CI9#>8xD6H=hzW5z6{NdowW)cx!IQ1hcb?{`_qqFFNUWkZuIFRAo4zaJ=Zq>N7QwD3t~4B@Qy0kdil#=Hox;9Ow=U=z=%Q^`1O_^AdhUvDrd4*y{g;P0b}*K=UV1U&(F(Ia^V-J8<#6k#Oa(&Y1k>^R$Y@?T1XjGQ<^T9qaY2V>e#rJy`Z>Ubrv3Qf$ZkQwc3Sh(L`mheYrY$rsk9{c;+5nZp<;cx9$@z9>e_-(uu6ZG17~9jXR2n8?h|avxArEERpadSTA_w#nnIIGrljb`%s(m^ICuolE-Qo{0R6}P^sBeH?{!~p9p-&^RS80l9Dsj0IYsU4C#58X=aW6_!Pu=wD^W+pj+6a1Zk5ev>jbDc}0v$3*6FgQ*(_W#g^)m`ndP?QNI7-FJS*z;yw+23Ej*r@Rv$zZN-GP1kLmExZT9X(Ic9KzC1ntJPE-4W6z?Myux`^Y2X2RPlB)A1h&(ISLOF*0){|r+!cIOC|FE)&_DX!S7yfmJAn^Q))+PDUx^yPyg3&|J3t}!yRV~wT$DA+n0uv?b7reN$}MGA9E*D465L}A2L(vok5&z_ho6W-um&z0mrdn_0Xq$t7+`|*1Cj#Znn|?GAxeGOVgRO~z;uDZu&a0_tC%lZA^4TRcI!mNF0ng1q5@sqP543z2~sc$D=BXXk*N~a)x<_%MX!2>G}vhF!NwABVC@B^K|m5Z@dHMv@;#!~rQW&B&R*c5US)kd_DQXzv!irSK>qi}6lph>rzf#F+oun(JRKCeEo#MCLQ(6wjO8X@{@_24)_{p#h_3OCsuv#Zg#FF4s>kefsee->GBBYGsN&LM95Qekb*smol*eKkOY|G88Gs<|F{QW7YEG3@vWOm8^{m4;RQTl{o^NoL^Y32TJsvRWX(}2&#2t49`LrB_dC0Odd|uK-5p3VPYlUJ^wg4TrE+b>d4%z8U8Zz$*ZeKEu9K4`?REDt;;(M0^Opcc^G+tK5kl^d6b{pRmfHC#d7^F~_HB>|kUD6QH=)b~HQNjFN-N2(wSeY_6gCiW*OAm?Waaw3zOkJ_wrACSV6hz@IgJby&c*e2Mg`J6Cq{qcdOdZAXws5ZWR?aB|5L|IBd1gO_)&>UEv87Xp86{A31JYNN@Wberm0wyHdW?qUVxop>gZfy2sEPp%`Z<&CguT2W*JF)!gehaTH^9os>a0m0rctg56D1Pj^G;A<nG`-4*>OklZ>G}5S~#~Fx{?Enm?~zeizps;){?e>b#o3Wf!|<PwkaO4w(=C3qGm#l(pL;R^bqh(8ATy=F2YjzZtgO{f^k|UTjw1z#7vPD;TqkvxIC3^$VlN4JSv1bs4RW$A*&Y3{@pj500+Sd>JCgH2F!(jLmbXXL6!G8L<kI3KMXRlWG<0ycp6mS-r=QQ;kx}5X~HHoUX@lJz%fj=NBxhI2DAUk9j32_S5}N9s2cxJ2&3LaAWE;1d(;lxtXuG#)RBbK6Zhlv3c`rh9iAmDqlB64fSw|Wu;)en3qsw*KtlDZbyQ4<hHCzjB5ge+LD+}vM_+vktgEP4_*SWC8qb)bZVs`6lT~;cF$AKLQc|QSmI|+sy=~Gs8$|E9UzcoR#)%(^2Tr?X99DFcMcb@BmI&z}dxdDBHIkHRvYa7Q36K7omQ0gqCu3VvU>(2>@<`*1tVT8q-I8)igF<@-;5P_bihan1;C-kW7v#NG=mC$6SwBl--8|#)YHw;6XzWg-ct*gwZqY`giU7MKL=PdL2VWx%$w+h#&^8w})kUTN480^H)3Ac{Eg-LCq%d@W1ZD?;vjizWF(k<oAg)uhM!#Uu%MOxqqDs}8LR{x2P`QqLuBV!1)Mo^xa)P8W&jxmRuwj#vCa<Z}_c4CpITv=~`So-4k<!+;%7v*6uH7LaOG#;X-$^GxauBh>KNO9TT^9_d1jHD4o`J+wHvyCqS^?`nY7r1#vQ)cmhTw;#@C2m<n<DeN+^0gvm>wX~;zJM2(LGJo64h&K7y8rg#z*9^l^<;8{a93UxNe9QL5q6{#!<HL|Ir)>D~^`>7~z~6<ue#hM8<aD8q1}68@2MlCelBa@RSK+3`)10f-P317r7iw-V-Tu^K=r$u8VjGRySd}2i+>>h;)-~T;etU!$TyRz$VfHDekiHo}G7@{*lDF2m~ysD&WQF`$INXqgHSNP$t+zf;1f=>(4NX(}wYB`&@)jgi!JhA486N@VJSq@UqVdrLZPZj5hkLP>@R=dCfb(wS~~JLkox*LP%g*TA_r&+IdoJLjwUkqctR}?K`CB5CcU(Je$YBb#E#cVOSEyCX%03(=i|;zmk17c<W3Rlr>2-RXdvD*eU>XqWkWfNB6YJ9t<aEUKzsrDN-@b-hERQmK+cc<;5sVh#W`}q)<)#O|5qjbA}=K*r;$~Xrn@guu-;6^?qpbx1mpyViUuhtz?e^cIHJ2oI)Qr(8Utz7tR=E;X5xRjLK@kF&c?T$D)Zt=)JXs*Nss=>q5@gSM(DN@?<aZ<pujjs*zE``t=yE%F2>w!+97Uv+REga1c~z5~E$YP>Ib{YQ`397}T$UmT_lWn+clVOl3NWA;8?{YZ{!e(sV}Lr<V+A0FbD#Uoy3=kpU4!c0bIDwyHr#uuA_U3X7C#FpB#8tD6_FS#&}r!6LiW+W{lH;1VjJk1(%pg%hANEmi>feR-@I!7iXG#Ziezq(ahmfIz*wd>w2kz(XY7TFaW^0z`GdkuB&w6+;z*=Yfor76XuDRaX+BFF97w(q7$?CuQKvvR*@rAJ-ZnnDs6R82VKZ%olY84#GSYXVeWw`aiA!hPUqE`VR(Qp-IK#e5jpjeO}RPhMtsdflM7Ih}!O0NJEg9Q=?LyKuBGzI3G=Mq#Od1QRQeXnfD@Bn*Pn}a<o>Z!v$wO3=WCZyT3+*6y$idCsxR|Lg~kf0@T#X-R*u|TEg@q9dPdVTQ@q(!dsbqO%ioT11&%Urx!5!ftGgi+W7-GSa9@^!}R(p^Uo74Xj_mBx#)#$9N6B)AaJF(kM8iB5S}`Dd=dXX*?a^8x%|urP65dbI|tG;-5{aF%=nMokVp*Iaw$npb=<@)bIxCr?T3sVS{r$rs4s&~h^8x%FMfTW3cogtHy8<(EMF>JBdR`T-TpVG%JSKNotM5+U11QnmkViYBzDyU$zyvYM^$;U{T0Yi{PI&vp64Zd0^YbPIl@GcuPj~=HCRu;Lfs?w9WX({B9RXy2v@;@5XqIKRA{rhB889ottl_rHY^syVvzKIlao=G+W{(&&uH1&{-lHx>^8S<QHZENNuUk%ilFWpA;Xeyu)BH;i7|!hNM?D7sXl}LA$Kjcv8)*mkt!7?jv8RErjz`1T@TzP(qC+mZ^cW9$izwBFkH509nLYnMhIoLr9wTbHh>vzrq&VX66NOkT3G0F;b`TR+5s^>Cx<!xo(RZl<7d*IOi1r#s7PwZ_?kC7_ieh8>Czw`O0=Pv3YKuFP)q}YB>V;Q4YGNo9CZSW?K9jxCZiD!24%M+hJkcdr{6Q&b5G{R5?OEYk|44nZynj&O3SRvWgOc16@m+LF*@<Kq?IiZZ0e~}9<*HRJ*bY(n1fXkwaW|=pXBsly6TOgywvbQMYwT0La$AWV13|T6<pYX)Xdn*<{o|X&CC?D<5|y#+L31Z=EQ|tlsb56qSLqKDSVp#@MbMaiPRPF9D^X7Tl-h^VBs@UScG;-E=Fe)%Du39TLjL2f)d9ymdI`M1hJ7nZ7ADGNvZSJ9m)qVbP`|06Cs%*;3@$EDJ9%AidbCTb+VQ6WWQumxbtiS2lAjyQUkBIAV*zuvK)@yx6S%DR$R=txd|BWm4jTi+RFD&CAd*8yV+(jUw&=q9H*XzA0J!BPj&XL`5sa9>6k4OURsH6b>>v7PlW(VKRFjo-dCdLZxEne=$%-(vBA~pM7k59Ir2R77=(<B>2BsROI4S*_IQY-U+YND?5Am4b)V@)nN9(x>uaL$G#3-c6#@Ca5ws4-qdGwhsut>$o(#W)G6X;>R8CqY1Vd#fp%D}&3U=eOv5bJt<Aw^P)5IiW6gVyFq-drn)!5I7{l9pNna`Y_aW_U&1uPiux3f4lhC+%lY4tQayH}JqV)%$6W8~yyg(0^<AN#ZU)`SegZV_4}LjZ5xc`N~Ifwg^=@z#2Pis3Xc=fyriq_bKQU!J80{9XJJyUPhHPL7mAt}%H)sLh6T&BUa~Og%Mz8d8u6cw+|G(o`17<m01(`nnsrzA@Z0eSqTe(-)*{-Rv`Ng7yCBPtD<?O5D?&4r!zC(s&W@VxE_}{C{O?q<-P*u2d@Oo`RjiYa*>p<cO0OIWUI>FRW_n>X!brXF*Ir(}{<3KO6~60L@Fk1GVG`QL<*-fH`++)CXwOIm~Eo0>kD3ZXycTy$oC(NiE)Dw@{=JayX~Y-}=i(bO7$VbNhg#1n|CU%4}h@te;3YdJ&F2Q|}g6-a-&tb&%*Zoua*Bo^2zjLBIxPQJG9~%`tF}b}TfW^#S+}Wv_xzkJViQd&n4TO3k7>B~U-i(|X>>E%N%sMs9FahU6Bd*ACAr;48wcH9A#b-%3d+G*9eoc;yKo@FWFd0Rt%CBKTp%a_?V2Q+dHF{Xl7NnE0Jz%CaGn@hxdAeH_z?-Y^is;ca2CO?VUJa892fh6@0J+~08<mcE7KVl?W4+yzGe;{a+j_1u=Aa{!H7NPoqIa8%Tj_=sc+K1Peq-8ubS05r)ws^gs|x}^e;Rs-2D2zJ|UHzAn3&n0c^DkP|j_+SsZ1Pv{MXx@yfP{7j3h-dC*)8`V>lBpVOkj;TgK&%3+3eCRGH#LsAN@72_Ha_Y6+ihjSky8xmtClD60t|TWXY#3J2e76rZqUFbg3wd~&rCdl8m6puQNX1t?y$U}(6vs2gjI<BX#ZRk5X1rs&0p$Zt~=$5odrg(`l$glgN!<Ib?9B-6d$sP3io5FKUbR^nn`L?ze(jaS|&pSZMM_F>>6g3i6S*?hQ^reAM5cRMs#IeAFHS}iJ9l>d>mYh_W)khWHWpG)bvH$gLqRq7<m*4N^t~b1oixsn=R)w0EJT<HF_9`t1w-i^d2H4KJLa0y?y@v^Poq26d?#SR*m~h89X5brG(Thag~u#dB}B3dIH&i&s$t_;ouaES#oudmg~jo@D5kZ#)nMEn6hM^dw9d#rogqre5_Xhb94rNMi7uJX!|l-w#ATiU<{v>qyiXcdFA8>48$*K7#8QqNr-n`Lm8DZR{&7JRr0tcN8)y~kxGj?MoWG^5#9GrI7WH)*fCAXNfWA7vz|3-97ymT;M$QE*qSpv)a$w)B}FiO-lhV9tA{@HO;zQbxKZ6S4Nhs=*bU5C=x>YL#Z*;pqAE!hqD#8AjKwyUtUx0y<$<b9a9YGrsW!79z|A0Qw5MMr%M}N;Y-Xulxl3F~XGEb0sw$MZUMX=lBDqbMETOta5gGs@yGE{D6&^itG5-r|65v#curAM6>-Zod$pLbK+$MK1d8wNpw5xxo{ej+NN|OnOsz1mX=hTC-zXu0ROZnJKfsUkx%UjUBR<pWXp3G}&o&1hCAgqUBrHuYP*dWwpWx`$W9^1JVm<<B(<#{(~kmc5p_N_JI3=4-Q&8TT7@9^wDr?R9w{xHrj?+JHIYS)OvSo79<K0jnKD5h<rXh;nr)b9kLXBF<hRNuZ18Er|NOs5$}uy~CaTdIX9a$p_^<E({rMxpurD}Bj*w@Ct$MNM(p8B1F90e6}#x@~5x$KNN@TAtb-(}<1=4;V?~og3Y@2LYj|K60^QR^2$7tNg!&ftRXJtT$L{fAVYX-Xu_Lyh?{ecKh;r1+`paL|ue!p$B)`qrKBe>ji`ZT8Ef_)pn(EZcQQy!DanOTc%V^6J!0jD^(wGTC~V{w3LO%TPA@*JOaQ;^#uV(0pg8$X=$jk9?Xp;eNdOidCrp4vf9-(_48$XCE3J<7XJAi(^gtR6bTN!e{#evNgOig)ldWgP8BF6s<O-u_s8g$af)h**0ts_t?T2e-Hz`sU(#N5?bVcN9hgMto?_Z;B4rpnXL|blLt6Tf+;SM)&xx+e?9hKlN%|9Iz>%bB2&Z(%mZFq;s;3G|r5PEpUcx+U0QYB%Su#xRqQV$F24f44frkWt`t64wzyJH6KW?{|*B)AGG1m7ZStl=JYrqGtr~)}_>a=gKWuWLi4t%*NKv10STmXTcocM8Kb^>5U75~=G_~jMT^r}gCS%@ji_5I9rXsQb*DM|q=Yg_G|6Ppds1;|9}Jt?Kx@SdNrv!%&8kjq&Io%=66QHQy<k<8U)$_TGUFaqw2oC*+2f6Au82c@E@Q5KHd4FjwFHl7isl>xGMwp97CrL^HpJjax^2d6#L3`kl@xW(sRejbd&V6<rK`Y6UF^#oksq<Jdb<{0~vmh_NNm{g?XhYuWRwywNmI1J{)R_ZUBL0+1wmcK53)tyJ8toR$*7eEG{5QVR^L%~0|0%`m7gPW$ket@^Qs~_dp4E_F_U_*gfDwV=WapNVsElk5b8s-=YzSb|E{xDdB^!|Ish?bHNv95Dg_iG=@9q_EG`s)|B55uoIxCseFcDJb}Vd0ITGk<en=xWtPUJ^c}qrP(UiU*3BvhGSPB}qqnB~!?UiauF7*#(l7cv8$~aD9twR4I_7zVXBnBNqrR^7Cn&jr}EyF->~__oy72086kJekIX}P1w+BE}Hl^6(UuFnrF3hhDP&M@If&LCj3ylq3I`UnbRAW7fSVu1<r)+8gj1+{AnTIl+&<fo!?cNKAcjBFTDWYcnuuG23FmiW6tRpHb+_#$uyDH@i>FpiMOhP%kMA8ux_@EZUzY&;xPtQr5j0w^g2UgcpHFEYr~NT!f#_b=56BCu%_zKOxi~~vKFEyfvp9tGp=KSqL#HGFntbWK7IV@!>5nchoATa%aezD`SIh=Bvg3g`R5JE5-~_BKBk#Vq?N|jw~RKANB+hNmkaklz!Eb#{QtSWr&E$oT_5Z67A~_h`+x|3K$hIdI{c5e<RBQR)i?ppnP1=nVfz2wN<XJ0-~SjD*0bvjd_@RUGa&H!g-?w7r#W-lo<Q7Pi=7A0>7{bfpWi*U=8@H-$v&q#KY!i-^RNHyfB#Sa_IY0a{k!hVm!41UlZPO}=YLpF(4?b!t;ivBMFLAu@5=~g%Ep`8YZ>3aesS4Pc8l$JE?s|0AjWjuUEkAn=|!6?OYr8wvH$v$OR9LU=KB=CURhf=bjtloq~3FO)(xY=rEVvL*Uj|(YsXH+hhT>Q1j5iY+kS84OZd8i-|hFN7Cn51vKuA6mMqpjz*Rf|BBnyy6-h?TmtUVwy1D1Wy)sL_nkmf1x>}xCO#hsHd3<JH{^iFHKYxEq(tP;&ho3%ty~aLXA3lR~H~Ti5dbJ??<Nxxl0Tx2;nWz;CnA0|(L7h48eAdsu@aeah)L8ow^`D^}e2<J~bZ_#?X@LrXj4bHXJ$)V1q{g%d2HRtcd!9YMRWDNUj(vVlXE8uPKSqU`Wtr)N`zkmAC7a>PsOGVO=&LY6uX^V}F6-+^X8i!sgsnDKtb`SPufKlh*L4++`JJ|r&C4I9tFisws&uvz6$<($#mV^FIsfZlFPFdF7vp@&UH!g<M^^HKdgR++J6B)QXpTs$EWmH~u5br)I6^tdAmR1(mxngJ4b_@7D+S4hI38?A5nO#aJ&52-cp8a$J8$$@J5OTLs(|IT3EuZ)qA`&zEW3?#&LIGJ9f8ACfn1@^UUKAvJh#q%pVgW`d0%j8Gag%mFaKgwg`$1Hs;)wr99G;{=uMRMuV06~9$`2!k)Oaz3FGhO)Y3~6Y_GV5N=p2q>JkyV+)LN6i0s)0YZsOsgVpR?R<l)d0Q~--k{NNv;{GU^IFub9Wgg_%;>3c+K9!e^cL9Hra`P^e-B1i9FqMtPy9Ye(*<U!-O3qQ*)GboVC#!<adZ|cAUdK+Hv-WiaOGAJOA|W2s<)GAMZC@3}&8I1q_#v$)dx@f)C3PGlrH1H{tzCPfCQRK}KivOO*l-cXu7e>o?r0Vcy7KrHd75?1w<M(Po4xyG`|&B$Uy!u^Ox|wWug@%r-CL2F5YDE2nIb-5sp>w&t7nN}NVFPcV|U{?)6CoG#6xDdY0+F0_C4%VSp4oL&z^AEuf*|%<w)T(yp45rJNVL=!dgc}^Z)#Qe>xrL!0QSzU5Mec73xpjD|+(QZZcK8#u&AnMC^9<r``|CxmbLQ!yHQ0fn|Twa~WKtm7ep}74hbkNNT`i+QO-Rg+H0h!f#`pzLCA_L;)2H0_VmoU_R8*_Y(TXjJls+#j`L`z~6{mO;PO?9;=FcBkFURGaMcUxoxqwQC<|}Qjc6|nC8;8eRs-xu<Y^HVGe+Fqd30rRPv<`x31PhR<wv966!mZk)huw){3^O_iDSojxm?J0jF1P303&mccURNTM--vd)*`|;QF_HH>spW>88IV;`RqR?gO2hOUS!E%rU*aGI;MhnJ$4*I_b+O6$ZVEG$%>jcZ*wm2AcC+!e-OX2)6;OVgPks7pT+Sj*mJq;SZ%(WHi}M&y~${zz{ILt}PWTqv>ys81;Q`OMKC_;lZqu+W$I|!L>Ik{#~WlVJ3{oC~cz|C8M{3vRoZ4CFsW%X=)@Iw<H$~eLgTko%&$ZpSWXF%-ee4p)M=|ywGe{-=8xzUY-%Mtfq)|%h4@vKZ^QX#y+}kpl0<=+lIGF6=iWo6TTjKK}9&e3~V?x?igC!PmNQLsk5adASI?wwK8I_iXRI@8~Q}M0KN-+11o%gd#r=IDMAllY!a#fneM*4`}w+NL-MHN-QwA181oL3jFt#J(<1~#5fd2A+(4YD8=E&+VmP9+FWHodE2%n7Yd|?HiNWf6&c~ma8deaFb=t=&uaJA6xQ5u)dA8%}4^K1~LF?GL#A)(^iE43XC>~RvCv)cpbs(6ASn+;8jKdKcy8DXJt)q~391pq4?kf;yxUCVkaZYVYX56<=Afc>NAsFtQP;#LH^F*{kNSu|<KN#iNWl<~TW5!Pg{p;<ft=l-4{5Va4>Uy<U&Loi@D#|9A!yt6cxa63CbT);689WF~4cR~cK8p$0uiw8FlG4C$Q#~~F2k5=~FCf6wD&gF|uHOOVsWX}?R|DY^nCLGECaSPNV>*P&(<y*=>TP%$L0J4@6X$?c3ma3O56n2vrjzV;-g-{PqV-}_>w0i8aNs(PcAHsRQ--Erm!_g^Bde2&wRW10ex2ErXcQE6+~IqlbW}wJX`&DbGR=cidA6S#WdVn!rbzIb^j!-bjLCp9YcpYE=02XiyOL~xY4J&&q+9}2wAZZ_DDh2A(C8=GixWX{&+EqzD!XD6!z?WOGNje*@vP>PTigzSf1xl^&ym8R&x>VGpmUpf5emNbSb~L*I<ME8ejf7W0&OpnC8|NuLv?2^`<jn=a>0{BhVq^Hj)riI6CqLd$^n}F0wv^VJGjX!5GqwH_AaKfO=(VK3lz&zk}oM<1iWA1Fh)6<*^bFMpaBT7kG`chzXTC>_q*oBPUI892slAes`?eQ=|Upn`EH*Te_EF7iM&bicZxSc+yvT9PcyEi7)2ss8%&OOk7Ipk*tU@tdF~E%i#W2QVgNK{qUYY8fxX9-1@o@Lm!=}$Z8n2%X#h6PZ5(N|vVoZP(2ULHqwJYp<*cDZuufjZ{Wj{!@3*9`)Bq=o!e*z~*hK;g1rAoM#XC8G(bbegw-2s8`m_HKkmnHxCu+##PwBJR#=ZDT;MJj(h)Y<_H(Z%~MS!utSHI34oN27~!;XyYI*r1>3V{#qIZKrys`^InA7r5p6>;h8&Ls6PDbNK*rZ#u19OMT2OYi#sl=GmOCU#p^qtq;z%HX$fmW(<)WMVvZE)Ceu*w9M}ppwk6*qzoWo6#>*gUy3+vl~tAdD1gv8>GB<0kDQ3U#mHx3&e$^l|qRY*@t0U#>X8_$<9++mVY!!Xjzlm#gig_usl=zL0U<QVYog2yRLSfpCS&|4W1>}#2qfj?1!#z)ObfLbAGLn4BgOTBnP=!c}#K}>?j6XJ2JGP4e|#P%9lfz(J}!;0>1@D8FF91faXRTJMxie#Y(nB3XHW$r_hHC=#!U4y@RZ*W9rI+&32Kzcrc#Tt!|@bGO&%Fp~`0ZK0|NBE2fw#clL3%&n#fFXVdw_XuEn4?KKuBr?F(L>@yTaeC#r1AU>iGv2v^+l``~$jqHo+3y`Vss;%^T4$;+4mU(P0^DIn2xh&wSFfGou|0y5t+HpC4m<GMX9{rrxR`NC{x{qo?LR2<76g7lIjs_Vrt=vILFjdSmBwOFe^Sh?)5}_)kMG>#!YRk_gnB-PSV6AFVQ|Or=ZG#S~_AYLR5WGc-8pE70k_8OcXoj(J{VVK|ZVHqeiK#h88>w|)QN^#}T(Wcrs5e{9MO|~jeKw5tgxJ~5D#+>=p&L9*4T7*qB*<?dHw;1lwn&TRo1;F%aL#m22P(ZNkY}obZlw^!Ggj8DDd1RvjswG>gybvtQrxYS$0N#o8vXO339myK2K1M4(aQ6>48=cI&CMFmknv;#H>t3WHV?YMHlwqpzDg&f7%KqRradd@GYEMt^<h+0VBQwXyDj4q7|_HG0y1^#`a{^`YX7cn<b#6u$+A5R3&ePdsw^6~lUWS=ETkF?Fmr!YR<UdpZ5I`qk!_%Z!|1mnKrPf&EcQu9sSRUC(1;ykX%LhsC|J-0d`Zz&aEN7&K5eUHrEQE8fIb<uggQRB7@b+RL9QvwRc+GJazf$2rc<MxLI;e3?7(cK-w|?323=K2dNzQyqMU~;F$k=4pGV?p#OF64%mvfakh0W7QUiv6F_)qGX0WJZqw#{kiF#iKwF-<*l|bOLk~AOjtF>f%VzRMj*Ucq;HK;X{PM+1R=b#nS*%sttZA55QT#Q$O2QQUq)P3l4tN<3^Gg5|WwHYbkC%MjZb`JIE%8I%y=iu^}zigVOFN7xM&JA!Y1tHEW;RP68U%hjx&I_pB4s$}`AJAwj^`|%vpK@=uF3)p}HV=pTwSlKq7cD+*3))#dP|O+Tv0;=z^aAjTh+1Rq(Seao@^ZNQ3u1JN>?GZYF!cb&J!iCi9!~wo*(5w1mPWGH`D}Z)z&pU90_lh3dTd(kI(96Jsy+A!qV={4I!ME%`UTP335LJ=MG{wsoRYJ8SjiqmIJG^UPYr3t;%Qu5V}TkR4ku?DYnP}NphyTgq$OhILgG(=|86YjUV~_lYy15Ve~Hgx$G{28%=P=vgH8BGoA7ZugeBc>^@Q<AFz}UN;6=%tFtip$H=~QM1s|O!&3b}+HP!)#yW;y_KK_l0PIsdz0Wz_5pZ%f@w7-pw!QHs{=#he_)L*YDf32=f44;8PFG=HQv1FbFaumO}!LIfx0~Q#9j;uRGXh-g~znpBX0j~(G-gGsH-qNnE9CrtaiI0_qOE3@B=BVCncC|i2V6zXT_YV)wM%0>4f|5e{EbNRdo&ufj(1Or7R3?l$>e`}uPcNJ-8e=oBYlBmOG)3_mLyn}zsVPp&k&HsHS~`-aW+3NgIJ2!fdC%C3awtNy;gWOjUlgk_S~2{RNmk12G7gOyQ_%1q7QbmLSPU(^t4>_rOCeg<H6deYdRX#ojjLF^m0nM7&!|MWs#2_V;0@ft&`(>t#5!B;afv4!x_lQEey@oOw)VIPnwat!{1%nNI{5W1+}H960%q%{Z&jY5o1gprAh7vIIPQb{3Ik3nk+)4^vPF70k(?GXm}Ro;F%>*m=B@>;*&iLgF?2piVl}!NYINDJODnD2^}tspA7{yJP#N_M=OlHuEuxx14nfG(gIPW99k^x636M0_UQ<34HRW~5!<CH4D}JLDI+~m@mJQieZ&X6bDKMxdJQ4^(N}vOpH+&<Q>3HgFbbz*>CsIm{J?_#G_j595-Y3|<Y{Q7nivT9orz#__R3xiTr&Gk*fBvT@EvD0c1CK@b_<95K0)<+u_wi?X0c+7(z7}n{S1-<(+WG_}+Y=*-<5tUWQz;x5OZx^(<Cp%Bx^bk!rWA{#N=moL>SZf1ixu#RL->5K8Azn7wNWc~d?C*6LO^xx2kd4|%Ub`Vo_$kw5QE|<66H6vjxowPnDA1CX93eJZUZ+K#kD?Bu(n)MRcprxL~cu1!sl=P<zxN!r;k5<`1Da5$6KXs{`~IO2T^|VJ<s_3<}FU9X1XnYcSMbgGB9sxRQ337A2j<1#?#&AbWo+~;W+sQu>3e2(`gc5yFhZ{1Hz_v^hZ(ov8Y1^{xnnnAe-E64y3q;V^N!2JGKP#y8{@I_GqX)Q-N$S*!i_J_M}x?R+33^I>y0hT!R5=zmW!Aa!W-CjC*Ne2#b6MDaEU=I2r(MgIx2n6(}9#mxn!rI|8SS^+Ly@%9w~*;Ke1DdnH0OM1}>7vHMoruECy_jkOT@IF)rAeX;zj>Z&QKRiDiYg}zo18lBDH?iG65*fnx~aigxa8H;jdONEH_S#?HbfLHd~Pp=g%qQZxaMGav(QvAz;gY!PQIUzikkxby4iV|iQU01ikpWR&n&w57Br-yD9r=~f?KF%SIMcp*nIXoRj<p9(%O;3mD$*ixveHgcasfY7fh9}l^vdcp2E4&dB3vZE-T<soUMH#jg(8xQ{)2fSQ%(86Bv8c`_K~GHhu<tqp`v(-mpkM_+D&sz>A-7LJ*=RUGu<7fVlwu^_jZ**e(`mNsQymXW20J|ssqNJ-h(J$~M=VRlYb=V_+A5sM0)Xom@9<Pg32ILqPV+xC5>=LrC#rZ|IEtaxX4$K#YIc??qOgGXiCok;Er-g2Vf8DMVXMZ)D!BZpZkx>7GMO6X@&wPpxf%x6^Qocmx28HV@?efo;EcIzED26fmYA@vSr@hqw$F3U62prbbc|M$jMwL(r9LuMKq~4z8q`H@gCBF1ktB=7c}dBdjzHWASdiZr2eA?T>MCmL1~TH5ya#GsH7W~R)d~B)WJ|60^EpxcZWE?Vou|swL!6Ddb;CZ9110ZQ3J@ZFf#%BE>=yP*_<TaO26AiwA=tR*OcT2MaBeADx-(OPVr`P0<GE|C^8~(vDs=#aDpKH(nMI<!KqQSg0Hb(N#u0iR{BmI5)FECNBy%h^O-2~rSAERc4)-5?h9*(3m+^idQAUFI_t5j~-_S(k->p85i`v}tv?zxMeEPUeAuD-yq*9zsa86bdli}9O+?W9<Xpcsf?KYX`8+kk2A?F9ov(GheYFD`z9IUTfd-s-#ih4pLEZ9bGVzx8>sH6H;ThVfQ1s>FWD?DgXFt*S-n=^n={0%RlDjrdP8)awMsF;O9AzV9xGBNn{w?PkwBpPamEy|YAHcYzH-nfkZK;zQMK%T)?HHa<|Fy2XQiWu%lfG32VB1GzlGJA+?K9{!?Me=pthpWN@0vucfj=j7>1$!)t<5jYU3#nis5=#?X(vN_dEs4M^SYdz)b0;*FNDVj0nrc(R0Ec)hnF^uh;Uu@kz6Xj*dQVnCp$@!j6_qOI&!~O~tHPhjDOCpoaAQdBmaR<|gI=quv=cgyH9tXVviZH|FZ#w{XVaRgl}nS9L|r$zULuC;nk!$6AW=m4xr7o9s5Rate?!0$i~TuFZO4_?`BJA_3?zfqpKYm1*3Rv!lnky7eR+Z#Ojg7@$_i|;6j2Ps^U<AEl)wY3LEJ-t=1*#-$cbP@99?0ZO9Jc%V?t+6<Xz<jZf0fzYqL5}fEEPS%{QH7?+)5qs)gm=0AxfPeUioVU?cEi|7+Z1(rnNf+-)`KC8gIg6=n3O)(mZ+QC*0srnDC2huw}0S({F(WuzL^<G(Yc%6MU4laWU3ZBhzKcJ$qo;&9&<26~@Yr{Z2u%aUqZ91(M=4~s4r-!l4jm?17sZUl2<IA~WX0LI*^|G0$(Q$IJy^>&wHN?@hk31z9!)Z0bwAsHBbKph~KdZjNFJr&KR)9`hDtbYR(d+Jx4K8DQLq%k(1PBmwgqWTQ3j-QllSPfNM(J@X-gD8@$vEWJetfrt#46;RnoD{o`%V4Ht7i-FhNO>y<wD?SURoJn2oAr>Z%@wWf0Ij6C@6R^C(GOe_D)sVFFq%s=)Jzrup;!djWxQL~Xd>VhYy(!#5rZ8>nkHabm|}Baaob^kzY&kuY$F!0_g8-L>uQf0HBm3{+TJ{O0N*CCCTyK1L+unyEUjq5vy-eir~L=3*!!U9I&=d~5PeD(rgM$91igd6Jd9!;oS0*u=t9AFqScty>ung+E={t09Qx(J<in7QI|bEw@79w|`vgWi3Cy}JT79Kms0|p!)NI;yWnQNLh={c6+uKP<ywL+`wku@s+@>uA2NohzdOGE#&Yxi)xhfh*b%Sxt?N&+?rk|A0o@5>Pm2`=)<r8Ro(p0hTl&t^0lCnvzy}`<m<wlmZlWP?R4?t4xeae#*ii7y^kg68tstOuCCUfoMnRUl1Uk^at{_s1ZqcBYeHP@cv&*?Ql(9|Tf2F^~2{=_i!;tpKW5~pc&o2S#zFyB|!`U^udew&8MCZNiO_(|B^D$)bqPufYP;x+pxj<Og9BIJqJ*VlAz!251rySojfIja-2v^D9Prp1+=2$XK!gdpK<;5hl~x)^psa~IRvCP2X>NF^y#JzL<T&|1EDLbFjq)}YMJur7;hsa0pDxohAa-<=t@O%k9?X?2E{35^uy-j*)2dI=}uA|-O@qK^CJFb}RAlhQ(eL}GA2O;LK@Rh>MG@fEU6EX)B{8|NzdRZde;)-4f#50elmp1MtV7P(dVYns)y&H&%rQJ~>F7C*r(T&BXBp9{}U!TXr`vEUAz$40SX+IP%SmfJlIg$sdNd^5oj0GlSjDT_jI5WBS~d$1OK3EFvreF1rc?^8_lO>d%Jm}-l%ug7z^x{%FZRFTJ9Lm@iq8WR0p2v$h(k=#*NRGT}y3C~O5ip;496#?oUbsidbl%Pry&2A=sLA5?NqMQb?ZhL7=Fa(?Q&P1lq<gBuLr#4Dzeba#sA8BnjbNN(xZ8R4l=1N5fz7gN4EKLvUY9Sa*ksV8)tq)0@#+NDdY;MKM&Si_cJK-Rx<BiqKC~qZ%ml#@emM#M&D%C1OQQ0XER9zr!&+9L~E}L|Dn8r|3+tI)~t4i9TAyD`;*?f*B*$93K#xD8t!q0%Reb7>py&Hlw#xf*|5W$0QZSyDGdkWUa;nOlYrqfi8z+@2tBMJGxxv5OPB~hw->H$TCfGzbJV0X;+X(R$&U-bUD1?QT`{BI9L@Z6WPV{&znX!Xe^J~?~lQO0~(e4DZ%9&%Jst_d$W*ap0+^s(v%gJ)6KJ;(&PdC10wvYy(`CU`!2xqgIN<(o?0dZg_W=@M2{rbf0e3$}P<KXoMPBG!*CFPl^Emh{P+%@gA%I*kuY6uP79kn)iN`m8T;V&_M!ZC%u<Qws_&QVkwFXdzZ4E8FHNTw4^Ti}MK@e9<qqVxvA-!_}cwY3ivvx7jHDj1qg$b3#+{q&140(e<I$vn`rzXLaQMH)WUHlDjHx9r6;Pf3K_N8B6YJv^H`M5{u&cm?5#zoprU_5+}Q23Z_<V>RK1Fqf?9Cnd9ltCm!Exi|UTxSo5j#39krDVoLfP!SXr`RISlcGmSB;exHBuS2Iz$I7?qeaNm;M1m>dfoerLbEkr^46`Zcya7SPN!}n~67K82(7nCdpz}9@4GQ#H6Od0_$-hAMC$;1J3*$=(E-Qgi@$Mw9#CFD(lRX^J|Ad>}1>adG6dR$93pB7(dT7<Vu726kLH(4EwU$>jiZ7<8X{tA(t(tVT~@^zF|uMHwl?7v}ehVJA)HRj-EG`Lwz4I^oPHeHTXRBs%wYX^)3B1vn1^eUJKKcC-8O<g=PlKNeUS;YQeXVial>JZvpKtYO2*vsy3?CJSpvqD*rawvgg7MdpuSEhZ~#^rvEcZAvhw8oEGsGGN?o)cHi{)07^(o~jnNE2a?^G_Vso4aJIa)AOo><vtx;(S~BHzrr5>L<WvFMFTSi9I2cWe(-)sV!U`p`@S<@7%3Y!2hhS*m}b>QKfwroW1$_liy@Gu9bh2jiRrzJCc1g_uXOFRFTxgm6UOh91`*rY$4JJX6|K`dz{FVCkV$#SqDPr&Z-n9T^*^pA6jYVLg&itwVHOEqIrz3D|(z?LI#`r$;~}+z`<|NnW=`HUxAiU(W1GZ8`Qsm|3)>rx%9XRFs5d8_BCNH2B*~&Q7^sY<MUYZjj~(28S=*!qB3q>=v?T`5y=3NT4b6Tbik&@{SK5AqHwlk08dAujS`bklDhVDp%4e$PE(-^fN7a4CP!<Shu*dGT*nY;>JT%ra2`3gn;8wZY%29OSi%B~(tvq@`HYRieN5eqLT+sCwj%F6B;BSl`3-wbm1th5Ll7C?PUa>VFdt9z18hO;@~Ey4wGV9P=d|cd8Ur;<TK;LOSv?9q&qE?>;5kty=x1gJc2<IT600pk#2U`3h?q+MdEnn;lge2N(Wwh8w_CWPMl5`69|K7>I)}ohOHrMuUz}`dY6XiF$KK{PFuV)WH=hBuu8+<T%6m%BcEz7brWmAb9zD@&=Z@M8)n`vtVph`_8#<`r`&p)X*umo^iVr-lsc|o}4e58D35snmzkW)w;F4Q9)#PF>Wcv_KUI`IW(Mm^KmXWw?r+Abk1`klX?>gEQHCAbDYo0xiOM<8N+KS5pZ7W3hAdp$2L8A)!%RiXIH%SjhcBOrV;v%GG1eMF>J5vKz_c(3>s!XR`uMti^;!UFI&A`Z|5sem3sDpW@n+wZD2cjsRD{YI8p;wQ0v#Oc(u8><zn6Wcy6_p*t5A9T<=c<~y5DWyR&RT<jE2tRxVOERfY45Zy!|lQPqzR$NTBTS=i5)_VCGz@oB2!)%3tXt6Vf;lG*YO><4RChluDlL$$y{b4_TI*OeucVKeH^5=MO1(DU~&)nSfJrY4uB<37pk9pb!I(^dnYyh$c~P9E9E9w8#aL>s*J179z)SC)8UF@IrGWgD700R46`hvX!#QkqidFV3_lE_-5AYqAu6k$=ib+*5T|wx1Hv;s%met=Vekn66j@1qLfDUS8Er~IygduSK%&hA4ttwEr&3*n%MVOsvPtl)9!zxsv2^&s=wtv64m?pRz=-a^eA&66vBE;O4;>ZUE4ao{0w<<9tLv8l+7kGw-Zk<y81w4qkSpC>mXfD|r*(%v7haxDb-?`f-S>G&89wL4cOQKVu$G%m*QRb`H8_;DmyR4*ZaQ25f0iO)4+X2=0;)R80zFm|AA=~IF6Q!B`$wqgcD+v&qOI7=t<4iX0#kfopGSLX_fw@hU8oRHN}Ydpr1;-*CE@!-Gi3>kUn+%5iE4QRUn`i^uup5Y3&@1Wdmr1Me*59a@BjYik6Q=l`Z+_TF^|thHvxNm=a$FE$`>12J8>F(YVHqFej>9Itys0I6C?;=7JzSG3k3;f>}02k`Ml<M)xewewS`&vZ1HbK83Tu!3+2IQm}d!mAR4DOX;8$F!hsp`J-eum1>CLU^0z4JLgis?8+Dt2$`bH4Y7Koylrny#gESZNh0MP`Vp|Wu!dEMPDif(67+h&oS^~{yZ0HLklTK6-)s@f$>p$7VQhbWhGA65=5L58Glg^&r3Hz1Ji$o#>FHzMrkrL37Bm-9a+z?F)1_o?`!ufL%b2lR|h({%*2nb)@19i5Mu9?Too6m=3b%MAWJ>)_s5R8p|Yew9jQvKaKavQXN=LxPN=F2hZCYP$cjkM47?E`)4pYtUf#sJ#G)?+4p62Lfzj93h5qO-Zv+dy@fCo*jFrh8tRXRrQ_zO19I+y$wV57ZnNH+uThVA{m+&3R3UZUOf;u)l0!r=zTY`!M_}zhuFik6*_on=UptX1qC1k5@PA*4SK)crGQ?ST@ZlOb~78**#bt3g=Uok@v_;jf;E2qTUw5gxkRTj8LX<S;TM^l`6=SoC`fP+P$1}+2QX}F<v2<vJFrwpwTR+h}3`=7W!6TY*VZ73|=ERSTMw=gzzs<(4El({(Nfl_tS;|{?a3iJl2_<Pv-{LkOnE-9n3ucrOg54*2!$rARv)e)<wz7`Lt%-LZ$u^l*uvYm1S0mY3$?gPAPZMD(ytJD!g~H;v5n_AFI|mcb(nuj848dL-4}fKIEJ@btwv!DW#L_k4%z9%Ji)U{Qv6|hke-37wfiYFxR%Pr#EUQmU@lRis}$pIqBW7J{8Nbi4l*u_V`<bKk3}(?A+hq8QE&62uQ&^x7pT9Q*XT@a+B05x*A<qFUPb`YZlW|<(lAWvee@3<A^e7gi1LJKC5UvHcl9I^H#Fujp56cfJy9ASlmqruH@Sb>|fmSfwB4N63wSL`)N)AQ3Vk!p5FcR@uv@;K9=tmB$!7#xcvC>=km?3XLT`zN9*`JrT)uohEAzSfI=T&yOe@eQX_^isk(>yJe(g2tEev@)YPhyGxZ^AI|cVXyk4Tcoc~v62yd1dQVC6->(D2JTyEQdam}x8?JNNocX#T2h;70q=BrpglvY7*BbzC?Z204<ALI6b!*KR<jdlGDYAeZ{8T5I~PPEkb2PwNyJM;X8qPbtKIIutZX>@frRkQd<ogpebj6PZ)Hq+byATbA=Yako77P}!%tDidLuM(XHlfqU*QPBdbggIaI9`}92t=(;kFvF~RbAW#3b;}-Bx?5Dmi(||uch*UIbg|D$pM$2jp`QVJ;tndD4ttz9tj_njoE3WUyuY7Qd$Yy3q3)NjN>lHO^N`OE)vO<I-hf&1BDdQTku$o#E%{_)skKWZ4;dv2P1q~%!z+k2ysnuyct*}Rzm)BU(d?{nG&EOutnrfq1c3W9EmD=9ri)11_)L^AQ&m0Pu|%7t0>z}@uG%ZBPFIKj?`hsv`mi<M%DSsIWMv$D79X5mA7x_rDSzJ>&gUNeTa8l8FR|=pUm#W9k0ja{9Vt%BhwlB*xW~JF4hGYikpqZ?)KyiQ(e^O)_8irf#{730NhX?=OMX7i#Eqj;P@qX-IjkTkVNn93)I2Hvk7G`(0J|;h;jkkVz=MT_!5FoaCD27mPn6SfTpNT!dCPi0PE=aDfiX=92<X`|c=r#%MF2;%FKYUXVvSSZqAAJN-0jL;NvXW+J|G=ou<|~#m8-g?or+^&3C;ECLFdYCE;&ixM>1#;_7v-0zvH(x1DnLb@}CJL<21D~vLs7ORbn^vm)b^r&Y}44M;9duKvzMO7sPkM4JIi)Ckr0f8{eL$oGRJ-G|d`W11QKk*;ca!<Qz`j+;CEm-p#Zq#u2MfOuMPeUGrH<`nL!#c|5L`$VTguKfi*0p<-oIkx;3hR1?d|x}iihN=l?_>Q&Vbue!3jGAge)^z!s1YVOfCt-0CzY$(PA6JUiKx)^Ti&rOVaNFs0Q<x>*e7!J3u`ReCCjhkl-uOlkhYKFCSKkf%&MIPsVijt9J=lc>3r$pdv<JyUw9`eRhk}2qZF4}E8gS~^a01%^jxvT0xs`pWmx`hia7?G7`Xtbjk3rvdt5Bpj!B7|gkJ9Yl0dS$eRQav35{>6#NltW|rek@^0G?w9AKGu|+{m_#55EXfQlorH8wfgfUn|)5>+St<<T-;vEt-6$toT^`3r){Lu$<<c_UTgy^hqv1TCk->DeGYv>@H)H?4r>Ta(~!(G44T?u{mB^I9NZRINmpwkdEJ*{bQ8u5%mh$RT+`uSi~CVt;0d4z31Lm-f)>Ja>`%E@w2gOhn-@~5SE!Y(yl8X<h~`+uVLo>WB}%A%TR9*NEs|Z;t%WMbPLdQ6ongiqX-eW^mm#QjpJsJFwFweiS`r+MB8xoQf{F}qv6-YV;4V$GBv^qGd|&vH6~bm(x6T2LIckBu$4F9Vk#E?9co3(&1quJDB+2mdvQl!6HG7vsLW{ny+BvUfZa|;NV+tbBbO+WNCT)wi&!A_<wkW!5Iz^5F`=DB?n4|%}yFfGXN1nX`I}Sguwi05Lmm!*yB9{>I4C@_Kh*H688pS7gpn|Anri{QS{i^|FWS>V{{=YKhs$b-;u|n&JgjrOJ*cKFY2pM$USw2o6@5|aIO~j-zYion&bP({wV_iG=TyA3Kc27rIk?uUCYC1e4N_#s_IpLjW0Bw;uc~+=kwZwj^C7aeM0|5qLpw|Enj9i<{Q|GsMi?8+;$M96BqR$gNZnHiz@N>pkXlVqG6!5kO9Y~M@GHS87jekiq$oys!ET|x1hS8vz%Uo|$YduO2RALz__a|DMk%HdlBZ%94>XL|CANwJKSOLB2<E1Bw4slOCh);vtwg#q7DY8Z}Ht9Xir|A`4km|Z^=_sNZum*76iBFXHPIIfEFx9EcGHY!|ZX5O4&0t}ixvD;OZHW6g+u+tj=phpfylLLE(@)h=g}09kW=~hY&#UX&oE(QPs3ba*&7`{CSN4wLBB8|A;Iveu95M@9=bUO164hnNMH%h^QfQgbrB14mus-4Z+DAQzC98aI<kd|H0)?j@7d{67gx2j!wqecbtUdGP;Bdj`YSSY9RHBA;mF1t6(ICNGR)ckoiT$usxY-n@R})ES7;Ds+`o8QR^&>g&170&{jx7df3PMF9#{H=Q!=MuI=8;jeH|=Y38+msy^30qTU{cDtVgfT$VBC@9{BA>iZ$_)S1w4&5AB1>SBx-JaV0ptoWq8_;G}WSm(&;rlrTWPXlW8Lcmw7p|+z&b995Q@1O1&BTobRf|V$|!!DPMvs6ZIJG&;AervRmQOaY1ZP)r7#vmO~>Gv5`|ULN}S}n|pPo@iRS<s_LQeY#O>03PG=Jlm>#8nI=|YUyucHm$znQf;w`IbI}DSMuG|L9i)rmYP*DswHXF}LG8pdg9aY>HrnY>b_l6kQE?>MlEaHSZ;L(qJQV9}Yj9003hhmxh{p>>RXYfLn3=Gr5w<@tryD41y$`#P&N4ftG1v^%=0InT&BivPv9pVMQyG5t7+e@D)cuWhjkigiUnPR|8nG;BIO>X}icuLC$@8t0){S<o7vL!t8Ylg`yP5DfYaw0KXibyy+}07W9E+yO6K#PPkml4hy;>(YYOeRZ3trRp7BusHNeA3peG-1KwS-HLR9t&hTM~tAYo;9^ZBek3Dhln{Q;-Lug~0Cq>Vp={P}=s^b4J<#@)Cj!`PV7$BDbfY?GNk3A-WyuNmbJCuv3s?#Kh{+-@a84vKTu5`+EP6(?uw*>CE;OvJ?!<o)dgyyFEMnB}%lpue&oWxbFn1|6nzbOlP^4+)me=cENEQ^<#Jv=qX!EkoE=G)l_Xk>fi=?*QB+^UN6jCO~2_xy+HfEnOvoH=RgV?>GAe@@&#A-cTTQ{Uw#)qAHIB5HwsbT*NHfPUE{8h@iJ*dj1Ot}*Vazj2e^uDL4tNlqIK$7x=Eco&xSN}DAC6zX(K(hyPhwOk~8}`4*vdjzkK-lho3%twcC%^htEW!f!l)guHMa;Zw*eM+8v^jNWq-80ZAY3Prv=}<M)66{0pCcs{mlqG^74A<k$AdXh!$ks!l1GyI>?2lj+(`s#(gd=k)g2xD5CHR=r5YJFaeHHz`#SQ%0=Bd>$MO;b8;q_ipyPhpHqjsq!+U%e#5j4-ie5Fm)N)TvL4?zt>;CgIJRS9LV}1Bb%2d_HOg-I<I;0aIfB)_9}UfBW~yXuYbK<{&ruCqL@)${XU+S#ouUs@;2Dc&AG9rCE(wj-|j{IqUCUeeNy5)Xjgz#l22@TISpAmHbkWvtcDO=eYvb$@FhHi^CXhdv38zZq2y@1`zCnblL->K9gA)wouU3wWI@3?vId#KpG3HTAyJ{ut+TJ?RfCfE1(!AxUzGmxFE$<(?E}`kEz}vNtvsJojewlnomYF<UYEy;uT^*{5!a$yPQC*HuATeSj(OK9)!!<)5WCz<*RY7}*#>JDYW;!L>|0jT*%1~!ty^U8YdGs-<5~?0Uil!$7AKYo#_?UCRK&#QT_(H8rEdKwZ7kkB;Bn9X!qH@bnII3~Ir3!BZVOQ$J{g{EaT6o3L`y?}2_hjL)Ww)3y0nTa_SCAdOc%va<A=1K>?Nw3v?{X#@oqfIZ9H8&EFEf?T0h+XQP^-1#;!)e5f`uqZ8r0-Yy)0LTivXt%bB-w7%e(A{dN51b<H+@eI|ORU2jEZc4seEgs1z|_90$9OANzAtBZ6nYo!-suwG~hW25+Cx=MZB>QkW$ueW4fJASVmDSTeHv94}qHjN*wbqpl1`qRl3o1zBIJ>3g2a<)SKse46%c<mCKA|Q}PDXLF_i`k!gKLqC-&KBo5WL|@1f75fZ<piXce0AGuFeNuV;4y9Cv;@+hOlIM?u}<H}-gQy{i<*G^>Ne6@Kzd=jChq=TFP7joU&XWAYYSw=tg(?&^lb=>+fyPaK_rf^DLSpT#oE^GL%}Qc$W=XGE;ZYCr@Xhy9xv<4A!GKwQ;C*3w7OakSy3Q@sHg8#_Jw|*SSzZVqN}6({SoH!G~o2gEujh(d#$d3$P9zMZV~}-{oB5qRDz;((_a!@`vV>KfzAo3_16X8eRs|Zg$|+;R5H_349+TVCX&xi;MRA3)I(#6RCw=AkKWXslsRQ-V5V4#G8sB|-S>*XChH^{WJ}GoC0)tBYs)JK;W|^;Ob<MUp8CGGdkhm7u{K*4T4YvrkR9Mj%Z;jPlQ`z3?+r|Iv`v!<!zz9Or4@luO^tkphO`BEOKN=geWJ9L>t}LVt*X_{O@%McEidGN<XQc`5F^0l(HEz`5U`f0Y{7NW3q1lNaLPq`^NlqvoiK@el+%#~f$CPUVhq3)iD|y?BP*Jaf%`L|q0ge?ur@1km=2v?SIL`QelU2+VZb)@3E45eaeSL(A{5N>h9S{&3CAa5W1)x+tY(sn6wGF{4Zz+<Nu!O;LU#CJ^iq$|5JiDtG_x<<R>#6p93mGYM^XHcP?yVu&*(mOmX5~FMKeA|rD5San8tKuNuR%H8>+l+?lKz<5xir6<LM7iofl#J*tt~jtJ4)g|62<+T%=WZ+p9b0y(gJ3KYslAYxWjeVf87qjQsHr=$Ako#Hy^?zOElb|8=+JDNluv)I@{tYs5KKzsK7eu_@2{ytEvvB|ruoow`N{weV3a@nYEwH*MmV_cSE}j(QIDRq$w&*HEnHzEr4cK(NJGQ_;tt54(Q!lM!aslLg5V22aZgg{7MIoiZ&x&xJ5ZL6KiAPfqy(YKB_nV-<F)*mzL3x8%CV$swGp21)N=rnKNMp4NPF>-|vAklOl?-qS<Ziz;OsEf()hL7&t-=9{r@ySD42qTV6xby$ZX6f2IfnR#lJ%^%#X!z&yx#qNHue%_`uC!%IWm5>Yynr#6>&cI2C5R;<9l9VDr-X2p$uEg);{1R-I{X&`-8!!i|X{5^;0)St^eJW}?pKqR7k)R~MUH}Ef-}y<KjGEkR(57VdXaegDCdYd|F&QSdZKOq>yCcO<hjx^wLaU!mZ_mKq<H~}0bF52KRn9h>!MC(}5Xp@rjh5&hi6?dQ7L3i6=<k^5DeLt3D%Qy>k=sT+`I5crN(}&kD1^$(XT~AYd2VP`iqJmR;P{%AfVM|}_8$WBJPg;_#<JtQEij(N*Wx2B!Zpo|6lrlKV>{Jh?C(T#J8-5kRzU3}9Tl=J46LZg<({*oicpyrdjB8`brSrGB9_2{VrFP$WNLHA(4QMeFTLykQ%GNDR&319;EM-S8GN;9nOhEeWyi)-H#P#c<0t%}71h{M?oMm0bM2R@!REoZ*^Q?5eDR>b;>vp$z&>D*S4)4C8^eReM5C3;MpiD1rMHZa8!&jcL#E5}kIsQDYf@}*QsT)}vpf7CEeF9c+|B}CSDUP_I5Z8dg;Q$C;c+k&lzOmkiH<?%*BZ&t4J}4;klCEaB)7qiVz9L%1DtLSY9yqphAvZl42I-29tuk+!GPvQLKpIpC=b;3cjK`(2@Cm<0e$kaGH2)(m%Gaw*E4)3=7)jttS*0-l21ij%dnWy^nHfjh*wPYGw$p+ws=Z0QHO#S?-8kduz`8X2e9N)^LU?3Z(pVi#HXDwLwCA1=zg$~eL+HG{aq+gS+8f9gCxs5HkVQqCZJpvyH-TFG{pX=e7I}J<@jMTC7bM7H0K2pyv<2lZ<Xba!1T{<O*t}TN>ybo?83XQAlx_Bjf2~j5CJ4tSFhq~*O2GiB=L~ITGfJG@0nk9+`<c+MBlj!u54bIQa7-G;Tp{_R<3`AJ<?5qawD;v71eosQjQ)<&V9wS;Y=s(8`)?gKQ_bGN9BDsjP``s*&Pa#!55($JWLG=Zb>94#D>6dW*pl`1=Zgi^%({kR&iw*7|1hepmevb$F*5mcLl3I&o=8tH@kjzFU8$Tc|1{*HqAa{{`t^^*C7l8`pZ}y|9M>o)mXUk3>i;0aFYt_sM>!iY5MR4q+A-EjAE<+T$_^4=`#p<E%jkE_JzZ7ZyjvhmT_4$=(+*{nL2g-A#6&qf7dp0+25Tk+rzLxj0ZJG#1hyY)b^HO+bl*~jsW6q?t{_N&%ib&%Qn!#;T23UPz$vIiG9*hmRvWK$uf+9)`r$Lyp1veUs81S%A3I*w^g#zHpU4+pG<XN9Uoi_DfMlTYszv}Uz2o^emJn{)M%&B0iz&0FdON2gq)IL>Xhjck!9BVJflAq0iIx;`#cg)BR;=lP%fCBhLoiyk{U4li@6NdH-kkT8;utPPSpD{s8wKesssX`l|(JvuO$Moq`k3b*UcqD*wmUy>n*QKc7aw*XIq<s>3-9itr)Ka4_?+;tNYOBSOF}+XQT|%YBN&6Pja!w>>TRRmGxd(&cVI0=nJ8VxpM>Dsz9FeN_YW=Cq8jb)p-HQx?xVp^a2`9rEL($;ZyF-*5!GQ(dOY$gB9?!YW1t9Z9!Y11`V=?d2ARZD2xZZA|iBDdvsuAle`@6{(=~tB0EVpB1}DiaT~RJ@YH{tO~S)rX(Vf%&$f39yaOC6kOl^>$41-C3kF$HwFe(TwBA-h2kDSdzaUC@!SGkVNaE^{Q*u@h3$3FFr?#i_sUgi+JdKNMEa<$nI62!`yF|4BMMB6SEfFgh5`O~xcVjvC8bo_s+wXt)OMDhP22NOJuHSziY{EC%gpbo9Ea`TuCyYm>ln`LxMai5nG@6q0*#r#t4AF!0q*+gJuYw-|?!oQFt+{k0<=BnhvUd|(_t`JnV4b8b_r|24Z6lL%UesT&DSxf5O$?ubK`%+;Xt89T1#-kPx52LVDFYT5f{p;RCq|Q`Uo9tFYrrc4Bh7swU8YTbbSO&JZdL9Nm4!<%57n!{?gUJ+^Tvv=Vd?$DgR>E}<dC4GP(BMgBa5d%r#rMDG!B(X^hr<fM;1;Njj@^jW!Ovs(iFvO3^|Hyf?e2lB~gHjXh-tY4CLGlN0-Lc8GBI<MTj<Be(e2=K)mi1vu-oWBrBc9!XPweOhLncSPu0d5R4q-#04mev94=E)z0*=63YK=E4`|Fxg}gxDb_mh2JT?!r>$LLovj{s#1jr(zKaUK*Te-|dt3xfOnD4`pQ&LT{Q4H|Yxx8Lo`9Fxx1;45x?8EQe~isP!f_wmR~T?wiM(wRlP%K2iR849!7P(ykE!6nGIuTLrTXabjiK{F606bGP@~I!U0P}Ft_Qv<`8Z2%BW^JEmgAhH&bCEVGsqzTqr)c%^fKU<DJMYESbI(RP}G#yB@b6JBCq(3R_JJQ##nY$RSnJvC8xlkmhea*2q}RMXx@-X<Bq4!MhC36&34(@>m%;xWX`-#uz%Tx5t|nQOsZW&Mqa5%R-I0#h_(OxPfuD*r~L*Vix#l;2IK__wUq4R&-4P;qP2W2+H$X6oH4ca2}rglMij@bR)?liI4+j<4VK0){ULSZNQF(%ZNf@Qx5(;cD=>={@QFkCe6Sfvq&1~cD|dV$&hA3MhOH0S&6?|V@eh3VP1QjRila!B-@!7*DCc0pOBJ34OtV;#ejHAXM8Vo}NmZ>KBM`YQVF{nV`InFN+n+xE^x@M-Z5(fvw)ykBUmryI$@e_t^P9IgnVRXg_}vjTF3P~XrBT)6yM54Xj}uRKo6|v+ribI?8^H48a7?F3fb9awi4O>y-q9aL<;S898Tivo{ex_B-oVjnQo4<B5{@mw{O$k-q&*rc&r~2A40e8PjeUJoGG)kXB5MT3ZnRawfV7Occ$ZF!BbDwa828e`5El6iQi@k!aWnwj2D#>CD^NPfFAsYLcLYuu>xGU*l`#>sz>7;P_ez9nhztuDWB0ALU4uO<8*3r*aVqOL`eONTx54LyudipbLZPo!ghppGxO;`(Hg=7iU)-o`ZN{Qp*-{~bn7hj3(7p{q9h9v7^jgs(DtyRT)DWg4#lI{#IPa6obnwhbCU8wf3A2N<0T|ZV-4*bxXY_n}=w@+hnnUd49O78iO_QC&(@|6oKpoTcbcmkJ`r6xvaT}O=IG<&BVofKzEVRDD8zHgq775AK?g3VmVOs%>yc0dGx>&|6%a$CA>TDA9#Dov~t~0QIKrsvoRsf_j?vomF`vjDYh64nfzK%&LM&jKl^)El2X3IX+@vvkt5l*B1nF+{B1bT`*Vp%F)V^O@;R^d#IJ(Azysgx4bo;aN5e`+MEEE!K!@w#vnL#@rSS5ejMELB8di8|K@&cd`DDhr0yuS|xm8W*eJ@}s(KGHc6ZYLv?pJO}4$7+BAzhQi;P>cq%{IYNOm=B}|MI6+xr!n$T%*f!Wc&pAsBFJ{m&T1_%upNE$E$W#HTsP|}47r707%vDB`EEeY_C2KkYaVKCweqS8KM)a$zsHq#sh*R<&sCCt-ENoRL?E8`}wcgL?MDe>#m@;*qDpL<}Hs17ko5+EZ_bUYmk-k84Wo>o~`z3rnp;`kuHh>Up+;gT0-F-N>6fNDEDM7I|$<FcIwbpq8UqO{R074ZhaLCLeQC=XDMjU`qJSgJ`Jr904uy5)RFAS17mYOCb4DYKx=4^-i4?aVasMpJQzmF&*!TWpYdG>E;qVexmAIC*)?s;02!vj8j+@_F~JUdb;&L%h~D~ZW)>t$}t02H)Gqsn%h%=3-B9qy3x1LoQ1nm4tp+zSrY*R8#KOGQOJp%E5rqc<_z8GqDKeXFf#IlTf8>b?~ov?v%`Xr0X&Kq&r(7f=<CsK1S}Gi+4MLZJ|@9YL8GeEQp<heHw#wZj%=OK2M=-Dz)JMt`7j>0}_!V5=HLmk1c|BsN71cO<|Q!cGw)bwrsx#5JGGTZ$t2I`6|(VF3XSE&|71UZH|L7RB)@*~5iYun>u*i7n|zz|59JU>2+}K!v#zno6XG8)QwjDPe#^yp>Fa(DHDS+hX4XMJ2r_tDsN^-nEKKmGfs*KZI4`&*YS<0|B@(BzMc!CW}F@RaM#voyVG=pfuV1-t!lIW3aPn&D6@JNlK!wn_Mpu!*$J-uSJk3BK%xJ2?x{~?~=bEV2Q>49HzG8O6z>7(=7&)!RpVpR3&TYc2!CS*M`14!3`!W;vHoLwpfZN2IBeX&MHda0o5SxAwcsdHB;n7up*AGu+Aj`_Jc8@Gbi${@&Y$AGl8{PohLvG0_*0RPO^6g?Jd>9a&G`KqK!Vu;(4$Uc(MOA?lEaL=nU?*n)H&=Ynh5NdQ@wMHqfXp#8gvSi}J&6M~19TC)F}i4eIgV8B%4uu&>ETBlb2a1tmNB?n!aDZwmvx&#O~$ucu{6H7$;axzvY6my2&1{W{DL7biD@xiK8Hs}ulZZq<L>!h)%vo8x-BOED#|((Z(^RA}n$BKMFCj6R?a5KFz%mx`W>=F(~Sx<1yw0g65Kt4$w6W^B?Jn@^{jGfGi?23N;VN;a&9s;%f4r=>v@N!D2KBzsm<&?N@hB0)}yUB_iGQ?iRSWkjUBl>=IQro1Zb*t^Yo$kpbG)^>na(%koF8{p^%E(w)-`6w98B^qibi-1rpg6uNhEo(Fp@Cvp8E9Z#84kAqxuq;fmIk33xu)p7k$7{9`i`V-rzxZ{vM~#}O7kF)No;!eV6Ic_rPLrW_3MQ6TG~wAvR-DuRgH`N(P;?!-fhLGPB@5HJMq7g3L0}$6u?|kmu}^fN;5*T3%<A<v3~HAqSw0T^a$xdd$i<z4>b!UB$)<e*qn!k1-4?CB(k|2njACjw?Yc5A(|<%nTJ`PiBqZMGfi&9{vUhIN7J>r{ktsc$a#H8du#a37jib83IOcXMr3uqd%4biq4*W{GMA-5Pv^{C6Sa(X+e_u)2B-h?xWyo?P%i77cih~CrDfd3*Neab5{CG%Ji*i*3jUJP^_VLWRW0kK5pl*No9nn#krh}SmPx0sU8X#zD5?TXir$m2Z7<zFBE@_F=G`h{xX=s@5D{K9Qp&7qTLuC_CWkdWV>~0n50q-a6q*C#k{S!x7i~<qz#Ov#8Iyc~bx3AsZ2GX3>30m5kbWPLZ%1#7Iw{Aj^@HTLq{B>OnJE6IYX>Ajr;1Q&fl&PLA@KI<jUp%4NC?RW5W@lKJ#kJI`Gt=BPaF6fK4BI9NP^PpxL(7Cl3UhBums!1p6LFCeIdoCS{c@NGSB^<(p+6!qIH0B|J@2Ycp2heIStb_dfUAvjmHaBFsVM7~h`)zP2oz7<COnJWs{A$0>RM-j@9iki@EwbvU=}V@Va?BlXQ$wO%=}nz2hL-o*f8xoW+}_<o`%AOKrOzR;0S<C6X29ZAvlQLT9iFl3%&&HJi)$zyutS=Ci<o~Q7=rjMcLQmIb2=H<}a$q<E^0(9d!+felG+or1(hgs4J??o!x}zC2&RNRD_BE^^Q6ZjXO$EC5dJ?6ThHZpBqt5gIKq{G$t5=O?qb{(`Ry4*}YR6CAGfkK!=aCwwt+ps=PLuix6|AA_U)v?^Kqi2X(a&45rAArO(!fBu?YY6nZwdVrA#D#oe875Y+L;YG#zT62eOitvO4VffAK!m7%EYln1IV5Vq&_7hji6x;#u{sHyE};GI<^?a&Y?{F!V%N0V#>zXW5Ke0kw#K-oTMDaqaqK^kKj5=DsM!MC>g6Yf0)>*Mfg86DGUDo0?lh=7rV{NLPECf|}M)jjoqqC&uydJV8UX8SY}fvzul|J;IeO=SMJ2O@ax%h@rxI!LtoWD}p9J@Y7IzAU~?*$@voswmflmmF*ZURC;7^@72(sOuhNg4{f0V?$X_ZD$iaAH7^ZLap*mC2u{__K9=}D=JeX+m{7fJhGoU5_J*lN0*n)sdr2I<jv-Z@e`fKhb0Q#(RE1qNCAD;mpHNWBi6Pq>eQ(Pg%_y?4<57-E0UFM^AxTv3e&~;1P#9E7hADWAFSc(P^vWb)ScUGlzv8uJ?J^1DS6Tw#m(sYQ0v(iO}4W-a{rsMOK!<sm9`FfiO|2-Rr8D`cQslYIR}YFaed5?*yzr>+HHxGT`>hyt2TA53)#`BMeoe<^yd?g@3lpBM{un9)cJ&01ST;heU4yx9R{k_XsMaTm{q^ezxS(|s9c<-uOhf_$!-F3QTR>=&%zd>p#2I?*KN3?um9nDHbjd-cZds076V{wK1~^6b804y02gmQaJ^*WfVu34Uf%BT5VqrbUg8q+Cc&zo?HiEE0wi_VMH)S>C7Vx+uQM&eTc(Qb3$dH54#uzBP3N|kWn6!SNKWZKN)7os%Bt4}5h(WGus1_@@}C-Wa5Ea*ET)E$v_G3JM=Gi}j@PvVMgozfwLf|l%!8lL@1&+K9vMmfF2pQif3P#^zd3aX?Jl4o#U<=z_c!+Re6d-fEJ!(&z%dKWlZ7kOK5XN1zs5Vl?0;J0M=jLN+fvVot7iYf8cS&^%Q>Wpu*dl)4(rWbvQ@c20Uq`SrcZIcE&Uslt5Wq7V6&IK&*;RSkjXNKa`n^}u8vSr(1v&JRw>|rR#$AjVVbDYJ`2v?eErF9G91^+zsW|?SJ@rOKAQXPuxqMFYT`=DxJM2Nc?z}=X#_L(vdTS9<jE6+W2CGDp>t<d3X`sm)Z7oPG;^VIW%gQ4J5JF&#@7`+&MzT@&Hd!&o;cv(x97}ML(Z>2%cyA4+|Lc_U%-E(n%rD^+yod?vpV~lFc*W<YKo|r-tqByEcr&+t=$ay;|ft3w=Q%pbmoX;fJiMe%?vtVQ{#RIN(xapTQY#BqtHf)NhnEO`?*kv18%3O&;`J>%oUTPHOxcr+Ig;Hh%|MGnOHcFoZHQe23t0jdK)ZZ0Y+)SJivU$M&UlDZbl(DHg{W*_a2gN)0q5*y{1YuFVrE3jBh7%lMI-Tr}+W4Aa;3F*N55%w)1mZbS8~~8YV6OG}Wvg1)t|3ku~s~C=>KEvjaOTK|G1omLXydXH`T@rT;we@3BecEQRRQ1(w?_Tu~zyKDLj6q#B(=Vbi6kPSh_>wluYZMT%o@a~l}m1?ij5fLhl_X9(pzrDwb1&m>a}QZ|pCXti@kZHDTzrz$b4X^agW)bRZ*Q$6h9@e;)c9@o^km)VB&JI@5gwwGT&rC4yuEuCs|F&DCZ2q&+E2&rhLqb<uw+_h6YN)m$ysNHuR?TQ+!w6-<Ro=0`D7XXZj(*$oB+^IhJQ$|pHLKGHIF8uZkoqhR@_y3vT*saMUM`W~o0n@>d)wVPO^x2NxLNB05%;5c#qtQZp-|z2>a+Yki>m}b1z;149xg<mMVC&PVf28!@{j)b%*Z^T=S<+w&z3<-2vl9L3RP3;-dp6O9NTHIq(E5ZH<?Gwql!q2^dDu~6oATFvy#FvbS%n%IfhUbgy_Lk)ZI)A$^5m^GQGEy0FK!=(UuA<L$YSXH@9VwI)d$g?ejjbBkhi4ZP>yst-V&siVHo)Fe=f*#d}F&ZFa8oG)bZ;S;|5opyHmb}T0Lk+P`X+0?FH8d6Xg~_vZZ31KUH3w3l7~j3{{<_)Y%vn2m*yf&i8%16LqR7rCu!1zHg>|MV59|k$Ypc=g@gOK7x0TY3K%rF+RB-e);okKY!o&<*T|;^gDU};ls(~Zk&<=y1gV8X#d*UN&5g-u`O!52At*k->lqJH=Q&!^uds3$^BHsW35C)diqosM_JAKIc)s?cE5c1`G=oAe6`z;*N4yGYiGMmx~X>?1%CO~-~_7Oq2iqyc?5IX1|)sBKmGQ@kKh0O^Dlh*Ef8df2S@#9$gk~@(Twi7Rh?3yVpAk%YnC>4lT>KhMzB9N3RB$Psu!tv$4#|&lj@(xRB{Lk_26J62mrfv&7)l|sN{Nyo+3|1Z6lfW14I)hD!j0qG3=iH_`Ux6o&Cz~1`3$lwrDr9nRYjP<u$wgp6XMW5oOmmDUKs<=lrjKy<GlwUySo9clG;tS{8rPdRpDNbJ-R~4PmZ@VOMA?67_QCaMYO$DQ(piAeAjFt~wFcA%Ix{35GbfN#{T2bI{e7CIp}_(Gt&*xAR7iwe#c(3#(Gl_dTg)Se#fIop|RNa^eYB0j(o&m>zjN=uaYCK#qKn=hj(yd?;IXs|L6&xU`uh!2&M7-{>#e2lZ)jOv$a3wgI&U-P!N2%X_eetgpgLi6p?fQjpM;HLJH5RoC$uP^_M&aJiSRVG-H04c5+23w9Z~K|ZD|<$nA9!7hcU#r<!Ol4%~;@lobMjxA0sllSJkAknu2ZQeyeI(5)nn^0(D@$LbSd-fMj$pC}jg={PU&ygpqg3fxWNJw7CPGz<WEYZ>sV1h`92X!%Ki7u^WZC@3Z>7p2F{E!MEJWH~t&RN-lO=OhYc)AyeT{HSc_{04lg$);B>}nKqS*1inrXHoC-T6M=QC%~2R0izTsVM<Z$tdiu<1cim<!-m_*Rj8UZ|`SBW?oAps~kl38id&HEHMm;R)^H0ZaveAzR`(?%y1LNOYs9XLZfX8i_KjrJmL3+%lX2VhH3Q-Z)08E&Q)I72wTTM$F@J6+?G=Ye^ga~*cps%h5A$XqJC0^n`{zFiEvWJ?d-L-89>LlYE!~t4k_zk+28bB2G?k%=PVd?h*s3H_+#3_DSzCbOlIM?u}<H}-gTk?vuQ!jbsOnyH$|$;Zi!25s$B3@RP8`(?K37?$rzyFv8uXy3bq>+*o+Q?+_uOy=CmP~e0%0xy0+KyV60RTgd0TlTgLHyr;;yqxOKH2vXs-skx<{MjEva<3678UX&|BWGclp>hJ)SPlS`<=$G#ig-s%+`Z>^g|1zi8OFSN1Hu*Po?ar*-u_kqqdmKgqk;Jok7Ne0&#pv#8rha<9Ar)|msFFHVGvXgaMx85mnDj;#icB|XQ9EoEFs)W&{5nZx{?9(&fY|J&Gw3Rz)^y4%OdJ{|AtK5i;u=y)T*aEC|@A_1<S1<VS$BX#+`Bk2*TGtfw@3!89OJQLB5*fF;sHz5wJGOBd67p(XldU~MFBF$RXzRDR;A3QzpYViGA`Eml(>ACvD4JQlbDloq8K0+Ow2fjpYJO6rNNEPp?)97tAyJGXe&|&K-*2x<(py)eoT`Ofiy;puKQ|1X2^fOILkJVBp*atouQ@?<y3AWEjL8O@l;(Gp_LeTZK8HrO8ojJVt6Kwz3e2sMx5l`Hglh{*k9>J_vDt?yUmQgbSWoc8=w83a!h>X?J;h4I>Wusp@lE30bT;Yn)nnnZ-e&S3L?DNysq<B(#9KA(t%LPYlth~6v_Iw5O^lnQ2In8^H7nG0{)jYr5#f%VOZz4-Jjp=xi13*DJQY1Rs1$%;T)6x3HV#Kf^zJK077w`3{W#jKLXHd#_5L#j1-5Zc8I^y_b<N?pP-Q<J?wlxCR!(>HSzJUL_FV;|Ji8olrTxy>)j*h>`fgyHOMaZx(`F^bEaQ{#vB%Yp1;A*OX1EOGZmgNhj~{>D=qSa5z{HmQ^Y62mFm!kSR!FG>zfGm}&>x`p?!SOYSyln+_I3RZe@~s!FrXR;S9n8zK~&O|nyN7!LPY9BEkZv@(gUTU^M_5G16J#7OnE*q<2)Boa`Sl;GwJ!)i&3rX!NtIF*EF_tr3y8qo<3F=)}n1AYpZ1P1NNTor%ZF2_o4pz<Hx^Y;y&K{q&Y1rrxRsVEEX+K<=K8}l*1er%Azj2wTWstWip!4)MmoQ%zZq2cO|(D)8bQ7NwEj0$FH+}yoa`!prH}kixWXbz_52zS(g@@nCe?4E<;+IAJ1w&xuy9)w-Xp2(kcf%p^-`hOO=rcD8_!Bnq{I_Y@g$WugN;K4#76m;Xr@l)Qe<+ZBVjN-I>e2=3~Co@RLJ^@|?rLp1hCIazGvHl>;>U<yFYhc5stdAXKW<?Oh6Ho6@Xrnz%<%AUF|HM=4$eykFojMoGTej>$Qo0qDII$S>!Ypz-d0*Sy%h3<(PcoS+Ege+B7#Udi1)EB3c6HWWo$#ox7bGuAS-v%wdaQX2AxkW_cKRbU-lvTY+R^4uM%1V6N+VgNK{qUYY8fxX9-1=F0GE~a6#8GK6vuyJnVNTa=EGHs#Ewwkh!vU3nf)dAG4ql<O&^a8d~PkvV;b)^P4Srj%qrP3}EaGvc_Gp>~jMpqL6Twa1b`g2ojDI++!nk`_#pVDWsjeGHhi3$5a^Xo~)I!`Ry7=W?AS3k$i*kJ6|QIE^6GARtKkSF1uvt%t&%{qGjAPePGtZn?AqK+Vvq62h+k*Uoc>mIp*{?fbtKP4IX9XgISa3@n4{02=RHjC@J*cDRe@tEz5ep>-#MS!cHy3-m(Kl){Auz4_UcB831Upy$VxQQMkY;mua7`qF^g`@S`jd*=Wnm+DuN_MW&vize-Ld%-0+nE&cgElY;KS=9fF$}lof7jKnQdB~Nuu~G}nxtAogvY^9)Dj-*@cgb||2iWXx}n8L4sx^dnB+FtQ4F?rWPsDnL5+lN=g?)8Siq1_UBV^k(CcLMhHh@GBY{N-VLXxXcjK`(sVw@C0e$kaBzfo*Qp&WU1)GgjHN?x+P%8-H6w$>rRB<qUpP@J66;le9JNrQoJWt!ro=vCjAI4Wr>@^lAr?F(LOEeV4hwL(CAU>@=3Ek=1po@w@hU{ZC7ZQX+_JIYDpaxl~=6yroSocwzfO1(bZsme-X^8z#`Eb{c%kjfB=q2{(=e#nMw>fE-_)_NqnEu%n)8S>noNZG}W<+;gwsg@i1FxK)8c9$}8u2Qwws=m0Np6J%)~XiuhMxIdSSMvrQ*&|qo8T={)Eeewoh)FuMl+0+>tA7ybW@<*NKDPKdb)38SNs~zB}){ZOTdL5n{KDnVHoWRv9mi=X@xIBH+YyD<ZP2jP>2nI;RO0Dk&?_eM}3AthE-e=4+io~6EWR-^l@!gCOBa=E8J$im~tNvZZf@FDUU~#`84|HLla(yFbwD~<D!-4bs36(tXi`*o+0DO25wSe9c_J7(ZughKx)U*$tcDOz_n?|4f+g1UQ2x#jitKGwpdAFh6<ekE$AR1Q>U&!gdMu}@7hK_D0rVN+rzLxjF+g&qJcY^#jwv}R7Wo`bAMD;v1}A=7ZsY3ZJ>k0=oTYDEz}b&_DM%6>TgHTh#g`{6_h9_SkMG~Nzv6S?~HrgR>?}+7$*RI67drec41t#ZIElqa#fQz?tW(w+3D11r_ceTAUiM{>34*jl0jEhGD)6#ttjUqOAG?*+~<*a8u9t<6LZ1zG^8vwk<>6$Qn0HK1}@XsXuKeBqTZK5tpcM{B@p<mB&}2YYApau+8b+j-CWWYgzHSjeMqZY&p|6D$*Vvl-wG)41mLu^%21T@fgFXp4}Fdmzyf?m$}p`qBL(~<SE<g<p&ngXQJ3W$+`Fv55So}fH^8mbggCE+7hrfAL+4bT7trb*=0tx2K%=RIpW--t%DvgTJkK%OJRIsa2cA}4wD`0wXkYk1F=v>^hEW313&1NPTD-MK2Szr@%i-=Xh|wvslXN4()B_mzoYD4qIQ1WAlkjj@8p&Gcv+dmi?*NAiq??oLv9a4&RDBP#C1{P;+bZZF?V{=zS>es0sNthU5?6<ulCyeP$?oWMzK^~rz4OwH#nZUB#sW1s98S(Q)-F*kK#>q~NT+0K?apeuv7CDiqCKwd_don4K8qa#CoD79?=NcR7~vaj!pG?lmUO$-6UHOKz*mBS7bSDT&{`DTj2^@md{mUlLDm!8tKdg~dvJU4{VyN?7HDH`i#M7QAQM~n*)Q6-5>;Evy)h|h+sMvpWYk};DSxf5O$?ubK`%+;Xt89T1*)w(0!7ZNeae6ZhM*&nJB#hez4n)rtu^2kfz_L?2GLuZ&6VTsATjZ=vTzCJp_(++yUniFM+j{8f%N|2!P$sf(@9WLD4&I$k;PM>(;Zq68i&e+F-KinRPX79lSN}}rhgeWQ-Cx@@ft&pq{gW!PRo&uqJNWgBu~vi&dqRUTXpiDu@~h~glNMh=ia|4R$;Va_$8C9l-Xq*8Z)M#;Xf>X%UQ4(T6kBTxBz7_)^$zD7@8hdq6tmgO0OqZ%|8*YsuXJ-cmsDZ^wZWZvCdX=U*ZXeE}s(lAJ)VLTYFptO-y+VeuK?n9sK$h?rZr30iJ-D*|($R8M<|QOB0i4&;jGP5AG`rIITq9Hi^j=>ET3jTF79Q$-2FUM((Z!jpZL5zA<z@NMbd*8ftXeuS+Yf-Sxm%B_C(WZBQBY4Cf?uwk@KXK@LI4)q`0*?j5*g$_bD()?QOS6gA~_$-|Y5$SZ!M6*`)nF_!J*RcBX1$tf_XB|H)cLQ0?mnm2qSnCW=xY;=HOwoFPX6`CH1xSx|b^FG1;WgA9pUIZ|y7VsH)r6O5%I-Mfc{_{UQX)&Gl8+bgaR*}3wq1Nhs{Fz?BTC|q0MO*IGi<4_&NkFnaF`_tbwfr`f!f~;*Z?H6e=?|$JM=ESeu{f%vbc?KBwgR);|4JTm_<XP#NTesXQ7d<RA<phXKy~d0>}E~NTK}V-eN%N1gW@O>_3o|bbz2@W$~l<uQiW#$(=2WUH|l1dw&jwlT02G{a$CX@K7aEsAM3Y2ef;Uer;pk=-YRYL=Xbw8i1L%~dB*2AZ*ej;({1s)BWhfffq6@#s>gTxAX#y;%O0k?&FP>@)5CG{4Pg0kIHuDiz;=P;#0P{;@92-B@?%kl4E$-P{y{dm+Z;%7567Z5nRU#)gl*U`?a@$qrUKbuu=8tc>`AM(tR$1-bc}=1xCR5#ej^RK<d%vO828e`5El6iQi{i#1ILTI=4C5TI>;{%dj@v|P8sWkjzyI*5wpOHODvbQEhjQ8V2s_j+I9{0tZb}>$j7Oy<LHa!UsYF4QLXxHRzdS(U4%wwGq`(&-ZplPoL}6iYi-7&T-j0~VtrPfQ5oQsz4p^<MT@BLA!AWPn2r?xvf$vnPi{^K&t)VNxTd0n*+tjYZSZGzSHQEL(evq{o5iVV4zZ7Oh+|PVO?D1XM^QNdbxhOKA$l_FYi}RMZD8u*e3s#fHJ$9T(E18*gv7#IBqUe62Ut;tZ3Q&)PV}_uVi~h6TXHO_vq{hs6F%&_&cOZw#V{yX0g%eLPin~R6Hqo94iIemIwqwUiFc#azx;HXE&Ei*!;-;HPeW>Z^$Q}<Q{)lLQt=v#;<dI4XR-j``o%jul~RJ*6Nl6MPmM&CCF6-IUKfsHsI^)4Dyo{DrHUvl;C&(&HBQT+vS3*K%4FE8aj^<6KdRd%v$jm8M!7t}b8xPPf%SZ9DEzIdPK-R5BNRAe?ix#i6O<(;tZUYVZG-LeoU_F6Vg?<f)g<Hfd1$GROcjuddXENmk=x+MTxBH5VsTzlvZf;tcLEmV_r*bMM8CR<n!16EI3@3aT33zA!d7*{zAxEQ>-~IA6u;YqDO2aEGW8H=V{YBBPvk(!`;`KONME42vNpSg{SrQ(P_2O+8$bv)?m5$h?mnDbik9xol%QCfWaoJ9TI)Q4ub@gD0HKN$IAms#C@&C6BM!hQ9+Yu}o(I1i*f({E7Y4~3OHGpzhWAw;bGF0%2cMxy)azxu-$#^@;Qc-HJo`5^(fD_(kK>{?_dG4i;Q^mMZd1rgo*k(aXA_*0mBeJY^)feR01DcpQDwVL=J`h64tL1;0rTv0&70a)?ga<y>(<`ArJ|yq&<G2*(VLj<j6dqAzSUN=oL+$kb>9jPS`>^ew9e)XAQXSY3#f`m)Za$g88#|rp->3dj-X5oKK*Ub!y$==+F^^bCA1Bb?zA^9qd(BNbTW`<uvHDBO9YH}5}P81I}+dtVW$X@I-<-T;+oIpEk%)ho%i9Yuz&yu7lC6huTa4ri{f~d?BPNxSct^Z#Fq3UU}j4qFbh@~pu*e<O(jyp4YH=%lrX>{-b$uIXn8ovZL#lxqLSW|RZyq{?^;Er%K0;@AHu5eXL3r_fdJeXlDlPVlf|Idsw(Y-&ST9_P?~Ih@A-?qG1%F(W@_cqBqdSTO|F-S;kxF^*CI$15q>VAgac}gcgf!nu*7124pZB4rFFj4=@tXYVD)EPs*<&HyDBAvYeQe2;0BWw@s6?rTP#Hs1Mz%xXB8#zfNBu;5TN;!nkjN3SP@58Sm%-e`@xvdnG<<ed4ZdmnZVkt&J&;ofpzmuC)vA$_Lgd4xi<hA(MF$S@jTcFyx9L5_n0&rbOv`@O?pY`wM<1BJ*qWB8)#G)VyY>vMfqX3BSY4vlWG~M2KD&w45>0+*w<vF5qq1If|4D5_oO)7w}pY;=hdmW*VD43nifaIT<XK3%f+{hejR3ri<29{+!zkpRSJMHx9UG`VZqeT&2hcmrI->}X?H?dDm3+Wk$Xr6MjucIh^1cXOGQsbbLljET_5Y;0L7mA)uxXjGd5|A&8Jh%8KtN`gRA2wB^y>l)mC(j)6yV{Bx@{ql0B;_=n{i$ksv3<uH!P8DcQxEG9psm$^k7tQ(hHz?A>NP<Z5$8Ydb(IY3}>84RG`WmxM~ad=!l45)CzzML;MPL3SDMmNl9Pcm>;lm2<>k2a%=;SQe((99Z0T*xzr&<2Boe#q0f*U;Mh-qee~C3%s^B&mF+G39JcQr^!$|1rtjvn(*u-E6!>E!7BDXD7p^aKodltl7;D9qb))2ATST3SO+KO*eAMB@SSKiX7zd-2DM9*EFXt{IWYM!<l;_2b>6%6WYa!@(M|%hZi`l5X%}h(Mlm&;c3qj5=|3VOt@`$M5)yCpK$`6e**mvs3&DYf$dsN=IjQq!*hj94#!=m19CN#s(uC<J<+CSQ2Yw}8B5e5t+MYC3tUD#^zptchl520UGGw`tW$olz#lZuRlzX4@B!%K2emtbAMY*biMvuu{`*>#EvC7v2P`5w)j_4>%(?QL(r}%Sv4G=Un39W&%Q=&gH486Dmm$bxb8r|mUG&Icjm9_rD(2U=vp|T06vLSvFcDIW3fcKMjQmJ^&{)wY3Mu7-<;`Q}4og47J+t==H18L6c1TAe%x~6GyWhVlqTQ?y{cpErQ{<<!PozUFHw6+OQ@CZ^#%2dx5_$aiNFP_kBl#n$jvooyA;#z9enQ87CxW{*AhHaAsC{tRUp=ClNg}Jw-%dB3)iMU9K9J;9EemTs8E61d?&>xW)98gn~o_AFz&tiOqEE5ZJz}3dNN`95oRFri~#NWdt1d69_6P`tGRsNc0b*(eN_jVL$_>RR-FbkKdu;%B&vs3UsW_~QV1Lv_(Y?$^Pvy|m_Peb8Cpcdaua0I}n32@4y5FEsAEy^CO1z&=8o?u@<-r)Nb6MfU0s28T%qU`JO9Ih^8^A}a*@zzj?j=F|KzZZfPQhX$L)D_j{&Thi<61XCBDndnodPkjy#vLW7l0>tciC<8y&y6UjL9E+e8WRk`CcQI}=`%U2?B1!3l3L$%pu<O6+s#})RbCs-MTogl5rS{TcPdNMgSuJ>22*6m(r4>K5~uNH3O$=!v9fd7;_gm32<muaH8aXv3E?G%)|{owK#5AV%1~5x$^%sw2;1}ei?7QjT^^<})YNt~@Xo4|c4!C`{!BKXqe(V`UxKkqzP#`=pllzslw|LQAdRsMi6TVs;9J}L3HP3Y^>O&LjE?Cvl_M}&M8HTw{%>w7lW$3s>YjQ)Q6XSUy$0AFvwa$gK-U+&e{R9KCNlrq0}(v;<?NVT9VA+PvWZX5o_UlpUl!k{Y>0;(Rg`POOAfXHuPS}4dcojX)O8OsL2e$hv7xM|wzCPIk6x}Hp;q~(lD8gd`$W2g6_u%x?aP8K9@$SFiMojOqsz<Y)Vn2p@@DhI_=!&A!xDw==sKi)q<}u_OPtvG5o=o)b?VfD!i!Xc2M=0^70JrBc?#DSh3Vpaf(Bpoi>=tG57uyXC{>zz>dtL8N<X8-9`u~hlssvT;%0PxsP$}%CfivZx&KYsCAZ|RN?V7#MCjk^s(HqeyBe*HoP)%oxISh`Y;<Q`?Y6|pu9$+URhzojh3x3mqIc$a`tym$_u8VmBRJN4>U_d00+X1MK1Z;;4g*zdwA4&v%&On#-}}`}R4&faR}tK|WH*7iD14`bXJHFb(0&D{>o(lc*Z=T68=}RaJH!Pgivh4TpQenkIW?0;fQvUDxLz`Gz+Co2FK>5v2-|TzFL4QZlVH`)_6^8n0g^iGB8?u`lFg^Z*O?aKEmOtzh1gA22jkc6rgPiNGOoWuB&T#ArG|VRWz}ng2o(Eo*qfm{`A>~GxET#@7E{AW+Mi9ABNf#f$LrbwBY{ZL+8@0N=E2YBcT!UqkBp>#7h)E%KiC=d-<&#xb{9~P;u7|<`x|?DzSyi#7Ni_X;FyKx$-<RsAGUG1U*jEN_CKxhqZaDsZK>zPRkQzKjioe|<s8yP*yH>YhxO(z*{WQi01tZu)2BG!mi~>&RjK+3u-VJrXLMpu$YhyAxq4~~S4Su*Xu~^qs}%4*t1GtNFilixp9N=czW(Gl8IEh^-(;ictL%<sAI*Jt*fmuoHE|_n+#`pCJOx{bG=iCXS>+xl^5hA^F;don(7CfJg-KUOYVL<tnz_)qGJCD29j9m><LinZ=a-Pd=6-T>PaJUY+jC~BA?H`1WmL3i?&k*eFW|pXO>QndZUT&{S)F}Nn2W(_HAU1*@A&vUmVBe^)^3LUafPUiTNgSPI&(xaK%^F#W(FOwsd2vpC50%QEg8VmQD~#YB$TAC{ah%-0k_js=mKC`=8DPD8s?#Q?L5~pM4CFpOe~y7&h2JKgDsm%y$zPI0HZWu9$-FWqi`QnH=~dno4c*Zdk;ysX-s~@UQ;ES7wQm1#<!EXNe0Zv)BFHi5W768>qG4W+xa;yI+Mmg4U?9Cnrc>$g3t4i$QpP~lnMHo*@2yvAfCi(%Mh`KvnnE{(tjTK_t>OzmO^yu0?X|duBZ_UAKS-3QjN}`u<24%C+Zg`Tbf$IBE_+{xeW~Og7nR2K&|VeGlcS<(z9LhXObxfDVs-6wA#6&HbeE<Q<a$2G{%MwYWRMZsUCLlc!}Zzk85h&%WOmXoo9k#+sm(?QY^USmQFRfm<!oHgp*f7gjBTB(UxT-?%F9HC5gcU)b6{Ec14X<THBgu&!f873joH%X@a*5?o=QADI+L8Aqop97k+z&&c6J{`~OUE?AGLwBQjdPfaze!YFios`fSH;p%+jjX7K*W(P*K)@AvmbIZHO%^^$K0U^ln5T#_Mru=VNGKT>+{{@EKWY=E$`ENQTX-gj^1S&9C1Dt1`aJ)7u4q)^FQXnjJ9^7ZX)%0r8|JnSg3P5J9S-hUXJtU`^9z>~(L-b!NYHp{6=dGgkpsJ;X07q<_?ud+cAWHEI9_x0Z9>VxP`zmGOm$XilyC`Y;+ZwXS%Fbw?oKNsXVzOh}I7k`No>iBhvaf2()-6`KftsXQZDBUdh_JZq!iE;}d*;291pDHiT1&3}MhN{j|>THY(1c5>#=lee1i8|GkQZE*0-#62~B1^lf$i1=JbLhMsAHloFG<1W*7@u4Zzx;W&pTBSV@>Sg^`kg%g@Zn@~H%>_b-Chz4w0~{wq<w&^*cP>21I}{&Z&q%qn@*Y<`d~=2<bEpRu~s4?J$<T+qpW8A95#M`yI(&1{KHQlzS`}_>%(X8wX<C&-PF5{0>6A~a01otQ1MQUJc2oG1Cl=6pMLw{$M66C`4>L@76`J#gQNa4<k$AdXh!$ks!pj;u_=<XHA@@2Nh&mLBiJ7sg(+@t)r(ZT<EGlXN%hZTDmes&dT=lj1c2ST=Fu(}RC2vUPmw31wvo*G0ip>L6<%1*7<Ny8{9b?k&VJ=~0|m@&TeKV5OuHMt@|xX#PxUFxh_dUO6vq*_bN<)AUM_#TFUI+lyZU`REsMWtJ+1ECxoiughA`K{uq(6`iF!G6IO<G>l(y;$kjfSoSDgs!5WuW}1VbF#r1Ky1Iq2$369UkeXo=^@+j*nM+Iez?g;go&`<~P?EKV$qPQ3FBIq`(6fYuQ>OpiPs^d}K6AV)sPbL*@;K9nuHRRi1>T-r>MU=c>T?*Q#SdcXBwv@`1S<CwNvDRu*@54yA8ahG>x3wdCL-x7&}b)`h1DRfqEII6efbD~&<P2q$uUBe=>XB(`YpD64y0)%`_SrY#C`-5E^QOo_`9wk#jup_3-gB)AjS|<3-cR`|)2im-g0(t5%xHh5C#^T)r9{21ooDv2GzYF<U0I(xZR)wMUQjs9Nj;6}|7g(aDA>;(H5fAfX6cc$7%i6vwEcZoG)%YP5(s-7LPo2fGC7j48xAAl@kiTYxjPQs1KME}_0^8Ln@3M-EhD?P@Lj&}Eyra5i>NpM9t5anHo{~}OT}Nc-P|MwJ+ppsY;NI}hh}FC%M^=G|>^=z5-dUm=62A_~M%{X*HGZQLA(;UvjG*ENYy?Z&6c(G!Qh37e3zzeSEmhO%8UBCFean(7$&u?1odxp|`B?T$m(9pb(vi)yjLZA~e{s9JGFfB<9Dql%q%rG7S#=#U-VqK5;E%TMZY?WMNW#W3aJC(9H@66t!6;Rl5Id)FjL>)+-c(_#dM29`Q(~o*d3KK4?F`^$+{Y>LU=B&_VAy}?zD(|-l`m((u|q_op245fE1WXQ{mx_t{%C9YruD8<2bf<Avam->7rRVSE5s#FV$=76kD@9N+Pa}J8%xFs4Ubh-=Tq?CsO4sK805A^u2iQDx#SBr=MuQ1HVk9pMG$TfA8;ARkAq6bwBgp>nq+CPizA_NP&pd2CK7BPt=vFD=~t@4I7}DoxhIEEg^%MfyVcekfZiH6i3+&y+p%)XLd}}LA>#HMI-d<)@&>!-P9NXqL`#GBagYfV7^Q=_$mY6xoKPoiJyxthLIrK}zr;|@Y2nd;-7V4{ITOAdA8lg7Z_0Rc3$5FwjQm(%XU@lQ=t~aEX!_d`qj4Pbwt}=cJq%`z>IY+Gr9{>nxcEVY)xMKK;2gey$5|a7EtHigoK5XA*&r3!tgCh#bi!GzcC>J@eTDf_p|(Yl3Ew6S1LLRXxMqB)y+z2fnyI1<<7n3rZqMahwQ4%t;k4`8!QzZ23_kJ$Vbp#$tg;I7cMP?spo>$tX|R=wyuvoH0%}LbELeRmw94=YZUYu#OLs_~*vnLd8!H1JvMItBzStxvCTfQ`iDQq`Y%hTPZt>P;81qiDcn>3NdW3)|VgjvMI*1cZVAit|!xpW+L1`wgq*cj!14`|G6K1Ie`RxdJaw7=aTF$o0E99}VdtzH>Zs*&-JkeYQt@GegsmTwX4Id3nU)nV0>D{?TT?nQjR(#$M^K^xn?*7CG*ilG3&zGLb?oX(mo+;8l+F3Hr`J6<@UzUb-aUEoW<}L{lurj}j{uT>U!&;`Gm8;9ER(|S?u?)<<=`{%2x#inQpUq6B%W@`(^w7N94+hX0<-;u<TpzrXkH7x>`$6caCIVAK&cFZcMJ32=KNXVFz;{!-I`j|F@4mhP0j8Fy^Z2;&1dz*MnD93Z;jU@uUl2@GVSz?>2$g400B;#>cp5=ie6q>ofa!;gE-x2)oM+QXb~`T&CS%cdFq(0_I2br^Eptz$r_h(_<<DBND%v(uuT+h-S<iVanSw8$P5sZGKmUQL?|AQ%g0D!TCZtJ_X&#))i#=<U1ssY`QRUipiApl$SPUq$T_&u}{EcVtt|S|vY(A-z$R&Wry+UqeU24%m!*_EOCxYUh!+K1$)+=6Osuq;E45{8d?$vU0<?aCZ2a6B+e}g*92&;P?EV~1W^WLIP%t(sub3E{E)~nU_A5EPu)cj5TkQA*3qKD?r+|Dtd^W=h;2N}wDP8aLkK1a&|`D#=Tpv4y`p@+7Eo4f;|Qev@BdzGV0OCnpKSeBA}N%11^`UMZhC?~VnF*ye`073ToZRzzdL4-ZuUCU-C@(E!CoS-OG{SMmnvXi@gQT%Bs*Asb@>fiNL2-eoGi@~^-_7sVPZO}Ql{9YSevTY+*<T_l@QeWCpF#wucqU&h)z~19(1yg2-mZPxVO#Vs(uyJnV$VE$x72SNbvAKMdor6G%N#B-ZuExnz+#jtjex)FFr3N@z6gE3zV;2cn7h9~w{bxbzTFRkYI?HbT+EuQ}2u|)F+0=f_FawKc^UZ#nLn{%NFwHkF=wf34#{S-Yojo|ynD)btjBTApwZJNa5B_qNzD3jrj^00Lg*sHkO_5%qKZyf$ftIPw9eWA6f&MnS?|;g9@T*>2Es9RMGWkW1B%{u%%)o2igSF%HpWrHL(YxH8)@hqDZlz<*gLbnUP3`$=qQKxL>UOZfy;}z7E)W;C)-U1V)$-{2bB9xM@|3pi-%S$Q_N1F}QN#~g6eRpZswBlQ+?oH~cU$MDh{Le@5a*ht$}NP)!Bna7E>%tbbcKJ{YJn?wKbt-EET4<4tvp(JG<Ft^ZJ#*cInLpZgiGc$WL!=_lfZX@R)(w?(4eK0;125&Gc8M2JO$d?<!R`H33TSA+;@_iJLbeJcy1q(F%R0aySg_jpb<4gLx%+ORWyx8Bx4${@-QH%IOj>5?A{DMW!lyiqP@l<=rqcVJ%6T3-p8(42Jh1nb>Tz1UC`rXWBx*y0dg3g4$CaqmMWDujdQmEv|0k{F$1m&bH&-~zvKg9J7UL^X%J8B)-QS2C2w=mn(VcQ0$BR1YgB_4GS&2;^q7+QOqHIn(~O4EpNWN)a;b<%akq_W5*%|YSg=O5QX6{9?}ODX270~^x0(pvB9;0uXPIO{#x2fajNJH+^+-1b%9_Na9jk)+(RRhW;Vd)#HnMR>er&pRNvCPHJH*azrY&oK2%o`I>9C4TVniWo1e&wps6TRP`F1>Lnq*kT?dD)0&-@0{g+8ChW@m++tkQr-ua(7o^Sl&yD`ofy0Z`}reCk5-5QYKY%UI6xIxbW7Kbt~mooC2=vw@pbSVs#DH3agn8IZoLv@?p!0uXOn--6CT$arZFqe`3k7%cC$jKwja;2Q*F+NAa`VLhz#vyPTe3iv0(_S7s8<0bT3G;k-qn9g2EpBS*r{Z?5r+9=wtw42e|KnI6Wn?-<Hr~+C1CY_~gj2%HE-iYN+P@|w=KojsKMOW{<4(;`|N=DkoI05LB@lj~wgAb!-%hAX^wQ^NSwY1St*syZzT&M5>MnQJ4Y~;HmWSdM{Z%L{-fU%-%h%99YjPsaB;%UU^7c0yK(`8Dn)I?GPhJP{3q4{JmsB`Dy1%VTNeHo-H&^jfKz-J{XOXBOe04!;5ti^ril8PI2HIu5J&8_F46=k)px>$h`s+WuLN-*)#qek6_KKlw_0X|n{D!a{90bj`-ptF5wTUTrCvTTH}n&xkWCYHesa2szS&N$)H46g>?nqpA^AKYn4c>e=}P2&p{$KfUS%iiI6jnS6rQYAMqTdkb&Wn0iH?17igw9Fl?1a=sJaYTF^+ZP>ZS&^5+*Ly*<PSK#`Ga^hqfOan#*Po|T|G77b7Y<7!S>wFguUqgsz@Y-El;pnHv<-Iru&nfaa0;UFwxk}U6jS>HapDP@zx(ouyF*UNt$L_&k0P9QJ*~HnG-EXzAMRp-?i>y$=V<GY&=F81gdEZmv2r2t7r_5qEbFL2wCAz?^)LSwpT*7sCoD7f-M>yY6&!8C=k5^xbpNU+j7Nfj?*s#{#B^ez^?1D*^?fb)Xf2xc1ovv}5Ds_6umAJ&AJmR|m`!t#!p42^<s4{#8ykZ^<Km-7TA|YSdQ181ecHtE8EEvDTpT?snP-6<FYu$WyM4-l0fwL>s~ZunBlp_hZng`7Cl6~kT??Z36mqNEJwRgOv*vIM`k@LSwY$yM@goE_J0YWgcyKnNc6$<(6v}5|XJj=Cw7XLeLgP@GFy^RLN9~^8I9W8tX2y3aM4myKO4i1ZBk6u>+SYO;qq4G=56N3Mkn>nLi|t4Gm9ZB>DMGa2W^|t)6#Fw;G5nTER?6%$4viU8(0D&=emPe#7}}Coow&TpLNu;>!r;(+Vac;KR<U>^z2DqwQi*U?rC8g*8$1WYciKKA#@Q;IOJ2g^lkY=?-+SVMZC_jjO-#KQ{Nk0<KKc3<{?_sd0%lRD?~-Tevgm%L2)z6w9QWz@3Ik3nk#}^W*!De~NKS<erht}BQ^kX2&$Xav`}2cu4j&&Ru^L?sExPRM((2XjzQA{7EN2;UP=oah=OlHuZ9>f;n;?wrL9d?o4%{+z2S^(0RGUwg+PsxK+)0Z(@f%g>XmZ9}mugp4Qwb%f!l1VBNFWF)fevWi@CU(6=i6YT1GF$bF<I)YewU88Uz0h{6z|u=>*)CSwv8k<H3FVgWvh%KQ<cQJ-EI+^|Mh>pxHsL-n|ZG4&9@tq3l~~<-{-Fx1>8k7em!n<?;e~vbtDk*>`a&_?ptfvP0e#$jO`nS#+L*UJvs7iQy#~W!qOFAy=?>*F#<lJ2%ir&Gl|r}HZ*kC2V!*x0{U@3VLE%-2>ahP_g$(X2GLQ3$}fQ(W0y<N;jIbV0tQ-KIBp(y8Abeo5F&yjt#1(5KeaG0iDkc-l}8$+!xVk4l*K+vvfZ9UQfvoE#Dz>O_uoJHe}1;V{_W@Ae);XE-T^-v%lFsc{<+%fzx+(HzW(w7641SP4E}J%q>R!}A30<-eUCph%PPj*J^FOfyz1%9{Dz#wJY6%X5}5`8W5uVS&21U4rX=P{gbjiQW_?07xmz<y5ijRTnBBhK3E=sK-ayW$VSY`0!$D)0_Zke43U*mPDn)3SC!?MZM&teZB6P?t3n|d<t%n{hk~u_&*j{lo$~_vn=RG@6F3i`)y&_uzq>XJ<=SpHtoILR0l9hWWE<MD92DEYbu5H(`p4A%bVKH(Z?>zf81n#QXsF3QuE-Mt@TM9hdo5|fP^tQ1Dg1)j-3ytPVt8G~@vGTJ{s|<wB9{ciM$0q9f$SC9xizLct796Z+=az))UdAti8%oL?gDgf{J+II1tnNO6r$19G&|^3Y+tZZl7^hU{N=8oh63@<2cLa5C)AJ&_Gxzu2M~v#hY}R=%!!xYhY?W+#sdplQ;k^=a&D|rdQl(oVj=Uv3>%Q34ElVDqE3rBWgJS-Reft?H55TzOhZjm&Rl;_;Pk@RxY#`Y5ZInt;6Yq`*!M0aut<0All%3*_bnUf25Wk`-ty(MRTXc#`j}`J{;oj{JpMX`$7;8CuwEUmOrfLO!Le%U6R1B;)Rlq{z?F?1K=mEbc?pbr$E~%uU7?`=zHSJ=8Hb2fgI<xn5rp3KH|95b%hN1Ry>(D{Bc?`y-%<(InHV>T&=>&s|`3##fxudaTo`RQXUQ8%tETd!uUrG8+xoApHE95U7W-E`z&pDwe$!l?%R#M^;cwK=N<QEV^<V;^|MwhyQl{gLnfq7Yr)54m5Vts#PmFMUAobUp$`D3Q-RkPGX<c&S+rn4g#902YVGDLO@-9qE+7WNg5KEGT;J2rq2Y}_?dt?#QjkEAW#nJIs@J_plz9eQQ2;H{vPAAm<j^d6aBB-9onjl=;MMgKC6&~@_F(f(40cwvy<vCKsooEZ=?#)zO1{Qe~BtaEM+@^^?&a3w_nByN!H>lwnX{!LFxC%B^SSS#qB%azLjkSWA<A(@igaiDQF)0z}$ljgR=+?iXbSfhq``RL5+&3;@GkWB>oIp%6&^%oBi?5;nLi1XD|85Ix~X2oOrA$q&uKXuIX+EujO-hnF(--Rn}3Jw<*a!UqSia+5EOwTK(db2FW8)<|nq{1yZD93|$|7i4d$@!vv_(fS4THQ()Fr2s9Z)mJY4O}m5vxw*t0qrefQ^c@I0)!#F7vcR!4DUmH^i&NUJzPZoh@0&KY8@N|j^Vt+TzsxP2UOCC`_o{57t3l~5}Uv>+Y$j@Fv0*`=8w=+LN?qWd#d^i11{pNWE!v5hm+hE=N_mG3Ou(8g<kMtVAKJg&rv%GyTY-_Db)r7aAQbTm~HtMgJ-)EDhm6|mgk_X?raP$GO%*bradDyOw*r)euP{vWTT+iVb6&IBg7T)i9SNZ3{;5lBY$x~O~=`<(szn!S2cC$#W*$C=8J6%%8KXRm2bqoqY6@Ru}S{{$H#&Vmg1p-Fh9D=D#HQ5WQZ#^(DV!xgoO|i2VGd5m;e&OP|?bn!@IoK&BRZz+N?Sspx}dbA(or0ltL?c_0Z}&@QrAr7im=hHUhU+xW^Se&2p&0mDh4=QtmjVGABUdHnf3ebx)>B09%Zr-b=IGRDn*_sOSIB04(DN`<jd#c^{op2(#m+pS0rpUUcZQUxNyVJ?BvBAacbzr#&o&T-eLl-C+W{I=K=0jZvc0`vENGUboF7S}5bv9q!vTp(#I>)`H{=nU_tr0u}nvMcZ8<q$8t0P-l!~C+d55m#W%#8u)Hc{%^p)%edPFHnh@BB4zgh*0NS<K7aw;@l44I;?PDHo%B?$L@{WMA{SXioWeCRJQq2nQq((^NS0(-amvX^J!meNBbt7<u!Qo_>nW!f7PSTev8B5Z)6u}$k8KiV1LRaNqD|D)O#T7kUj!3o%-_77$T|7y{lt(7k>CmV8K(Om7~KAZfnOQPt9}w!6YxiV^R>QL7d3HI@ahn~mSjIVuqW0!O{3b$o7n5p1a^y5!PA0}O?ZCM1s^^GP58Ye#q!)7FF_0;NDw1=2Pft{Cdz{FooH3c^@>df>Zu|n?eJX=mV6kM@kc>(-n)vna&}<0)5om*rB&$LMdQGtDBVlD)dpq?!iY$#y}g~B#SeNQEw)DYixqW5>0pJ(WS~w~sq<$#Tdrj5NVyn?<*rtna0I2Fd67!=J1N^?TRPBsu_?K5Nh;9qM7YWQK$s9(Ze*$6+zs=(6GbV~ybeOlQ%bMOogOs6O!_*;JsYlRZ!e(Xe)9+7<WR~*AGnwJv%CigDqZGp;Ovy>PmEM=Zo?&^v6R`ZW=})Ie4$_KUl=C-M>kZ$0ck7ZnXt=;q#Jymv_)m>HE$@6vKUJuWSF<tS6MsozQ@O&&jwO0*9ltss-~5)xyBcP#ju+*B-9QZC;vGv#yX+tjk5PyVDJc1nc6gS3*IPHO&E7*F+^+~R{0F;vRInh_oggE2W$E3F~hb=0+gv&o#CcJBZXzO?bNK|&WX54885o1<9RtOlWPd2Oz5AH7#vVjl%AKlCv!2s!x)N%IpAvJbS7UPRVrb*MEpHWLZEo+HsOipUhlE8sB4`8zK^3o!*?uxf?2qX#+#oDuaAPyHuGb_Z8(pO;)QA7F<ULvdua+cC$%VPf<^!~O@LEYQgjfz^>~Z0nurO)dV-)4Q46cI^=GFOrR}@;8m1?tEV1$$z;5I?sFdyTW>FZAx|l@i9fBTG#3Xn4RSI}#f8l-!q>(){LYIPekF8VZ9uxdZ;w;V#IjAG)!LHM=)_TsxoS+qUd9aDxpGkCOH&MGN>FO&Nimjym=FH(!Qr)O8!g5v_BYXz~s+PG28eIqvQ>@4Gj2lDdr}1G5Pg{D?v$d{5_(eeoG{LdM8fCJCfD^-SR{4ma)TP$>D3Y-9(A9;;&P@O2d&S8|5c56istg;v*BW^|TnH2pP1XmbX*YsslB45X)thA9i-!mF7KBSDS#lz3@RU$Ef|bx)no&dm6wi+%tZjDIQkSTKc{c*m5_%Blx>Ed=L<#e$35w)|J*p3|)ffAC60xxF)&E*Hbx)83jtLQL`E7NUTpLGPwav*Roy~caRNpp#P1z_AIjSffg$oys2Jb4vtV-D6UbN~7nL{@Z*?6Jsx4yFpo{t_K00CS1f}fAlZf8e^#44(@C&z~cTRd7nZD{I(+Rtxb)~C@G2Fjbws~AAcB%g*Te4p-9>d_Q1Y<>R}JB4Cx>q_lTS5SD7YBb@=u(9D`*=wF6x|O_LoKJAUSG9Z_!s^p%xI3&X%}#auwip7^DE9|nPN++sZ%1`Ax^Jj`wVj>q#E;znrl*rD+-ox1A$75>x2iys$2e`asvA{pFv>PLr;SCGea^HPm{2fY1Ni61Q0#UI>SS?HK@MxTjyACz=qN<LSz`2Txp4qNuh8KNP`2C#pJR*ADpo##Lu6iAg{DGWs;4n1HlF<-eE}8qwzFJc1l=xKLSd=o__XmX3?jDS@1T)AnmhZhB>qc@I7H||d4WnYK+cw1$+)@GWzvbD@rn!gk4#*#+!mMvd%PClA#CFWTvZcNHC>h1QWdOccPR5PNP5zXgg$HU=6i07D4IEh`=<0J-pZJM=Zbo4aS0#XAz)O$Ql+1M8_cz1gQy(mci4MmxcRfj-q9I9au&M7fZgxSkfS%X8^^0)0t1f7mfLT=!VJLI=XX+9MUzZ<fgyxO;%DMuDA75I3GFUmcg5}eWhFc|d%j_=a7UzUPVim}mnRE&N@!SrbjNrL+~b{MR*h}ps~&Rcqs>d=3Oc{Ag>t%tLzi3|PMG1wQN?9Q7I+sR#M7R^bS};arvJv|&NQ_WSnp+hINGrnV6;x*+&w>tyW{rM7B3cCZI?w0R&^z}8>x%r?X&glO*!8DlFhNo|A#EQeV3J`?4x`fE<a@znND0`ndg*4pDw}HB1LqTQR?&K{GvQrICRUZH#&D5=eX_y*?R%-mT~H(%nQ#hvvO_v=@m`q{5}Ea@)jD~G8Q*U#*qxa*l9-Dxx530qXtaNxO6y~ffpckCg%q1CUTkT)$04(d>Cw8(@MU6UtjOXmM=TrTN{#3SBUGl8rP-r%p>*(A_vO+KWKwZr~eabsSNIH`vaagLmMS#%Ou_Xm&)x9JWx%yF#yM9u9zJEVwpyFo#*PN$fXWZ7YiwpeS0jU$##jQ?+=y<0j)HEBCvesP7^}r)J8!&)_2>0_$zolx>5Y{$EGPZmlzRD$6qH)m!+AH$oZkTV4!)N+$Yrr+xa<PJ(F)i7bcamn$BF${?^Ns$UAtCC=)hJ^0~!K?6##6E8nYPWNKsR!H0*k<+It+R8L)Cx!uCuL1Le0`xr<%>p5IIEvt2KfQ(Ibmc^lC;CZxZ6bvha%+pseZWyCeg;KWC{onE5B-0ks6p)^A?Q@T=B5F{dQk6E{m^&)V;Tv0~kJwS<C7&-$SLxi{Y>WD(cLL$?$N&D8BJd?Qd8*3CQkfDWjJ>g0q?VkHb1vhbx6kBR5+gi_?Q7T3YO%2gZC`QnWmZ>x1<9DmP4Fv&rZootl+h+%Foq3CpMSi9X@7jj=YJ+#_9%YIF&b^oz+670a+t=1KCyDx=miv6HGKZ%Y?Ky%y{=bDoV$PI%N^K7f2~K#6isY<I*rc{ldl)s52(q2K(kb1*f#d(Yvt8v|MThew8NUJ;>6=3g;PF)?Hd}DKYx96dFrvwhxIa!F8{fY&o2fitMFPQfThu?A2o;e=w<2h)qK=uYVUydhsPhokFsGBv|?!gzsCFMs}HL?)lB+4FCU4+scz|Zz2He}DKYWmf2|04d}F(oHU1DKSe<Pe@&GW_;g+u<*EX7wo*pas@dI~16G{<aKhqG~Uqs<+#mhUIp@e11dydhFAaqE~ia*<1=%bpl_r(C6`=*>Dvh_Qpd{vfv4V{nUBlzt#hv48a#wXX)H~-$--#>5p<D-UIRC9U#;>+pfVJ=B~;8Btf^!wV{NoNChu`Swl4S3M)|Fg1HL%C_X>C-}5B%f4~nYF7DsaVup9JLVl_ptHT$Nlol-~aNrU;bRTKkrX}gRh<AG8xL~R!aQiQ-c$zeGXN>X|YJqr=vmg4fh}a{L5c|{eQoI!*Bl#?B?OY(f%{^t{u^$ncZ`%Ii*7PsmQF?RAmlDx=kG|IH!%m6pvrEgH-*-O}-CBRR?5_KLmw(dSE0d0q5#kW?Px4%!r9wBu`HrEm`ygq7D;}VW^x8tI9uru0MXOFFNlag}FtU&PBG++KunLW{>YvRTwjt^ZF*mam3@C|KI=q`1r@yV63;?)$ikJS^Q0VgAME8@-GazVy-~rQ29g>Z+7<JXfqkorfaA`D%&<))mdyq08?xUnmG3r26)crpsO!U5kMcJ4XmTr&JTL5t;MxXR;ts_J*gdJoLCy2c<Tar^Mt*C#t}T2o|!%9pG3HTZ22V5tyR7FP`2!$7<df0Tr)|^1ss0AjA677+SB5i30>{V1M~>P>Q@iu73M-7SoKPY+{=dAT+wu7Yqu9Q2J$&kOp8-JxwoNX5ZT-YW9O#@hm2DqZ&Q|pe|-OB%O%><1&^s@_6l}<l$pr6=Ml@y%lR%yR3$;1cT<o~8}znIC|qOl?g7)i_yeabjluUqeine|$jqvsvmGiDlDDx_o$Ufcv~>iSAQIw1U5r_xOKaQvM}=j&D25tOQsv=ixdXL1E8A;|wDM@r@CM^<M!yKZc>Gsk!&Ml&8}cqoTr_1ADjmh#&-TvhnrWjluwEUN3792A6yC;P=upeuZpXLd_x@M$LL)Ns3N~2+5m`wQV!MmPFeF+XQj3Otr5eB4iHFQ^6UIyN1RJ5z(S^<Cu2e7K&w<Omg)RHlW)2^1-QBueo{)r%V_?BM-fnI|ErUO*G$D2d;~1gwHoU1?R`pCa38loNDf8?cwT&7;$GAaM;=vq}*uk*>(0!TQMJr#<f>DQPMLmN*r&l=TkNcg;4E)j7@=fbqCkil|7UW!ymM(UIr&fqdp2Vg@1|LOL?6kF6WB!<o0U92ws%EKRyHPLC=rG7_i(J`J8*<4PpU$OgM{POA#ET%@Aco~Kjvoh=d}+h2yEVzu2^dF0<DfD!W;G|+K3Zaegwn6XgmIWI)}~Jmp$Z?zVRkFWH*mZ)ZW0x6-?wAstA(01e?!FWH*`K5x+I)?-xrXNgG_<I$Q#5uHd{T#>TF^$k&t$%N8{i}Jv63Bh5Wu9W_VLq)>WY0EHw-s%{05A^V#EgbBoXWR<!AwR8~&58p(0!%QXk#Iy2`;H#~-;#&PU$J&c8+mRKh<s*`9Qwa`HY2cn00{q_b+bGDZzbNBZxW1Z3&qx4|9Z0<o7tSmA-8o2nj+Hhqv|Jr~i_C$N#q>~|!#Dh%#IIcbZn$WYaT`iH6!F{5Sc6H|Vc+b@Ur!$d&n|{#l=nBr_R<L3Wz#oaJzn?9uT#mr~sY^h&JB}{&JRSKwOsB!tT2ibLxEr-276xpUTM^tiY$W8pP6sU+>sIzdWFv*~iP%^uq64Fu<R=Bane7E&@1x|RjhBV2oWv-n9?2n!0zqpQ-}JPNg{8B7E<}!^_{#l<HZl`F(<*VjakIAhAu&25^c^8!=a9aB(9u+R-8|NMPXzC*zxnnrPp4O5{5-hyaPos^Hi!;4+@>*4S<f}<0=xjP-t*+0rz>Q8_a{aw4>;1}JlkbTt~3pmH#Ch0j&_zzg#k_#<gZr4vbn}^Ky#ObuvuArMSqLMwqa>m(8|^2iYtG4_M8SL=k#6#?cDP1q~3K*ddxCIiP!eL+ZP^ajq)FtKD7^a&BtGV{{5h>R1<*-Fz4TY_M#G*wx0?waPZyKln?y_^t-QbKoGJeR6Ra!JYn)O7)1%1hH!5<^e+f#rhHh9?hwM#ppc9*+JHR*b@^nI#{tts8(m&5^f=EMlpKBzV#%Pj9gJpNFAfF{3CrA55i9hXdik?ffQz<`)L>O(ZI;&_OQrzMXH)<4=g)s&f<NB-r2a0F-3hrB<kJVI@?y^#<vNFgT2x55U80imITiyYZI=maGk@dRyDNzUmCdKF5>W`y?N`8*EE6s|Xt;}x;zUpkdsw%s*004&Ox4U1mm$^g$Guu^u7n@t8^Gd2F6W@^HNx~>2g~k&V)nN%6f?(S`y3B^n`vwP3XY~u7fL3len<*&18GNdXKv@1&-voVFAp-5c%3em34V^21B%(G9zcsP!$J>j2RC^KLZu{epO!I4mzG3YL2*eX#gyVj;Pne0j8RHyv14)$XaE}U^V`zvUxMy?zPpyqW)!rCej6On`eFBXki(aq-0h2EiAzzU$fQ*NuBX7VwzFLfUb(c~NF;-U&bj3X+u)LI8@VFa;fiAX(vFG&(9{xLN4p319#<=v5>mA2hV^FhR~mqga~nr4T4EL{W-5rVVV8ZBJwX@%dV#iVbu~_&ufWmj;@3k`S89NhMPaiesdkZob+O5r+*22{uGLn!1;XssuU$o{jNs({kxll;#5S;aHs8RwIkXaS2~)Q7vNbjaVC?VRcM||-8lz3xk+H2tsTNpeW5Qp~l1ody?dbi3R;a^(+!XT`y0ADv7igK<+_68B8|ZJN`~Ih#2fvcX)#CA_E0bT^Ns{lZY7;Pz$JUO|h=QxAMelNVTBkY4xRs7I58BOcG_~iei2{S0s71mC_imZCyFgsnTE8rbSD&Qo&mB(5$y3_4e>X{J+mjB>MG-$}p_A|rsa_VtaA*E^-)%LcA{oPqN}OwwDj^Xb2UDdvyi^<d)9L<wi)0$6o<(wzo0Z2TkH*fTvF$?!Ji0lkk?{ANhKyqiXcG7?q>zfZFQ7q7C*vK~SY|>Q`@I{twabmshYaY<OX>17Y_7P8mNBCEOiZ8$?b%&{6IJJk`lq4WgZcKEMk8J^Eo1p(Kd6=G38(Dd3>{2!A8cSR<qKFc_Ai<$H6go98Hi7d>xIwhc0rGmjqD4R2FO%+N-ne1T&l?5G|t^x(`pH*CyOnn6;B$n{!2dGwc~O;nFhVYZvB#Xkn%PstuJ5uL4c*dx~4m5AyX|6N`fhQ&r~TAJB@W1$(h)3DaVa?6nEQXC&47QLIP`4EBT?f{61I>=16+x2oLcVsZ@tK`zH$+ZqW>5<i>ZbN4hCcZX~AWSk>T<wkv)O=aT8Sk&PzuW7DmuI!&|PA$E3$Zm#f$@EJUn4%^!#5)@)Xpg9YU?IXvTZ%2KmNrqM2Iu8c&%sVk%qV#!ec2<(gDr$K2TG{0{&r5N)QXY?x`E<U|r!Kq>VHohej72N2<1$tMvnk=$d4|k48@Ne@b+klMLlgg+0qHSIJEIsY0N19KIp`dOyq5MbsuY-y!SZg)SONn|)j>d}O<n&IR`NPO>uC9;;C(V|Pt5``UP6^c19#Gk>FmX9`!zwl&7BxsF9WQNVyz8ya2Qo(1gM2-r^Rp5S$fiR_=qr!fSPBtYxrno0=}f^>YZ26y}nk-NZS}E0DUs{32l7vVbq^F8o8%dt||tXb}<SYR&Jf^6h6Qx$PSi`e0PMLl1Zy7No5K!R+RIQB?f_U9`i^%jrjaBi@9LBOsSQcNNT|FFXl2dp9}_d?p(YeaH6j-gIWbzrz8;gtRy8?e6<#UCGCy1xX)ZtJ%p}iQm?eR^&GUKthRX<E0#i4aWP&ACSDrRsQb|8SOF}+XQWJJw;3tmE4iC>whwLVYDHa^bMRG-{f*GXGPnV5;~>O&C0u~vxpS>4ofq)*ot8u$13;r`%%9>oyySk_J3Oy3+A>|LN(W}E6)nDO3tB%uaLk#Oxucc9^aAjTh!bS{q5~}}@^bikFNoGDvXgv9gsBJ6?j_^x^K|Mz_a^beVQD05oLBpG3tk5}R3KHK+!q_WjfLuas#JS$3Zn701RbQTRQm(*+X<S#`yz?ELr%%9dZ=WNBAj+Tt+$ReV>KHe?qY!+91bVvXzP$r3s5A49MTf8av|{-!2et<>!?Ar=du0uFaH&v#m)jJEHn4rzfLyc8*Rep?huxA|Eed9M}mRx1Ou-mb7G<OIJz0NiY@qPEt>TN_bT`j@H}|@@azBl{0H^A9%j=5q_A;ce9;El-^RueR`%i1BmGY4d%dOn^*(K4_zX09OD>L{mCUn1j+^+=*xf#5zyL$gk=UKZ>&U(Kx0~%X;EBN6P1l0xJw@s2b`Ox4_^d44f_|uCPVH{9)%pm5%}&VZA0C{Is6CwoC57@?*cn;P0`2b9gU~osCX6|1ZBe_YH%=Cfv6=C0*h~e|RN^&;97&H;)0~zg8I@nPd`RB9ft<&}S!~zIuZ+ErLlL44x19U@pxA}cis83RvQlQ3acIn#g2ww{yHun}8}F(U7oaT0xb6uXL-U1|D2>!N(!08sd%{(fVr>I&@Ei=^Y5R~EXR8P>c?pM4z7G|C?}-bxeQ^;qG4*2b3vo{S<m+4bTgxX1@C3ZfzFn1P=o0UKfe5_(BOLeX`3eJ0E0K3}qSz)qoJdZE45mz$O;g2#WzV&s0R8iWZw?<HB(WM@4K2Ft>(c7g?!LfxWglnRZBQHa4Cf?uwrxVqAe$iU>Orrb_YT}Lbq7cq>(rD_m8QIvJlsi(Jn<V<=xB1rT$i0!6<`S^r^2AN@JJvCDS-}X-tY&(Oy}ERqXP`HWmZa^we`{w_iHj|IXgJMZNrGoivT87&njcDR3)o!w_C*8fBj!CUQD<12A*$|*nnK1(0cVgf6XXhEvn_~@s@k{;LNG5Pe8IWF`_tbt^GE&!f~;*Zx|Y1`a^W%$c0Tg7Dq}-S7i0J5m>|s_{1T6KG+N-QoGyG%3U9b)g1`vuKk4F>}gx;f7jf1se>34N0BJM#CMEQE<uO4COiw6W^oy~xf0j<M8S6D5>>4oBM`eSu@Zj&<p25E{`$9{fBWUPpL%ipXuQo|fBWY|)PMPzXMFwTBTlBv^cei%iW?VYU_R2QYWg03Xx0ymyL<HMqD#}$aq<nY{5)OrX%b+&Kyu;(!lrkOS5f)7(jkNQG;{wTo7^o9q_~H3rA=-hwgmJ00x%%$(QtXD2HBvo%X>BUq*Pnhl1Xto=E*2rg8^y3kOm!c%S8#ad+T8ci+l!=;<Z;C4FHcu?s?k^ln(OcVXxqhz$s(D(793>6Eh1uxMbztiBS!)VF7I%zH8fctY@{xdWd|S$~w=!S^iyB)f8&g*JXu5UrU5Wdo#Ivh2A!{M$Q*EYOT#&$(1b^B35VBX_Wz9*<)YcYg$B&4;hOZ!gNIZ%YuXT>|CbAdB!q<dn!tpT~u8?8o#=`0-p7Zo=*?mEKW^xh+~{XoGaZl**QENMeP98F-=d0XlC~J-ad@mz}&-mFT)e7+-zBBdxdvGV&N?klB?YVtde0{0gb#9J+Hdh#w^R0oGW!U33_6}hke%>*gv2c27(m;smy1mrrbUOveB@CVAHoTDaA;<8>Ri`m(y%JyE<RC40dW7(zVzAKn!}SJYp>sZ?Pz@wN*Hi1pv1{e8N*HC#W-VSeF0ONK`EuPpEiZIEtaxrtDRynw_DFC@kRj#4c(s+a*~rRKGGCwx(T7!R1GFM`!k)&a^0(XLt_I)iAJLZXLqkmg>aFgE>Nh)8?U55}cqcF=Jg*7j`su%yZ5X&5Idyj8>D3*O#eBADI%63cW{%yU3&Qb50pavRIs#l+<(t=1yP*`E_v+8_`!+(WP!6BaY-faO-MOSy-x0tnZJk)cQQ16XJK9FlG8YHA_9j+1RsgIy-Vf@_wfPA@UdKR@P>>urJ~B3Dp|Nu>pi&<F1(^bYIPRq-g2ROc{#xS$59r(5v$VZv`cF0E8+caLDu`AukX~BM!hQ9+Yu}u9GhZ_Ln-u3xi~i<)+C9!{?~axwXUJ4?aVaxYx^ge~c(2!SBz|bM<d%;_>gQk7H4rd!ANuc)+KR%M>!nvtyOwY=SkZBqq&mhq*HYP|+R@mF>})*Bkjb+#%-&^mEJ=Z|Zls7aXiVxAys3sucBvMp&_peu&;K_)i_zw{{h6w|C${!*}69n}V^0*V&Q*gyK(l16A>g``aul!$x8j3Wad*2+G9Z-9H*VU9xCsA8b*!gqC5_mG<Us_8S^YCj)y1+tnbtL_m9s*c36`kpNE!J4Kk(5oh)g*L;<?R3iB{@55bT0RavU0>@t7;etI^;&_$p;YKRhh{W>5mh>aA%(g^e7K|`Jh4~{il}HUY$et=w!T^VOE13?V_2DG9#kmJ6CB0{>pil>1wTfDm^EqlKVORJwIi=b_0B#J)-Lk#OV$f?>N;~26*zyyUC!1e;{zKmw>}=XITDdeyN$9%C^%5~$_ndq!f<zJF=OeUmK&A0M@;3yOSe)}v`i?8@^QBI=7)S=Azt~olteo3jIT_qL>hc6Pm{i0&&I)X>6j2Ps^U<AEN#Ft1Ag&=m^CvA+<VLU}j;^rEB?0z>F`<<cd3Sk%o0*wlwON%XKnVit<|{W@yMxx2>S4Ke02$FnFH$@YHUby>-{TsSW`WM&Zfi*|DZiFd$>@>R3~iuUU5Kfov=-%u-Hr@dD>qd#QZ?%N|1+e@_`$v=BahferxcXz_`4^~;l39J`s~-B;$F|ol6qQPF>`4Tiy;@^GWvCxA+Ant1ao6J=yWLni@DeT@dyjbxOB($c9mjEV5QXwrBtYlw#Yps1EW7s2Z-ff>03pYs<?C-zHX28Z$PojxZCtGw2Vy}WB2LQaz-ht&*18Krewit=-P^oaVia>NV3L)i>z5qL6;b0iwrp_b{)%LO0tSIWkjT2D;K=@OnX&WvG?fpl+)&l(sqDW(p~rGXyEJzE(w=<ITeiN5;ZlGML;MPL3SDMmKB-^cm+p;)#Hf44kAqxuq;foIWV~Wu)kl3$1AoGo7ejzzxle_tBabr7kFiFUMqkf9oQ3VohC!=6ijTbXu`8aD$Z&B!6x>;Xu1xcfhLGvlEQTE(Uzch5SWKitb-GC9urk4_)fG6vwF1+1MO0g@^Sbs2TMK-x%i`?IqzLPSvfl}+eu(nZPBVL?Luu}QIzhb-751k^+!ac)!yDtLgEKKkQQ4Z`^9ZKLU6D`WJ*t`oYeU<oh?_QaiklJV{TU|O_+YtKD$UA_?=XVu<a9QdD4_vw<Pu7cOskQ-WyDYEH|>$PVP+{JON9&&n_2f6bJF+DWw+Wt_m7GCVd^_o(<QOuNR<hzxf04Q7Gl2=h{pBS>6K#l`f+-aCS=cCx)Rnx8ahOSjy}cPp6?_zOJnGFAUH4qZ=xlfRqjKOxV>b(hWXO+M-hNn*9?;S&RY^^2FQgtE?S(-{WJ?X9FqD>I5x)MY>AaT-%Aj>DEmM65a-mlm8qSW1Z04McMldD0l>^EM=Ox1#c9p<%>JC7!tA$XLg2lSzJr)I#ZURgM0k-m|@!_0m{^?&hRp!k-{?C)@4>L;Y3`dOb%Vt@w^<C$+crrUg*zA3=XI%O3$mRlew7RVavqA9B{R9u9B~EDwV8TBK{sGAy7PZoA4}huk%+~)V0n4-^Wp);X4*T!7N-xVa?Bl*GIu;oB6TeHk`*s@xrw4n5~xEy)=azfm(bs!4UwPCcr5xAvlQLdYnC23BCmFJi)$zyur^dChDelQ7=rlMOoM5HC)}u=C4%b@zzj?j=F|Ky%&NNQhX$L)KzM8XE)(~30#pqDndtqc1NvK=Z+FoN#fbf%rB_d=RuUyAl73q%?XBJm*1Jl^qHJhcJH){lCHjTp~6Sn+szz4C9jS8BFtQ=3Bfn<ooZ=%psR&oFhzDOf3`6saT*_{(6gl%D_iRlcVC2qK*t-qnNi+K2rn_TW|dC{N>plHh9cQ14^&+s?9A(LzAu}6@-UB~uCk-S>#UKq!-YWM&t&mAnq(vRC1|_k%ZqmgwC#f{C0V;6NMmeMLWBq=zPHSuc-~X6J`SI@*;zVWas+0J2pCD|{hRAb@mCV1x~C>65(4(<HNfsz?9)gDy1wcCYYEOhk@-I+MDX0V)md_FkZ9G(CU0^!=TXLd+x#_ULp<cDqFfU$IXD`;tMsv|1%rFh>K<f*+&pCCg|grJ&L((1dbxgtTIGvMK5C?$9T^g<s7{R>9~Nx!X#KR2sEb%XKY3Z7Mpx1&Z#K`2pZGLB4N>?U-KW$WDWK2#7AJOo#M;)CPMxlx@FLaV!IKwaL$b2hJcVm3VY)b<;DWE}#Wrlzr`2$GC{>zz>h^6hq@Ph@55AmGmpp5Y>SlD`Q2T0|CfivZx&KYuC0BCSq^(0<B7EO#)jVU#-Hp;l&Ou^PT%R)}HoCL#c3I+NRZPLuYL~k9g&b(tqTeiW_m>-w@AZlruHab9ZSV=N2uxy5`W(UXDhxEG(NaB)F{|<H|LCiks9l`puOhf_$!Y>iC48riXJHF*(0&J}>(SiV_y6!;HpGiTSBMKFivh5;+)BpSoGz12fQuI&xPN5gfTga7Uf=HU5VqrbE^!HYlVH?W`vzpT07)Nqkw%ZTWXo;yeWpct%UrR2BX*P3!T5FO(s}G<8Q0$-l2bn)rH6bQWwm322o&dc*qdRv`Lo6zxET*_7E{AW+V9PfBNeq9$E(@_BZ0`$+HbuI=E2wJcT!gskBp`M5MmbbdvGxH-<&#xb{BAv;uiL@`Wu@)-)vSm3sMdxc$kIDlZ7kgY_@T^-{T!&)<13WqaNz!W2l$JRdaq}i=}kQaxQryoN@k*!+Oh*ELAR0fTz8I=~J9<OaG0@RcY!8u-?nsXS8E4$Yh;ExqEI4cgH9xD8su9yEO1$)fL-rm@cZc&w{fz-+1$j498meAF@#NT~<f3kLGc>{4`Z0HE|_no<|M|xddB?6oOessdA4KdGZ9|7%8hj=-gS2!=&pYwTx3Q&0P3gnYC8a52vUf^ZP`P%Uj4`%UImp69*jp@|+np<nj)*jG7iL<I>^&1>SGelbc(Qn*d{~SF7&{^I>pWO%wI{b$q=aTfR_sZ#6?cT_Gyt>Oz;wXO36~h}0tU%%BZ6J?>ASq%wuGEdzKu3T>2_gp%~NUn++<@a!}lx&WA#xngp(hGiPvb)KsjB9}VEOe~y7_U*BZCfhcZzBX9G0<_YAdBF0SJB|C8yBUStSl?|&-mj4K=tl7i_L?ryT&P138GoHDT{d7op5_PGg4pFzU7xfMZ0F~^=u8>|U6@q<X}VcG8$K^nB5UA1qD;`w%nqEa1Tho4Z9~Kg&Z>x*TK{?A-*b`5MH<nm3oN%=xT8jFd~6>BNjEx&!ltFDPSh_>wluYZMT+xia~l}m1?iiwfZ8xdX9(pzrF*;Mze%PTq-`EO(b~ryT{G05J*C8Kx-oZDP{a4LO!csX$4eAnn6A>fm)VB&E6)UC+mHYKEyaRMZs}B!i>0#dLpXV3L`Y359dB7i;%*<svm`NifZErtqg7F3m)5@G*~_dh_6mS8ahl*)26t)<{wX6Uz90%4kPH8Kh0gx?j?e#0aO_d!ks~tNzJU2)NVP4E0DZROu+a-BGBf!6%h@QQeeU<qL0Ki6?f#K32w)etwVotXG_mdJG(J;$zusp*ps)eL%2Lu`8@->el~*VF+pXGRP4#Tz3z0%4AEEUP4a%RtKDs>hn9IY85=WQ++{fn^gOgRbkr8;(=+uvz*n0G`bU9BxN)xqrK>NevkKsqzpa@zqwEth@ee~4_(Vco9eXfv?q~KJybh};>q_ts~`0>A1<T<{vU7Htwh!XDjZHn=LE7swbFQL{pnh}&9EBNsP_XiVl3t-vOu+3kT7i-0#JDQ=?S<0P_QGp;(NbGz++gs?Pno{b;0G<1$oFlUJyNY}jR(lPdkK-fw?KKbG;4sD~*V8xu-rL_lZ~fz=hFR1*dHv$c$>d=!Ndw(c5)1VE+S*BH19!14+I0;$%kBTOa#KUOX=>=xLRuvEQxT7~6A`KD(_9?2n)Uav@z=-w^2^`<^0#09T(>{(Pk)22o#QeY%IFpf{Nq!D6R3R-RljMGN6@FELGlgvAOHNzUw{37zkkDT{|pS-;la`VGxV+<(W05%bE`R}LdT}a&eoJR4n;aN9W6MgjlvX<U$ui&{l-nT4@K3_V=g%ag?f5mBnSZK>RM)7E~xB!iJu}*Mjb6#^aP>~6Bk}6XAG;SKYy-2eygwC?x29VWsA;5w$SQ^@4RM@?^AUOGp6kNCdF~Y<DCEB|Ni*+$Jb!2x7^k5<7rv^P5WsL>)^623=LtfgyB$mD-!o|_TXqU8PeKns6Z;)SX@;iY(oH30STHo_gUvZ=X222mnH<D578FS(QD@iJ=WIZ8Vf5a=;xl)HY`pojZVCEft+~4RY2ni9!$?X9`sKlTtK#blIPZ{UVJE9_D};n23)S0B*6j>zhCGt+6V1vam~rCwzdIUgJJdS>+%|GA?vGpr9>8BLv2WC+M2c7i@NLh3@E0jsh-^1&@qT?ZiBJ&(}F|BZjiSrOSwP3f3l?zt-1eWDw*ej9Uo;Ta_)J=GJ9{n3leoZ(B|C~q|*ky?Gg&tSiF0{bT9tEDH~w$y^xIs;5jn0D(Gy7iiG5C>{MsFzz}U60Varqcu*H(mgv&j_Wn^}nJ$W<#*<VT;aQeFZO+OzY$B~Z+B3Yt?3&Rp!Y>~GRoHM9#_oog%aRgJ89hozx%0EVv$|&5s0^%EM^gf3$q;t8@fSMOa<|*@?fAX_Ro>5t%)F9DmK;RZ8id&HA~6h!R)^H0VPC1DZ+7A#Gu(vnQar&%XmoU8v$-qPOZao(a&KYF!?c;hM_YHda+N0=VdEJ1*p9cGTXM?ak17R-oxwOpXuJ(?swY)FlTAV?F;2=nJ4fwp2GB9?+LU-Ohm>_N>_2p0CU?=wm$P8hAzD$-;Lqt5PWj`0XEFnSw6%QGde?~p%%%l7*Q2G2T@<O6-I6D<>2kqGQI!L2z0a6wC1Zew$EvF8DcEk*U^6-la@!(TnA3(_^5vOx>Dp1-gE6Th2seo9w~XV*K_y?>aO-YOvb591k<d7(jEq?U3AT^cX&|BWD=}dlri<0vlS8P&$8nh5+UgA)Z;hKo1>E=TSb1ZiX3gIaar+IO&xS5EmKeV`IH$DFap*XmIoS*TUz+R9@V)qTH_mQv0Dtq9q%}*Y>Og8fjtPx+`}y3em3v2NMejIGu6q}<fp5*bLTgN~Fo1j0RYK(0d=U^>q-_Yf{}?pOa=&9G-u+jgaIbLOI1etXg#6$+lAxR|Zqt}&y5t(=*b-TZ{%Qfv(-n@6`xE1$1k{Rgo^2=MD@{YCMNGZY(aw@F5#<?1o>(+o+-f)pG<QkpA(Rh5^tYI!AC`3jtz2D-iL#bt<PG3^NYihiom;-0)Zl_C7Fa-;h@I!%PD6v%C`&?pE1jVH$6tT`{Xn~`iNLt|`S+i_sJQ9vr^1XK_-^Xlf&Kyd-PbpuoR7tCKR#|eU)N<YUNlWZxPb)v7sP}?nKc>RA$g8LP@TpiF;&WR;gd}s2TV#~ba}bZ<9w=7Qh%;Iyy0Lp<9cy0@XNT&J*M}eME&K@THZd|Hj=tojkQ_YZY&w^>(8eC=g*)2z`P*5_et$5#MTuoO-REDPUXd(HA;~PdH5)LYnQ0Z6pzL5x^9;VYcqf2*}E$Vm6pxVcoQxeko%nd;_p<uri5WZG>Q{J5p$q5eNB6fmzc7}7ndQG{=~gnZm#kN*aCsYhpZ4m$vGqfdmSvh1BwvTBDqXc&-Qs#g}1F1Udx=0rcM{ipQV0C^7jKqSaWBdXjdi7vI?2KJjf6VIbAII^&Bk+nW<4dfEJ&<i5}VxZt@O<N-=3am8M6R7Jbvi7U+SVEzC?)ya>F0!GkeMcrJEK&H)X;9(aCRdi_hV1kQKYve~6l+C#q$4rob%`#U%#%TDh0MUhJ+msxN^tAE#1EKR#$Ee0J$D)n?3ZpEZ=E0MIpCEGS~MXtjYp~<Bk6$7BDCAyCJ8ue<As})RZO4;?sVQ(gXrNtGg&5dK%jt|Kx`h{j}E+1v*AdsRNwUz0raq{FyN2`l(ny#+Y04IyWW=DzPA_41SGg-M67ie9Zv2?2l*sWi?oa>F?<Zf1&0{@iG#nJAqqYQ0FT*9)h+}qm50F3>;`ywQ8rZHli9U0pswrYV@>PP-^mJ)7ii9zolv_iQP>(Tx}*>)lErUP`nsnEKp6j=rO+vvXkDM8CO0KQrul5}PA{og@s7Sl%A6++5>g0(aIuBB)dwdh^$PV3YZj$7$i^Pt`AMpJvfnkX>12@7L3xOdCz(FNkd*7|P7o~?1$pF5nAaxQ}H-%S$Q_N1{}QN#~jTb`E-x2=-T{O`WoBxvPh0*%-?*CgdU8y*K!MOD63K<?8Z^L>kC8m68_a*>;r$0U!&&Z4pHLk2v$IjE7a1)7G8&=WKXd>3eC$bA6~l81WYjTFZdnfGqo)-DqlA2OgbFBK2dppa5#Jtx>~q}+5oT*||jP8X~xg6g$JA~EyrGmS>PVmeLn$9_;^!?TXCdo%bIcL7Z9?===Dr?F%#d!BR^TDGL`R!xrZIo&Sksy-}3_F3t2oT)(U0}CEO{`;xgAk)}sht@6u^<)88g}LJF^<VPgt{s=-$u#ICcI%hC@Z8&+v<PJ_BLho+b+x)^A)7^Ofg&%u>#BmSoqAkGawaN%CESQdaksU(2`0G}5?G^JRbus)-v=wy)}k+M^qsrls>-W5)wl%=w`hhja^pMJBi$4zHxg5GtdfyO+ZDftbIJ7E$VL<SvFR3Knx@(A5Ieg=6DasY_za#(hg((>2@0_x(40X283|Rt9rc+e8CG$}4H(EX%awFxqvx^NnUlR$3+3pwa@lO2m*Q@vJRTwQ>3pA0U3eYBFyMPREX4aXE>rbCy&CuW44H2>aFYt_KAbYMVCtnc743{-tN>h_8v4>X2zf2-VKmqE$iIq?8@FXFfdLgRARyDGuH(}6=Vu))pA@`LhV7|YAjV6mvN}q5ALW;hUd-y~1<TxTl_i#qqU}nd8LbU;a2RIP0@M<<=c_mAEImW*2pX|Nm`6)UYy<<EfG;V!DoUVUU#n!KZHyCuK8g4V#0MXSZS~Q}J+*RG4GI4I&LFbo*11mM1I(=Z3U_FZv@~f|B{8%GV?{X+Sz-_v=P{4O(}>Tn&XNnJ%amHFiKGUrIuE!Ev_kaJ*12=>g20KFHP_TC(7Iw*QpfdSNQy%NPXLy*H`d}lbBXCNUCpG-kynwpLMw{oRUndY1C&rzT#Q$OiQiO?LfwZx$BNaimW-6C>{se*a}&I5AKKQ{in=W4;H!w^8=;A1a0A@u5Z~aHZ~=x_bYxBGynq$hv?T0L0ga}yT#MuIlKW-v@Vv%o%XF!j1(>Z?wD__us1sS>m@_SNM=OEp1>hC6haEHo30qd=<?!`h5Uo>WC;5yBQxBlsOUB#h>C}JjP2z>a(n!`gulDN}ybf@vK+NLZ7aJ|<Ef{2#Y7b69G~Sk=gSgSRKM=p2p!vHmlDIqMl-#O^O7<wiY1h+w>qs+Jv+?0B7U;p@aB_~e4hgjYMMB6SEfFgh5`O{w&&9Hi8bo^@+h70kU-4P&EO5dybKm`yW{wfQ(I$NE4q-|6uX@6GBpCQkFz`w;Cl*?dqnlCU%z}?9nH*$2!MzH;1UwHOKRirFTOZA1@MhBjq_A;ce9;E0+G{;;ObXi3vQ@Q=zSmpIU+>c<hR;Bwx8&mJS;;&L)UNIb6j^uslmP<_K}TYD7Ox}s+TU)r*MKJiBZX8UUB>ktsM|e2V&b#1a0~jOJmBrBs}wu$RD4a_=pP=Oji^1H1SN&?S=bp_%>wQ2)PvACR3=fAI>8^=I9W8tX2!Q+GZjcviPso%Bt1?|b6SpMRQB-lA$jWtavlq3v0W#>GWJ3aMTj=sa_;kkVi!g$hTk&DN~ee`2#pz2(0D&=enl%V7}|JOowxvHF~)UI*ch5GEP1xlVywnU@9JLe30GB$wGF(%b1;0T?L%Umt=eJaB^*BaK2-R<Cob6b#YND>)QiEd-!tu#uW#XREuSF36Yw(oc2%CCD?^$_*x~YzaNMWoD-1ZTMBdSfjxrRGNKS<ercBnYh(7b@T2Qz0`N21bj}MYqjjo0kUG{Zp^=fxt;QN{`bpa-U+NftZC#kb-6KV$81YuVXdiA__;FhU7K+;&JrhKY2<*nr5PFm!N->5=IlQZVJTBRx+MJPEH2DODp0zpU#bU^clKL}<z-v%2UuuA0Gva|O`+^@-;<?P`2whbdTF9MiUg~^P)QkAT_-EI+U|Mh>pcro418+fitYqlGZ3lv(f-si6w1*}E2d_CTB?;e~vwe<-|b|yv?$E~&BrdBvEmi7%p<4b>tZXCI=DaYbSN$HBL-Zlb@TX$X_a`=3(8Azn$pP`kzJ`k%r5YS!w3A@?Tw$}fyx$ja3F({5AQB{pR+qdNzqg;XxZ%ueM+qDf3d!>20brJ>Jl}l8$c8oymw!}*K{geOaXZ!2le*W#3-+t=F@uTrJfBo&B6H))=XP)u(mybA^F4JT1hbwMel!5t3qpIn9{GnOAFYfNqr;9F4PshnO!1D8S&8JC#?E=Y(4+xvyF<wRG=SqhR-qXzegKToQIFRBV&XqQ~eb^Gr?+d_yv`53`nHpq+#xC#G*ppIiSxY9x>6j;@a192e{X!aa$SoHo(C)2=AuRG4M2gp5aWnut8oB3fD^NPfmxsNAI|8SS{X*wTWlYR0@Zge_dnZOUX0i99jl*|syN>m&)>sdbk5gIa**D9-tE!qpt@^sGQ0Qxk&}eTacdyXf#@5LB;zq5tnddq0Ld5E<I;}FmD|_tAdrga|@gZYTLzs?;e_3#_o}F6~#&a3V1n#LQVRlh<^=SO+?h1I;GkQKfbh9`$%^{9)4sot@(`4uHbQHA%P{%Yq9io}p-+TKoZUb`<=e-P1ta7tuq3spk35kWbNJy@B53ovxZ3Q&)PV~I$VjHt8TXL?{*(B(R2_N=dXJG$;Vi*Wk0HiXXotkp{1jt6i27*oB#-tP@@otp%n_o_|?d<A&*)rIvX-L;z`vWoPsq%=mRJ_HaxYkzTOcns#{_qJ;rJSJ7#9>+fPa{#aWIUnbb>S$6TAQ+0p=x%9Dx$D}-xIs2xonqY!BG9mY}lH1F$I?&)g7JLdpgsiT%O@MI9J2KdbxE7e_N^(BM;^X1x}lXPDyZrvc!yaO<ma0*fGyJOEfQL&@oy~GG1S%9(`m=Kq~Ye9quBJ#?LurB*|iNUQ$xi5tuuH739~&L2N`{T}79=fs8nk_rR^IMP*^BI<dY#vQq2wd`^hpZNil4^VBT$5NBi0y6No51<CuJ0))t4pj%m+-NL?v&nHxCAjbv}f{nXoiqL&E=aHhNJ2Pb{)@Ru{uS2iS6TB6a)BzBxh`=G!i-f#DB#k%#qj*rp5xP#k9N1s#5HAdpIhLCyBMhITKIhgBe?RyPP2yfJ<NYzBj0C?wL(kQ}p^3-8t3HlJZSHwm$>9N?J}y(pB+rgjin9sUq>`94w;krr3_wMDG*q@nXI^jQ<8X(ZAJES+SG=j;<z8^G{@mK<YpGJy6B=Q~Hu@oYyWl@{T;JMNwB6o;2Myna2W<+*7G7sd1`vur;SE&9EADTztPC58Stu03y(1_SgLnUE^mNIhp?$DL*%DfYNmtsNx7lxKES(JO8EjXB=n?_#En-u|a7O|>A?y@kQb(NGLtOJ!-cpI=+q@5Vg#`pSI0zhjd4~)3T#4gVvWFX~U?URC6I;@cz%ttsfmtxZ02Sts&{QHd+#q|ZObG)V;;m#lgw}_X+!p5^sFd`ct%5=wc-1OuRnF(AorGQC&*YS90|B@(BzMd9CW}F@T`BE^&tuC^P@Zgl?fDOVW3aPn&uHb+BqgEiCf7^EaNTqAwFnYLgrASl!U2`W`^euAP-1b;L+LxNw9l71-C`gajQ(O<RkCt!cjaVo@2JZY++b1>?>H;4!BRvq5YI<<RwaQ4RD-yN0L`DYOpzPGia5H$DwhP<55|O6PUPL?1#V_$g4JeKo&Y5XtedagWbF=GTdIfU-T`Dp8@)*JJlF_a?0=7IOqvBcgS)LIy`=nFN+qL5S~IkPW_2N^iqcw?A9g!3WUbs($w<|x=l{=;D&q(Hnv6VRADvQAvg7ZbG>7|M80fQKgNl1SFH7oaamCD~JuHS?e9P$9VTQOmxe?5b;h@u{04(NS|HmUNDC5!{*V|Q!DS?$%CzMj5GTI{dkPM9eKph~Kd!=s`U8>^JY52N5*1rM8F5_;~$Ivo1X^h>cQ_C5ps6K<M<C&5LtD$QvI>xCqh$6`v3of!|H3eN_kS#Lgq}X*VgDJ@>)|3&EdaYdW;xp}4Va48~*HccLD@xk|T1j`^pQC}ZAGjo3>g7~0noHEwOcnv5SOnQ+yjxaiBH$Gq4OWjM20Mr}O~A4+&E~-1_QU>uAs(;TMr>a1kNoEAYOgM8;$GmDy?Lzweso|@taX|UwNo&$wW0~n7O6O=^#_~S`=aSOd<L2zdPxe?xkp=q-a%j<MzIb~%y~>yq2N2wD$MHDHVm{&MaswFyBsX}Fy!Kog66z;^<?Gjz-%XhS+zy0uCxoafkjcemv*bn%hVqckyd+qI|+#&^gvo{h3pr%=?KBW3Xv&2opMs=&vdq2iN=v`Fpjxhr8HssN&D;~b>MeWCBn8(pyf$ZV%?I|f8U90l6!A38M55SQaiafaqt8z<vzPyq){BikEfJcl)EZu^qBN@jC(d*Q@&n+y8Y%4#7Ci&i=JyQ@n?At5LCL1*1*{*(VrNG-rR;uT4E`)TRfeHhWWa(*1s@3<Bx8rYywg?#4}-6t4KHaJZXzc#cTFY9Az;IM933wudlLp;C+veJ)aGvII9!1^cCqUWpiyO0;gLyAxL-|I8Od^T#R)>a~EarGoauRq_UK0<`%qBsFpA8&|*l)I-J=V)@5-mwd+h-h7RuW*JFlllLRPJuR6oaghmR>Xj_+AwS*IKkuo`SQOEOgSSHtwNqM0^BQZFjrYJqHs!rx&eupg+3v<BL#<@zq%BfVcZi)DNn1n#_)NR7E$i2>AWl`5U1AHGxfrjr``~<Ub8HF`J7hWF)pKa#Hg4=K&8^sILzGJppZuim@ZUk!a%>+jPY?=V4tc2hocI$EWU?unxwDSb}0`dkwyO^k(-bKAI-4<nCkJoT@Bb&cck;hv@Av)?B67^mPR!H%Y+)-Dl&7IwZ`z3Hi_NWLQ0oon4PMteSP$h|HH#5JWUY`e1PJ>vFy)-8nf?a-RBGYGbR@uGNE=s!k%7qFaX>T`k_>{ah>WeUQr6vU5#CNKt>4B~mg25EovHaP_ki=<xm_pB%UaV}bOWb`C4gwu->}E!JD<QnZ(3({~87NVybs36er#w(~fv_{Lzxlpw^2x(IhPuj*2CuV5(he5_g+G(U=V+3R;FqB7k}ogb8PK*5u9Rf$h9HfxO$iYqnE2i@f8u#h!TLCS+Gc0zbjcBzEh1nfq4#gDE5%<)l<J<EphyVVqt^htW3f*o5$O7+_pc>5_eAFZm=M8p-&SYIwLzj)C!4&<*_=li^KJ9jlnwEaql$7(xa8ny@UGIwsum3HMXP&|33Bt0jTg#(>pPp^`RL{P5o(n$D*339c6MY)tfD$Ka(q~@#iRApMxria{ru!*eHvX!pS;;TGk)UJ_%uY}b9A3lZ=`@e>sy@I`4MYdS2}gNg2Ibbg9lGuhz-fgUh@>Lt%T{~e1Z$Usu$a^QJ+@B-Jw)z>Z#ke#gKkRi9PsoLS6E#HL9D@eM9Z5ZJKOnb>#jxZI@ihU6ZyBd5Q3SuT}GmC3iPU8#xDwMR9%3kl5(XzT0JqlT|SVQ>$I-+81)5U5kFR#NA(RJigZ}YPf=9Ew{lZydp4(Iq7o*%d0TZltxSSG{&sPv;U*7W}<d+mcNSNz9p*(ES2z`HlBqo#6kNVoUTW6XW#$Bf7uW(23;X8kSqqk)^aNuV{^JpIsq<TeBl0(i366p9(sMd!$a7P>$$`w<V}K6U+o)^*#abe*hLyW){-r^&G(rW;VpB;_Knz0RtMwPolEDjmt|akhe%HSe3TyYZIsoH4I)sS-(hct;pWd8d*Eg~xLHgMBWb@kLylC`ZXB;_2aE(FOKZRNDwqdfpWjJcRXj45`a_6W#P7ku(0_C45ZYb9L5f@0%j$1z_I$Hh;Veiwl;B|&E>9M&l(X5!<$jNMgjxTz#gBTZn~$Mh5?9Ung)NrSCCj<wiEzgGHxBD9L$Xx4Kmnfi2BuGOzAgPXCRe4YC%}3yYoF1My&#iy4(0B-E!-WWq@WD%GVId8e^pm(yJ5Ph(mo5$-hAWDFESi!<$uUR(RW!L$v&FL;quc|k<`SMlzARGB;*ooAyNot8KufSPUOiGgkz+v0-<wfH4c-mkJK_wy)<*-b7j_AO+TEXe$4L^JuYt{gDqomb59&_@XK>%)R4<N&@yUTw2VuK`xkh>QBQ7eJ#GSwsa~zVC(MVzX*EsM>(}x1er)+d*}c^a`E-S-jH?S>DxW!G86Z-N%rk>F*z~wRfs)D;&bAET=_s^OViHQy*M6xS;=r@hbm#(LTIPz$(HfR%bk}*VVu)Po5HqoG9@)3YGMa4LRQlRr2@B9l1LgtCXYMraWA0`Ya$|kB9eKY((xV&2FW76kM024IL1g@OvUJ&i`FNTiU<+cGM|FMDKCqpi^P)3p40K^q`KRe-^=$aOOo^<4_lPn<KQlXUvJ%8h?6wUND>$nnVru>8fq&0MDi>))r!KJEZsCp^vGK8e3?$v?915G3qB>E(IN8$F3Kl8Oqs?t#co(E^z5;5)7@Z-M_muAKj{hc^Vvx3Z^h9ePcXZ8AfA*9Tv+2g%Q9%vg&ob4+4jwO2d||pu=U!$T(yu%dh;2Xq_qP-aF1e*sMJ|@gwh!UtjS(R=t#rI)8Hu}n6wi{x-~no1yN*^xja^#%if1pgy4WiK#>8oYUm4t~G5Dv9p!k9)Y(Osj;}tsl<2ye8Gr_S(kw=cmX!`=@gCW(nGy?S5j>ASTpvcVN^Dk$kg!Z}LKL=%%Y_|JHz94{I+}3)MOwq))r_=aM>HT`2{eZ#-2rElTgKhMFzE)nH=x?`bhc(r+i7!M7m3)NOH#8`J{`%<h)MG9WD@q(){&OFnUkpxG;YLQ_NuyIgYGUir%hKgM`6x})-U00ok3WVVWrHGU#nAqLjrY-4A4GTRee}6PK9Yh{-O}xPNs!itVdBUCT9N1Y#&&I9{2@xX<F_fs1Fl$yTfT%^+h|5mdaU5b58NM2$Sr_nOT#vQQC_SShwf;GQfDc5Hbw=4Kq0a7{cLZck7`P(7Xx(en{tlG*6%9vRaoscbUu!c;J4R2bc4efpIlGh{CjVI|Ge#wj~ZrC@8tE1FDH|Sxg-sAM@cNu?`vx(oekW@wrJNi;4HWQ&&o{=<)*2jPYY?0+)qV3)=osErcZNm)N0n>!^U49_scJT|I6Qg`E%X=yg&U7zIKkwWGJIsDDaO@4Njo;IaK|oMIJ$)jt0p$+<*M@FMs{@|NZ_Azx^{XWQPYw`_It3c0`M2cF(QmlnNc2B0F1C+Bg*H&~&umoHhzmJbu*<QuP}*)jkweKaaWO5ESa^fsr5poU3b@ZMmSb>m`1QJQ;PgWYH6dI!s)6p`0<Sp8ou~{`jrFa=U{9=9Vow7uiCq8@}_JJ-$!XDa@F%>zfqE5s!2JfB*aA;~!sxvEFi5zmKP7@i*<KHLQcnwlFk=xe|s$<*i8E%h`jY&16VxtDyp^Y-4d%iLea;Oa&xp;@oGQ|D4Z3S6`YCfIdW9JV&pcAM{vTi)$>bq@bUBQroaNu{1jI)&+9n30DD)BX}@9^LWrdiEshg@=2ast9tRFY}rE%@ECBpW|9O8IQ)L0zi1z{r^PiVx7ykUXbpzdudmB%u!XFz>Xi~%fDN@Fp=oQ@ZZGPt<1?U`o~C+oZ$rl*vbhb$&QA*t8M{H=rYz<D`2NY3LbT@okEvvy2X=gvnaH{45zFko`7TJ*?LeD%Q;<#@^tMYVTx0R>0n@$s1E*|&!S_No7J%o-%&MTX9V!x%x3N>5?E*uzbp)6o65>H!j9H>fYuo!rg=M-Zh8j;&WrSy0_Ov-G+pvkW@@UWS2D58MzX-p0{8wSaRT#S)VlGQcG-dQC9p%o?_Ri{>X`?c*UL8#dm?cBl-Ns+&P|MwJ$G7A6{#SWFBQo<!8d-7>S!)ntyNkpyBw8I(i-vurioV&2hs<yj#!K-88==wBh0W%!R4?Jrfy=#xEf3RX4j*mZ-O5#-Y=n(t;A1=9Zf?mbgFmViAa(}h7@_etys4g4^-MMirNlTX^Xwe8w;4dkxNB45!5mW7!La|(eVN=vD__onQHN+nJ%c}|S2*R5`<=-Q{L$9(P3v7J3NV`%<Xn%IE_P9*R(4CC#HPyyA4OFTwDmq?rj?8V8Xl{vs;6MPQG?CsFvx9-TwzWda><ux&ZTQdZ4btziXhw|uHQ0_9|x6uX~V6%HObOW7e_+lpfWOM1ti!$TBm`8(yzpXahNVvZ%+=P3LnQ|c5ACQaJ)5c5*2XYw`1jvg_<>gL&WVjbUqup&{$&p7X;_yu;!%_uHb>ai`&?~J=}s6&Ny`39Gr*-|1Zr$XSitmDjssT471b(*<ulmw}h6@MpbrED)EP%*x!~tMm837Ovk|mv282N_780`AVZtRaZEQA<fkyHw<Y1H!^_dvUs4H&<DVL?Xs|vPzd%|$VJcY~zx<VN9*-HL18wvx2gu4GZz+w&9+%VP`uw4S5v|Q%I0;?}5uXtdbjJW(c|8K=&U5!GKb<eXX>n_@(k8U@C87szH`ughWIr^63DrIIA!WbrD(i{ZZWmA%rBM`#cvvzHHu%sXjFnx(zasBaz47M3<q?t}JP#^#l;Advd2VB_QH~LvMeV8%jCs1k%5;BX?8AUeH_o%|J%6QXsF;+gP(9jNGPcp2Q^KhvhHq#6o(9cb5(+rw{1W{wJ{!YAexQ}BODa@m!i=s5oNg)M0<?3>x06~LF^w975EGB!dAEs2pf$>bR^Liz8usI_KmUH@H>-)jnbPy`KYLMdO!xJvkfQ;<n|d{&e}I11Ml;W^WlPa?eB5~ADP=HTrcFb*Wf1xo#CAtLzm4vY%=VxHi6XOTl-tNBn>-Gfc+TkZa-qk0a)~5+@~Swx?sGF3&A4713>?Xnxu?uYXiD<(XRT5TZ5v6}ug2OeW;&KkwS&*5{^!r1|G-R8y!T19L!{3TynIN!3{K_6o;6B}3zb(W3U8OF<OYq!sBX|M6V_(_#<O=<GC5W@pW{T>!$9D9_KUw$?J7cs>D(wz1VzP!#Yk$QL%hUPArx^LQn6>;tL5fuJ%JG%SbWI*78DspvftOivOAzCpl$Asxu~#x9@Sm1^8^vjx=a@eUZ#FXDz*X+aC2vF=a|p=(v>d{G6dmI7mGYTN6P^hVN?&G#pmmzhqi;8yaS<9+KEqZful={zG>p37lDE~<RVeL2)us5gE6AGEOt!J0S&+xdwyGb{Y&t|&Ue?c*~PH{+}s2Qw1~|89W26SCwKdz=!8<eB2)m?zw0TFseKO@gMcWt+46>v+)20gR2y8fZ6jCYI$V(!UfNMH0Ge8&>uC4D-s5To8;Q-d=I!Xs<gYXU8|OBTT(rbhOw3deVZ$!_C_4v%6tT3eMOuxMCqO+~UHmc`>Pii8vM6kJw0ABNur9WCftxRa*0q^uw{DEx`n9W8i4mOK&6<MXpVGNF+P!rY9`A@tSk{$SGq5oLV}I|y>=c}7j0$Z>#x`xaT40sloxhwVLxh?j(fbFjQ0~2bw0}@WjYzKV09~MEYIDa%8g8J!jqdxOQgVIE_p23wNmnMn;s=P$Vxa)LLP()!uy$tOry8xI7QM^eX`L$aaVs5b9<-a?Xll<_69oo0VOq}y_inlSx<FjmTHn9kGqUgcbB9xMl7zPH-%S$Q_N0|(QN$0*zYhP93ac;-cjkZh-Il#5?lCOA!MP?Wukr9Wm@09~rIMkamdNj0B-1eUERu`dtUM-pG<Ft^Z67k=(ak}Pgb~~{WTgF|N#MIcD?{!JXplVA6K|wAp2)m+<F<C$cKMJ2oq4Hon+AoHGP6;^X8VYhnDKBa*LFHxu!<C_&>6{s&9~1q8u5zhEyy4HK~W{oEXeN7;MZ6OFuA|iSe%^3lCjanr0b5eC4IN5Fon<Qc0pG<6o%|$djJxIL-v6MkD!VxscKNu*lB>-E&=sq0at~&;_UTb^5L!>m*dGa=p}aRm%P=8w>fFq;aWHhmj3FxGoXcRmMI4s?C7qmN_2NRFffuc@oiDUjd&Dy+l?W?B)38WYgDV++uri~V5K=*^relya~E7yYdhx?VgbV~nqiFG_>T2RHwDU##MB(CNbu2i#joL9GW|BP(L{c1x~2D~X|_AW&hF6m5B?B7gQwErkC;S)LTm^$ClLHc)&$><`b?7ytGLG!4CI-~R=Sqm^VsaHc8OK}>gcudRc@Y_;%=op9wGDTe4kHUcpbtp;CneNz5X;VQ}sU^6NSz*WWL$JO)9MWa0-iqsh4Jiv@?pa0&s0=;ZElu<h8ViQKi6q43>9W#u6A%(+C1GZR+}$u;k16Sx3vK4)H%|7Krf@sw^6~lU__`FJ^W0f@SWv$`Z>)(RQWKjMfG^IE?ZG0@Ol9cH%ebES(_i2pX|NEQNcGf`S1}z?T$V6-BSFuT?VAHpU4+pG5ow;)4&PP{Gm2J+*RG)gb=-&LFbo*11mM1I(=S4R>gcv@~f|C9yCEV?{X+Sz-_v=P{4O(}>Tn$CV4F%amHFiKGS$|6(pf^T}XP=g!3o0w?<VGN@IcbxHz(&r0IO@2j-{ENO47#eL=yn|ZpLNnty$27QHA6v?YVB;N)op{lqTuLKjnsT_s64}Fdmzyf?m%2alnkpjMw+n#3o(6+8t)MYsbUuApW2u&=58{odW?nqt<7hrg0rq-0s3z$DnOTzFP&}bTmt2hoXxnK4U&ufgfOqa?Rf!S(Bi!a-P`a1@WIny$Cv=W$J0A5jh*g->(uw_MF4qxvD(K<zTlFx`R^#IzvWW0TzPW|WJBwjcyjbx4UYQJv5>i~xeq&$fGVq>?lP<>C8Y7b69G~Sk=gVY~se;|H4LGyQCByo4hDY;b-mF!W3)2^rW){$ncW~0Nx5o~r%TB^i!TrEJ65OPRM#L9)lUjYAev8<y8(VoZl*T4K%d=@(koUqK?cYmdsV}x(C37@+|SknEgo-iH>2EG#vypqg`h1TQfW)!uw;G;?=2U$;WuYxZD&x6Mg4^Pb25A_(l*|Y#DY}^-Lw884MThAMlf_AiQRV}0M^_KG2`?QJSGtlTQxj1@OGS33Ft2+Wk*4;j3zyL$g(ejqfOKv~gyWMQB0Z#-*N@qj5jO#m4w|juZ#AjvU7WBh*&{eCPV&|QTuW1|o!-KOCwWpJyq)<K!J0q)EpxvE%5E_TdgfU00Eo%4l#>t{FHZ#5ro2fvWO1#F9Bk6H!n$vP5qq2;a56N3Mkn>nLi|so3m9ZCcC_=R1mUEvU6uU55G5nTERyt+7L1@gFg2ww{^J^G_!O+IL>cj;oi!rWy!p6{iVac<Vmfto;dRO;yPq?a5tZm>8o`c~#Z66ZjY*mXSFX8aX_o2e?J#oRdFD`;6rd|ww#kOgme0>XlYxx8Lo`9Fxx2y6DT?^PW!VZ^zgyTLvUtz#$CGw6=lze#+NF=9122&=>rm5n=vgcY*srdQ9H;0c8l30zdh8A7+b!qi#cVFQ9nl5z#CV|?hXE-ORvu(4>2KhHEK0)9s18$i_F`rXYK2@6XR`PHsE%L-~RH38E8FO8qSd|JSl$;8K+QK7&AfyC3pn1a|1T&p)gN+VYMbT~9+502z*JRFec5r;#h7p?=0Zgi<aK>J#N><%&w}`d>`oCVhm~Q6{JXgh(+YQJC3awY~^Vf_5)}mUz9&fpK56+z0`UE696C;Y_*4l4VD;yU~`-Y+Mr9VVBj$GK3V{xRUbVXKg8-YcPfKMF4=Y!2aB1K*et=#p2Slxku?%Ge-&7QWk{&&rNmpX_+aTJOAe68nwTb?n>CFt<hgl7TMEUv*ePq$8@V7qdOs@9GXh~1W03BP~x|NLx!{oBvK{qoySy*Pd}-sZ2r{c|Gfzx>QIzW(wNC(~tm4E}J%jf*lczY3g+>3jU4S-daq?$M`<E=^Cz$v43A^K{LpNr3GF$%zjLo8B>AMdjy8hYa4+%>9FGa<@2;;vUYGHo1M+63p)lz<{(z!{wP8WP`>o@736oQf*mFCdKKPC!=r;2BiH$8g$4l7bVc{t%o5j@)@#3ElqBKM(%mr3X~4=<zcVjj=(8nztFi-851)LJh)`#-ic8Sv0(vi9KLJYb*yK##(Ic+oXR@SzFGcVRn-(~)z@W(LSIXSMtd{4dxhRMwnokuH)^fTT*;L!7a~??)oGOhUfE+`-fLP!jSm@%8p3o${L6xa_3Ye|FrLd;CU8$h3A2l;t4HHkcUQo(p3(E^p_|32X%2CWbBJ@Ln<hJlr=zGHfI6n>=@8A#{@&Y%aT}O>IPYb6VwIaM3vI9PPDm`gMM845dw^9kY%8FVccSN27u%R+*^+am&L%-mO!%<xIs^L$6vIHU0w9(7?9`OoCqOnDHV|z3HYTMQiFc#4-~4i#ZD&{K%a*}TO+&i&+8>BPPnAckrQ$6X#kIByXR-j`_J>b+D&+)qCJxK;e;SFZCF2PduM0;p)Y_E23RSZ+R1t**{GQlF&1JhJ3x?`fX2aIBiz&GLsP5>@-qV>D<?;;A!MPd+*2}F!_}fyQ7<n*9C~(?5bV`B~lqF`YYwE&|#*TT;S)zF{gO1T^lJWX7_2?s00#c#(=x`T#G=9!0BS{vE^OBO9j=<artRTNG4q_wv>MFX_4P?ZTya#SwEh-C3)rs}}k(FAX=W{~*ZWE?VpQmQ2hd3L1)=g(eE=b<*6d*+Y0^Q2m>=yPVd_JLC135N;5NzBvQ-toTIgb=A-I*yvu|CVrc^!Inp5U#Zqz-^kMFb9+UL@oNB5A|{7{!A!j?i`T<-q<@hj?L-%(2`w8DaPw^*Oh8`1`?UXcG5&8Sjq~WhD6h8G5e%4NW}$UG;G+YID!iN)8YB^l_O&CV6(OQk+e&CY8jbx$Q7_W&kSMqoJ}rI`euXABQ{S{D6Lrx#CUzF86|i_2<?;UrUvup3n#@w$TsK+Xerr<NDUFqV4t$JZShXJZMuew(vSzGJsJ032&e(UU7e$Wo6h%%tE0M?j1py7`*#Oqo+$24ef(1%9hYFOuEwEyv=??W9ejI&tSV6M3)F?ZxNd!hC34A31O!QlRDze9^#s>@|H>@-{yU|D=Z+u!9n2I%R5}K=Sm!}l0Do=1sjoAp4gIp1eV#B2+V>J2B<K9gr*Xy;Re}LWl9*}5N{>ZA+$c6<hD5XK&7PjY!wvhz^hhKt8zX^?Ii39e<r6?8wkLSA-P+&H(3mN?Mi7Ud>&hVg7ReZYtMh^8-txqdqyjlCMgMBH@RLShU=b_uSJk3BK&-W77nO1-ben1fD(&y9!lSFrG38C=@tXYVDuN;s*;s+yDKMydq-WK;0BY5c*j|R4VEH`fp|W;vnmNZpc=$A1Ze)GWs2MgR>aX2R=Fg=elRAqaw6|8FK{z66Rb9?@&qVBVBLJ>CTn-l+EP6%_YNQ<+UP}!=fOtcV*h(wW6~_p8Qg6x=_TdYQYsle(wd<SG^-0SRg~7E{IJ`RA#3HPN=B+iJ^z1(R2e_m*JR`o`{<N{k{y5dq&eL8!a$$>8dTitd0A3Viz{X>?O`$G;#)?)4l~5n$&Fxc3<sSq1z<7v`ad3FK^d3sxZbW(ObM*CI-!&bmC+Wthh$*%2kHQ^+$(*n=u#DzPQ%yjvHlGxb{Th@K8BXDNn`9jom$Q)MfDk69nX|3SPflU(J@Y?K@>^WSa6Xwt10LbgKUu@C&jK~8B9r5v8IfO)NAE}7oTab3M=*=y`FN~Tv6H%&`P@N{u~XQ{lF#RQZJ{1(OjaYX0iwf#UjWq<K41C69KQ_Xs~)5G1x(*X#$poX*LH2w;%TR3-NfxHe&O7f8;k`S9^6)6ZZnI?9FQh@S_8JVy)9;sGWj|trbmpwn)V}tv}er-WN^R;WN+#(MwX8&OO=^^bP{^Fp70>V$Nft3I*SZR$*4Jwqc-MDpEcU-{oM*hane#6g20(t0yaG2WC47%&IL~b){XX4J?Y%y|i0pUZ(zth_u?<+et|Lpa;@oD`da8O-Be0R)|dL>6DW?f2OnLN;HmigK^C5Dy0e2Pugb}sRO^0DiOAQ0xeIP66=<v{`*d3liYiQ$&lqnmfFd^iGwF#DfijsB8}o8emterqTE$MqsOGLW8Aahn)3Am)a^HaAU+DET=ZOfi9gGGfS}T4v<A*jiT=be^yW5P(h^IV-QwvqG|bnPwf=?S8Gm#`WfPFHA)X1lT1C3S=Sf>sDqgdH;wXzzAVQvadwrF)1Mho$?D=dU#aW%8rLRa=DVu9M5jfqt2|>c!z;W`Q<6^86n!6}_p8*ArAeE&|Gq>Q4LbZHxhZaLZ*5S;~ur7;hsa<EvGIVf{zaBGen<PM)des?TCNxr5M%%i~swJF=i<HTsi#nc{!!o&cOv(%W8HvFGHAU%pRdq5K^E+&rSeOH@HqKS@RZgXnbxXwG!z2WXr*0FTMecR}DvP?-8Q}Xk3N(Dj;wPAe%P6e*x$ycZ_-r#j7TkvO*eG6@_8qg;a=VwNa3fHQZzebbVABLRWhDd$v0IO`2P?sspq(e!7mzpj*~LWN^e*a!>9#2Adc20K8`=Doiag#L3ei#5kf`@UutJKD<c_*ZZSL$Q+%JJEvPVVe2+;1Rb?V$vf+|ToyP5d~_4+)BavH>X?4>!u5bW|h6PZ4fv&!zBc2UyRS1wfeNPD}P!>8o6QD20aD>WhbCcaZGO%HUn5Dccsj^)obh9pko!xVb9^kQXeUE=PGa1iKtV>dI(TM6MMhSsd|$v}xpt;<j(JLQ3@3xu6{{mu7flTRMzG1OIdG<cmgl6JTdDEyf$K1Y*m1iu7rmwb8g&VaUkaHS+`Hw0;nZAyp`!Nm8L`4i833f9Nr(>6Oxr%R5&Y!Lw?3B7-FT`B%bqEz?P1Vuu?9=!(G9gBS$i9pvky?-sixhFFJ$Ak!;`?fkut_>2cI@#n+&gMMIm~WfErfi6Z995KS!X*bsgLjoaR<&SoFIwG$Opu$0Y`jqRTi@9P&qpuUk5H?8QOQS*w6h~aVinb?k>kUHEgr3(HWGCa>*psg>(l5;`sB^#neh{!#-||)pQHPfdLsq&S>NKs&W~8zy3(oB6%<~i8a#ONLTpG@_L`?~Z6!<>=M!A;RlV4Tjrz12?hd6&Q%~K#Er#?nO6<Xx6Y7#@tx?^K?i*@fZPR2st0VWnX}jb~?wYi9$V-Ipd###hEV;W;+Q>ObEQ;%MhQvm9_T4T^oUDo|m|E>p*S?Sg?OODkCGP%m<MF*-QNtA+Yq<?R;T3^N%t@akSYCyJrZifrr!i(Vp8X$vH50Xqv;0*A_bpjXV5x-fwDBx#Ar9K_;B-BjJNy10{>z4VG3W|$fn+fNww7DT7@N~&(g|?!;sf`OOdPP(_0a3v9Uj7VT+bygA#W0l`fA^R%oZT&!!FY3v6gJPZNAU62ydAywr|94vN{;Q?p!*Ly)5JUJ4ABo=cDwHZ=<YsY!HFs{0@6F3^#w)*aJ7?!Odc77)krR8FHkecH?+eJ76RbSz7z8SHV2^`utAns^XEc)E`33B7P4JhW?vVhtTc<4pQ90URHl&v*(-53THvep#%@JaCx$DrJT(+F86!9Bh31zEq>HP-Fyu7lDKNlFKn@tE?LecPlPkhzj0V^8Iq;S1q$%AH!yvQ^KI$BF}W&DJptBxS^JE3>;;*ub0~MuZQ<@1B?V=8mtmI%{;RrT+YQr2mG)V1_U0RJev#o=EB`|lioVP0NcPb@4ws*%ilip4q|EckAt9Gw3z0%F%P3XuaUxHiARHrQ6$qU>t8tigeWaFg>ZO?rpDVN0YWm?6^<#dY=y7=q8EhGgn|tDbgI}IAqlR4GftFFzqGen<+`qv4je2r(>v0ocO!aE@Jz+i!POE96UcZj7_hZW!%I>XZ$fqkrWn5k8Qu)jg%K(vDWS$wc!KTOk36xZ(aJFRtPe-AR5|dDpzV=Jy5C@)}rb8D1(=t~~j@GbDqr1*?6+`4whnR_l^T@tEmeFL}rqb62OIUzb8ZZx7K69sWA9FXOkQ?i}?a2ETk{;bCe!*VTC7KI$2qNRJlcmcB%*WIG09z2dJgV!H_JQsEoEM!*W1tI@%0Epvt7pUKWlCfXyhoG?`kC2*la(N5Vz+IGSixBp5mW0w5Bz&BQn^SYI(32Nb_;jZh>ef!V<729=TO+R6xE6P#mSbYR<KBM9&K&|!@D4T^A%7V#^?;8yr*<;cl<ZW6oa(QqbFMXxT9-^`m?8$m`yk4jtXk{ewL{ocJO$K;tSJNI`=Z$kbdQvKy3T*zrUqeaLFy5Dsr(@wtWaEZ;S}3X{F;W%Shbqqj;7i1`kmC+I6%lYV6Y5S3G-})x}-`FeXkD{L0`?jln-<1jQFbVFPmEAFt5aAK&r$p9zjViac^eM%x!K9}KCsr4gXdb{sZ(0Yzp8pMN<UCA81|{y8YCWV78r@&y6x;<nb4WQr!XJ)Op9O7GYE><1J!Kv-Ey8f>HY^R@EoM1Q+gJFKamO?)9zsN^HGzM(<+^Vdh0ryg^8SW)8W@}K+o{9<sj3O6zWPa2*2Q4?E_UY0KB$wz6T_6}%&c>FQ^C>s<(D~9&}YrK!X`XIVf@1xHZ@{tsr>XvTTOM<jE3==>8*NQyHH@0i@;tx^69luR69&p7v-0~&V+D0>i(qjcbe&GIKLT&*pTN<|ci}GTvICMudlsZefvoR_V1PY0r?`L}peN<CQy%?Z#-;{GiwtiQUufl4tq4RNk1i!uJp&J~=_~d%}=HGk!`{(U{eAG~dnU+A~L4zTqo=h?ivn-+Nq=|nta-r7_h&C*mL-1v)PoDKa$@5_r?H;N~3IL@M|J3~ANV7T2Tek@(Qhi@)%B~r*^^-5)CX#k^)Hga-N@W|~!exC9CRSTD<~}^s#J`8_zdr8wU;h4=zy0#(9r)iLAOCousY4lD6KG}}N7Ge-;D3lLzCT+Sn%JVL2MNFW)8Z>SEf@fVxuNm~Zw?5(jlgT2LJQ1c<Ee#c;Ag)!J!c;N$3Oq_*I)nN?+f;~e-aI8WVta^*4_1|asRyMcSPHc2Osk>{cd_LRO}9A!IV$3UfBRK^KQ?*Q>{s5;+<fdqLqbN5aO^>ZT9PC;1=N=mXs0c(`vUpE7<|rE{Im1BDf)pWnHb<%32%jbZSf)3~^frI=^1)h7RsYkoGeNXG2>qDLkT<vCpg@z%gLwhp?tK`IDIZ2}&SeCX>RlO);Hn9Ynuw!x)i47?y;5*5u?`wQ7%Vmxhu6W1K|j*3NNV6<xkNCi?E+bHBX7BlC^(yS|}0eRI2+?=tTdc$rW=EUKDtrgTasqIqt_ddpR^>Bw$%e?oKtuY*g0bUDZ^ttkBN(Jd*$T{UZZL3Q_oFP{2}eUPtG)BqZuwD-~@u>8-T_pkG3oy7uEn&@2PV&)!-7K2|WJs*a(*qKhhq7Q&jAA97F5vIt=s<aA0&uKOd7QcvMSF04s>m7GAef2GDVXP6SQ_eP%;gey>y7?h9UtWUP-aK!kP@W?d!LHo>UQjrKCbpuhc)R)3`)JKeI;(Y|$m!mOcOfJyT>VN3wY8zbikQP@8}L)gFSBX`tT_41+a}-7tJ;*VDvI06t7)BWKdvY$cjDtZ;&X%GO~oO!WG-bk`cL(3$$OTG0yNHRUC*5gA(&O14P;c<I~bnr+b!MV19b0?qDWUsw(;|OD|6V?_a<tFf-fH45yj@xtFL5;ht2ZQAYrNIF^94t<8H{C>jdePO)3152v4L-E0<Jek_07W%zTYMBH;v`7h^>~I&YbWS0NACnm-pdnO+X@x!Uc^5H6|M4ooJUU&M5aD1s9U?5d36t`1_>P-cP=y7z&DVRPF+U928&?bl@C=Prc>7_lsHkVc<SXyIGuu`$woH-58@H+8(zm_$W3^napI$<4!$H+4HC>O8cHm^Oh(Ra(XwWu%eEn{D;Y>^54gCz!cfV%Mzf@#g2XBkX`hMZRM<+k||X%opl1QenhB%{@I@B<W6EM8{~*DBo?zBXC73)vX^F0YN8Z1wPs<gygWfzN<F3hKS2RM$sNxm%%}Bv~{z+dhYo$5;WK_7wT?rer}6H*s&mN!=cwGX}QYG%#_#535C)7I4C9+{H_Lpd5+X^(h<GqE|K4-vj?MH^qr<-dln?fi5`ub?0}mF)j)OSBxa9BDBy^Fsy{9;vkDf0pE-^KrEw_suoCFNQELzmb6s}0Y3+WtcA>^tQ--wsu9Ar;MtYnDYc9X~Nn!_QVv_c3Y6t32bR4z>`I#};2rC&N565E#>^DWW0@pLBUK2NlTk(80Vq`quIY*?kZLHg-S}C;`b;mdkeR)8h>S0G{-su0_D&v@cU-CMe5cGbXy5_y?4C>@^09X}ZN}hg=-@uBT``7>p(@-;;748`s+V~M4Zwlj{gsofb2M(Agy4FI_3$$X*lS#alWFnJCGq$UXNgC-XUmKq)T)nl&y^mhPgnQ&E&gw$AO`dLkum?sMG=U{~M3!%y2H(qw_&-whisb)~lTEAYVRDo42{*;m$+x0d0<XF*47JFLnmeh<BnQLd(e|#-Z-SCn*3zSoCLHA^z<Q1<kfJFqd7sJ@vk-NmIDJ}s`O3U3gqgbbVvI*)ZCx#3H=vFbcUWDTbvavVr`OH(*K;J!w@illU9l`m@s9K0GL=(5(9i-cu9r5Awvh_s22!u`Ow64?duN{<m5#&O^p(d7KUy{?s)M-XYV;EgNH`ziXld!dSZ%6araS<1c|O{@WXi9qr!xNkxJ8z;^U=uFewEsodQg0u1jkk~qEe_oxEddS{rUHr+rhjygtJf!BQgF8CVV`aBtY1Cw-bS&fy`+{hGy;XHIv2r_Su7SvwMKP5jB`)H(_fb^{LL}vCe!+<^#^lSd5ZmsEv0gWs3)d9O-;=y`DB}a#Wahk(EV&=DuNi;Ld`7iT(jA8R#L9dK@0z&6Dn8=6BN*CWZPM0~uWa`E_FN`V_uaLmQ=Se~3}(I+*kpe)P#GbVlD)hEa&HJW27fzAAcrHh&@X#n4)$4cZZ3KCW2_N~C`Q0<YK!BRFzN>R43N3Kp4@C-*!k)CMA)lcV{Y8FZuVOL7ulf8KVaz2T!x6H^wB*4qR{_%v^T(6;p?7no$7EH)nzKSbN?G%2_SKnI~0s+RRK_|yRESkUg&_64dOm|W&m^$x8}4&d4jg+iRd5*|z~g46EBXf-cWUW7q^^||%_^XJchBsnd~Q@?#}+|DWJuwr|8%%BwObg`=ZGXlb%hKnjE628@o%X@eS?$wxL5gy^P<TV$?^{NC<J*sJK@;b37RCQ~0?Y>wQSa8HkW_Uzk%r$>=c_BW3ljU$|`#-X>(s9{sbxKa?o_O(a8MW&IMBagmvAfnpc~Mhc(7s55`F0eOsaaiR;n8+e$yM1hq-nmt%RlJRfzfuIqy@%lacCjizIw1#Mj5CNy*o;1735S>8C2JTn{CiG;p)W|J4_LgT5>c-T#Kt~C-+pg=yTthr{-u;*&ep1I<8v2zmn8~b?vlo5sU+ZIPlbTSY*vbR@Tj3St9wS?pNyRqG?C=49PMZwDK&eBnsxpe{>|**`|D`BBvzSPQXHD+>z#%a@SJj4no#aZ`Dn+-AuW5!8mlR>_w~^E?A%@H;I&rxhBYDkxh23G)zIMmKhZfQQlnSHxRUYc9$8rE&#}ZUS&R;u7#>{ORCuCRf_`wY|%ERCFDDMA{G~rR45U_$~+oY-n!ysH%b2#*>(#_d#mlnqwUnh@ekZ3VdJ=b7hh^qq-0e}!vx1N)Sk+^1(Y^>kwz@djzN{lEb9hHZUGG(1O*rl$}4g*C3qWgMxK_&IvP8s*um|4<m-$RY*8?>pwIQ>f#H=!r&{=n8lAj;<z&<Nj^@mLj^y$|fg}@>2ngC&#A>9{4~l=p4Fk<szImOQfqmS-)u<NYEG%6OO7J@3S9;|U<7-4zgh$BoEQ+EXIhwrp2(X*sr57|U>H23grrRwq%SfPP5m7rMC^#m0x($7L4H{pNS6Z$-DtS~+x3Lqj%&LRfnv9pg<f<~M*pcdPMcqX*baQ@v${XZQ|4lb}or^%HTe-YS<I2Zqm4=5pnCa!}G097(v+@V`jCg3et+r+jHKYxVofto|gqUvIr9~JUCy=`tv|{gDSs<xv$A4{D9uCrQySlj0!jD@d>Xow~bE^DXSQYC|n@bG@wfvAr>&7>Rz|(sp-)WeVb{8d4xj0b}S<pm|#wu%iM!0qaPGNWq_DAV?M}vFcdl(uGTCSG6p%z;Hd(mj>+F=LvHhP3E9W)J_U3|&)O=J!j)RORM(Vj;Gzy%5wG|W0X`CO`uS2DFInHeGr*Pgr7Q6h}?X6WL&*sUQC!qjpdN1Mh%6NxPA8S42p_%@A`PoClgVW1iD&>SYz#^k4ZEp$ruIdd50&_7&(dUq5y7TZ@dpIi2wLu^y|y7|frhPtqX;lCYxgl}eP55pK?Xp54iEj{O_tV@hKmXR1Z@2X&oVo$loE>BGsp@hRa-Q3kj@;~q%W_Aj>i)$L(g@|^Dhmy9!61ttM+VDp^$9F7I=#fQs8j&r8@2_dHwIaUHD^(e({Sz-x-|>>3%9b&vF><BD1@v;F+BIgGiK_N!e_&u)eagQjhOgq|Y3&K-k+275Zc##+hd;63%=2fw91UyJsm=8=!NJW3)0*FT28%huRqfu+=I$;=0kXoZ1R<v^W$Ia+&+^%(@WFgMV8Ox^OWjG^i{Z7gL#I6<UbZ4e>j5`KF6`(=?U{LezP?NnCOixP15KLpeAHJiK?znYv+8NrMff0`XB+z<nQNMwL6rd%Lu0BeZGyS%<Xd+yOo@Dgjx2r!M~l>?N*?elB7WD-c@12P?ULs$Z)?0jx3?}(5vs7L|7g$t4GUB1@fH6k@=WR_b5E6njkj#k6kBkIQE{=Ii88ngmIh)Fzcs2gxcjP%_5x<E-2|`Y3Eprem+8Vv&X`H`fCdIKTN$-@&T(FCVQoQ{lG#k9*j#&BYLpPY*=EvlZCTuxH89_3mZ;iE45y-7X|>g33G~VamsxR^aA3DiJ`=w#LF7)$o8n56Q`zn0zKJ9xAO)7^2S%jR!VBGKY+(K_Y@jwfK`=&1Wc~&B9TK+<%g&L)1xwt$9(PeV2$Oi_ikg_3D>^Ncon2-HJKXt!4fZ_Oo}>lUZ;qb$kr@PMTALqDT%X0sqw7IT%ay>*Pjv{=bkxBy+pY5kUlL*5GZE#h_qi$)R!2m+McffP%mYgUJV|&OQ0x?)&6s)SxN<&;R$H7?W4i=(g9NxiVT);zm#o4oBpVUiy3`7$lgk@FlGH&JPsDmlSLjS`4qdg~Kz^b*N|==}3J`Tpgsd7P&RxgU8fTupln_rZ;atqSPQ6S1X-uD2hg$bg&WiNE4%eUdY_=2V+MC*rJisCXn>Z~KS*^R83?475&B*RaOAI54k+2r=V(d1j&2~+^eb>$ikg`LVQzLP04bMO89#wIDF2II+7F$0i?y1K{-8;}Bi`vYnBvRhXw&QlYRxX_q_-J6RFr;?DY=bNQXH;0wVm7+QWx0))tDmbZ5xsNhp%+52+vL_*VPBM&s${AyVQ}U=eYZIe&Kw5G<-1^+%`U5C+YayujWRooTD8v(AzS2L`a<i73%DLbu((5j+8K6=21}e2U|Cijjb9Z4!b-g)5*MT#hfeWtTgF?xO-+WmhPFrKNR1D~y<-sl!mYiRe`0`lL-)YR&^X0M__d~o*U>(ehZMK9B`$gT;W;jH%?7I|rgH3v3U`FBedX-K)K*R^9t^THvh1qnnvg)HF+0$SOhu^(9szGA_F4_r3!za}ZXa~0!AgBjRDW8}KHX|_@YXU2`trs+5$N6UsAeFEkH!Iyi4R<b2#oUDQM1cmgYT(r44Q4+b}$`53L!){V1%RhdY?@Dt_y&NJQQ+Xv!jx(>>xx$iIw@L7^MNm<SuK<^pC1*btCul%`T&uM|pY8H=m9Y?Si>-DPeQL4IzZ7FlAq7%goL<EERn5B^;%71as|Xb-LKziUQ;LHg^oP`P3jfLF$5bv`m(C$4GELM%L!d{Vkni|IL;A3Cj3emQG|3Q*i?!H6JM7%#NP@?8cgT2~cdE0%`oGpj^v*+d68S{0W(S7h*Es_8yT7<I%UWmqsC!H?RnWKd-RJ6!udZu;gep!t+rRyxYYw3@#y$731JLiXuC@g+091_FBNfF#$hVA?u@MQ@2rq_Xy&bju$?FX3suS<Kr)cZ2u#bdSZT{Ux)H%PYnN-7HW&afajk@AV6Ow?lVc7y}%5BK^OJnW{rl-!`dm>0_&#T@j{WNUn=&?bUKH8O8_IVoNo;}Jdsvj*Gz|xe=0qO*?hD1n1Lsv+^KYJIZ<f4xGbL;Cip_92P>Ba4%pz%8HYZj5Mj;pJWlw0QLA6e73OhFJw`(^sf-#^-XZBAvhtfD{6TYRf;_Ka4fa`!&e~;0p?HL#)LJ&=I7delORxJWvlqHM#X#eV8WMON&akmEj{(icXB*Xu0&9s*kz73Nt6lOLfhG&hYhor5*T!>Pu=#GX<~N(y2#6Ql4w%vG0%i#q`dwpEH@r*Jfq1(|Eyo|c=8p=c^K|RnvaMezStek=T4WCbBAg(})M4<MH4{lPQ`%)RUG(`p0+ic#0VmTntIves2Z3uLw8BnsMJgGMUaPtV;N}6V9q0^5Z*yo0b`qJVV%rpdgb6HV+QJadFgOHjfBAMJr?xf8fe$k!)0)c{CUnI#vnGsdBse>5J{Y4UNPysBWvaB?ZJzeLT_2*L_&KrHXRp%f#WaI?zU6(?ARs~Xgj5}C<oV{|y<5+&SkFTyJnGW>Cw?Z*Dxc|AGKz?8I+f27O4;5_?xvtowyn;fnf)Z2Y@X-5&*dvZ8W(=$*+et_Nk_iC*A|nyW;3%<wO5SP^St;5#VL6YXP-s<K60~xwy*9E;9MO_mQ5`Cj0|F)H+Khc|6f{Z0`!NY`I4?Y&-;5X3sxYCRz2sKY<Th%J1Ocj5rpP3x(q@j&^?GMZy;6_MZ6jLs^Az_TxZJe$LPi>6gu|qtR^C>nL%)3e5~q$6IRYc9fg157FrV5jB$z6tsf1xQvc=*Jj>jfxntNZ>jbLZYr`(3TqypMmm_tScs<&>yagiqt~#i5u?=J^+ZZ1wVniIvcD8W95DSk0o;1dzu;n8SE+}IEpz)q*vol7JUG{vNTeOJ%5tH?X&022xX0^ySZB}QZC2sPdv*3X#L^c5{3)DEx9eo;P7tKDVwlK@LWH`JSbSUtO<Uu8SQCNm>cI#)EZ^kR8-CVr*!$otrC@NTrzlIPi$ajZJG6g_Wm&8KBl~x#*AQyPAaDHA^pNnnte;APBam{7&bJ-l8+zTE+UV+S9{Sxs!yJL4-*#sV^8R8n=_#%!E_UU>{EPiTNz||ebvI318QimW*dE-?AF4_<z1<f1n8Hc6`CYwwdGB^V%Wu!nEsMaRL(1N6eV(~?*bJjeL=M~Vl9ZH{>x9tz=N(-8pkc|?igY9^Jo#?wtgtz5rv>OV63eBn0#qzD3gAnGi;pU=qwYj5Vs?}w&o7l(+dsO~H%-{s{a&&efHN7zl{`eg`pB|OXkmK9}3oT<(S^KB({FK;8?JU%Zjf~%gjcf|86~3}d1{{e$p<9X60wk^R@GEjsj6oUgO0Lutg1K#AI)E<lQ_v!t?DjGT0NE?XU!kRXj%Ln&Gn;`_7Uy<-NzmfY5daM?VspeOf?ZCSUipHK9Z_g1@1Zt#!MVz4ed25pL@G>I8~3eHc@e3Ug=^6*o$}jgl>m3q>(Q|&dIBsvd`A*KDvY)Ony#={=}CxoY)C?3kx_l_w2iHtiaf6+6`=Fd5VVLep9|1_3D1(Fh9}g0+E5!3=t5?n9&#~2w2$Hlb;&ZpE^&%*m^EABUSKCv<&4xGvp9SrdPTLRjMA=blgz1;G{O%NAtmM~AOr7aotc}YB<#NEB}zT<+5-%+?_H6|(e%YOFSdlD(N&?I9WiJIhI{QllRyf<Ti3$cGrFuQb0p?D>t`S$Nla)pZAo?7-5JSTsJ>|5?b0A3J@H{4tVZ#0cPp2hEON$QI6eGn!rsuN#Bv)t>UW!kMQyzE>XlV$`y@asL5qG?H&{yfYLQLYb%r<0eD=|{^tBJ5z8`ed^NLIi+u4KpeV~yxX1M?#m7BVfIfH~dqQqp9?}Cd~qsNQ=ptTe|%(yFjxq#zob?J&(OkeO}b<4Is$|0)3!gVRWiLM5#N@d8Dv9d|oNMws58%-Z>nTH$4ogC3oOq3xxmNE?PM&jPgoMX=3pIHZnGT5A8X1y<McM{9D&yb-EoiyOgZi_+`CK<^spk6|+>C`|u1oscuOuY=nPO-%1OU$wf9FSqfbbEy;2pN;tBgm99Gg9YhydH8uF_qd_YdS~^<KCVnz+4O3=0ih_tymsTSPx>Emw45j43C(jxw{=nQRytx(V=9&l6#lRehG-`g{X!IWAjv$thIcpESlOa+hVCss{rH}Pd<Y~?0$@?eH?^tRG`LbnqTYhv1va#oqx+X`C3L8dM}B43K%8OVfeFqI<V$WH)Wg?S<<X4Io;Cd^mM+0MrB38mkDW*?Gt>pDqb0v?(C?~RQ2?}JY5z>E8}VdnV~umQti^<j!}T=82$Ym{q6rWyQhr4eg@V$_oPnUXw1tgGf^$y3rw@Lz}7H?gD~Cf_xBSriEt~An0}96?cpYv2Ky)?8YWYag&`g8Zhtg0cQw-pDhC<o={-?6HcdHS91vZ>3(b{-Fme9T!r2#9Br25Pk)_!c18kbgdv1TuZJ}#Dde2Xb`yghfz>a=CU0)nDwx8zYo27cj$QiSaG9_fCSJ$ei@eG&4o>-}1>A4S8`a(>?Fw&|CEMby+ccY?rI!&J_MbgnpB-VXXW|qiiqRL>OR)$Sf$09Ztbt%&Ixj@^d#0m$763<~HMz$DTw0%i1<9eV~DAIw68hmK*Gv%8Sk8<f=$R6`aS0{o2P5VXmDYMuWq0SC8k*-L|xl}l>;nUgh)wRJS6_+uYhaDS4WyItxN1L}Ri#)#YG;mUXu)RF?PTnA(MEw8e-fqW{97!5|p}L6wpUd9uMDGAIQ-j$a=;~mf<a_tiT^XTNkmhFQ4Ass8bs?yf63K9q?&jv^T%wLW98;a8UhrA!1Vi9$OlhmQFo<0_G|&AF1(+Me8%M|ZE9gGPdQ&90CShEEXn7ADGB;@jGi7uSvv-F#xdQ)Gtexr|XFne_(VXYmmGrlkPk4K^>mA0U^U~I+qP0GYiOt4?TpJ*%#|5_krqc4huWWZ47<nRK-dZ?FwB*Jv;ZmBwAk`6vD)Zh(kKo~zuP1vxU@89MPH5@g(3gwUEue!bSxFUn+e_b2G*~5Mgj7cYnMfuxm%&!VMcN$smPuTMNr;OaV;)@@P{kvo%{w+pvdQY0J{Ns@haRZaIg=5B3>_q}_;U`PMrH%<q@ZF!mJBw-J&U@B85a56wi<_GAp=b8wB?Emp`*_M@@?#nk_Qa&(P}@5trWAk42A2yTV)^|;t#nv!1#9+^CZr{dH@tGZPSOD3-uz#W)jP!ZJm0B!afM>R%jf>wyiJ+bsebxmz~Xrjx*s3IV8u_@t8m~0G}Q~f*7sf812|-?TpL(jf?EF>c%j~Q)nR76*ZEW!2uwoyzR-x=q^ihp^nt16JsmtW>sPWg2@->>g&HQAjx?zDJ7*O5YYgtlv+uF#tQ^xDfD3J<_=+@knv@*Y0(@pP(W!_Kg3J46+ysP?iejg)CpS1$&fz_FE4rD0#_2iA}=qn2#SMLX#C;4ur^<RSNaP~Mlvtr5+Xj8&Jr{MA%2n4k$lh`41wUl;APdQ!ySo59PSckwlN`90-iII{1xtQYlyl{jy*IL+A>LNKoud2Z%=0S?Sq9DIRlSWDOzfgflaBLDBy*Nc0Zz5%e5Qdj=#qZ`nEN9-jL)u>>C()z2q}dDq-7v>ygK~7Bgej8oV~=819~FwTlV?@f_*`A({J(^h}6Xyz&5Q@7j(fQC;l8ADnKVJd^f5VO24J4>osXPr&rrwVdZ&D*M_@7FGkxP#UWp+%eiQ<>9tfZ9i<-GPk1(T_kEMW{YM~Zg=kWl4a<g?)#iMgP;u`Z&f@VwpAGzm<EUoNhB;q5+#nXewo#439LqO!`WL@^_7u!YP&!t%1;dt7{0HX6iQ%8vYM}Z5aLg}lKS<!N;T=u$U@!IDC+u^gQd1~JY1#UI^4l#mO`|sw$PWojQ|p(B~XCMXqwO5G@-#SjbMVLgo=_P(p0v(6zW5CgsgCfbm^eVpy9YCUvbO7iHWF{Zs$r%lfVp&%&3bbG(ss<KTSEPe=N~IzkgjHOQ9rm-UY|U6uhFMNtaQjn=vhgwh+4}>X@nylReEvE%>k$%0jg7fe5;4H<*bqNeY>EhE8J`PGgW!<pUxHSCeV$*>NrcZ{%|Rh;83}=RY1UA@t#=NWqds;IxRxs_FDOVg}dyi2;+Bk?Yz9fAlJx1V5kNGiXC62;Au>GPdfT(2WC1zp~ow{uCxR$`>#qrIE36IkvrzF&g(Sqsp4C#6_KLcnkDP;ubi+xemaac>TB8f|Wh2$wP+6l`r;<X(&Q^3*j|xw(dgX|733r1^8Iaep9|Po!AArDxvg;SNaT49Y!^HPVAV#F8c<p>ISr25ynIm59@QJL+9(##AU4bKQX!&fsppzDD;h5+&xA15qCt!5%JZ%)?inh@WCJ}PU8%XJOHTILBk#H7hZsnb9)S9E4S4a2&b@*a5X^95mSFzhPyJo2UxPcg^05Zv+GMpQ4q6bwem+VhvXP>Y39rv?MhXqUPsgX^=66%aRxTwr<ttom?=$<WN3jaTD}EtfX%I)O*r$uh#VbrAoITzdPbH^jsi+@YP)fJjit~TY7xgl@D&Y>g+iI5#lMWb-zS-C@w>$aj3{Bd1LNS@Z>}?!G-0XV8)6VP8Z?f!`YPR8Of?FO(tsd<`Am&w&(wEVl8(;ZH2K-i+_}vhIJuty2)<D)P5L!Hu;T}lg5lB(i6=#nsJ-$AK+LQX{h^mJfk89QZo0=?qB5&mWhI2f17pUR6|DZ6TN1-gQu?T|q|N41c|ytd_D$*^#Y?3WZeqb+2zOfLH?<(+rO09_RFPPs8SP8pqZb$I3!L+r6E|FUdqYUwBziW~{ttU%tn4vFYZAgp>b_Fm&7B%Ew#0Lz^)n+@=~{e3@cPjFaayP+A=I+&wa(C1{xHz4F;3IfR+FJ%N?kQ30GaZVaH>2a;H96>C093C5vYglGwP3lw@~JlmLXN#In9RIDzy9+KmYjcx6j}B{6`AIhCn<A{pPWly6BdjS+ewx1}HvCdRd|RI;wjxynb>DKG|2FJQrmhZB^=o4zk_r-0Hm=B5Hk|dg)%?`tnRFTM28}7Rg7pMRCvD{4(UO>w?aSv`>e-zfis2@sq`({CNAA^4Oxo3x%|gDgW5V=MRH(7R;5mYR*imI8LT>ma=#UZtGwF`TZZkjZnXMei(k0-6s!O44wbidLMIL?ZQDv<~%6_Ry>q(EJwP5;|N0H=K0TY_s=z1yFIb7T`>cHiTqx=7h$;1<v9vM*S>R89}MRQr`GeWrv*P>@X9~9;P|+`Zp!lUP3IH~k%rG<sNxEx0mKxxA++`RyJpGsiw#GUOf1m3Z{~AFw)VpQ=Tj^~%j5V6-n}Q97#zm<<a+t#KW6*K_x0VM-_=h-8~%Ou{y!GEpXQ`w;t-Vv1kR3K30_*tIlx`)3yV)t|DOdU_j9LdbuX5*M2dkDv?@|7teZ&3wik@6p1hnt!pL8r7tAkz|I6Qg`C+)<A1|K?1SyWkF%ND-*gwBDID%?_sCdUv&=}4ENhj_<{`Hr?{`x<E{KD`5s`6L$0G=mD{b$Ir9gxvX?!i?ZQ=uHnN078OYnl34s-zqvI3F9uEuL@Hi&VU0Z1rP+s~8~Dryy|Bi-#dx?BcQh7}YY_l}Sqek;s5CkAFW$vgikhCQMWvVdYe<*bTndU%%Bay=-6uvGqenw$Oe~@4RM@-`ibkh>2eDn-m8U&vX92|Lf!9A76{HcDbwH2h_3voNhzYuf59#FjR$eEpYnPZVSvDjyjDXg-`mmv5&86!c@lqt$l6!b!yXNeodyJt1q3yMqi@UTOx1g8$H(6*%iN3wQ-+&QY$7n!8AJY)&(+_NyZ(mBXF2rc_!#jB49v{e3a*wY$zC{P(HcZ&jpt@<M9Oe^84MBqJ2=G7Dh{XMrJSQ$XC4l7`ES6koUU@SxJSL67etkQVPtJ9;&w&RS@+_J5_cecDc8{VG-H04c0EyZvd+~w5+DHBP=#ezl}E4y_S}KK1wFiZ^uWO2RXGku}pr5?}CJiG1|P_>?&K8%Ak&o#k&VQ?!{j?CB+GT7s@|pI^Xo03OehhA|W|zo*Q(5>9?&Rzyy&H59(sj5?xx`-o7d<)g@(8@k1&!t1NYpI%#Dqw5Y~CeW5(nCOFI37~u~O8Em*-(CH_&A?|mh?x)0&N2Ni>^-7}bj+doj_rPUhlU=pX#W$JSKtc(#FOs?E-T0Y^S#VHVeTXZ4G>@=M3k@M2J(6b-y8mbFwW|`L`r=hQGPlf^zcN7U@tlLoQ!5fHjS578pQI!9`*tIZ4lou~C8A44<CT=;CaA=~1HX#sS$IR#1IY#GL)W?H95yfaI(<-&o%wS72@GA|sB%wO1vIFnT+hmwhi=Go2biKBL(P@foLSb<?mJ=ij;CsFOsDV|j_g^ET4dICD$itA)lr1?pHPOKF?_D<tft~&F0}Y34p+e1N`kRmtoo1>LQ+&DBD0vQO)h2JKc|&-oZy1^09fbEZdpWc(54f%@Jhhwc9UvQfu*&GBULed5OR>qw>%%eBxZpnb!9nNd2%ErIrP1+dqJWyDz}45SH7&TuA{6;PsdoL@PlegLe+JAbceA|j-ifjcAoKo>~<?q^Tnl;9wVN7axdYdpiMVpQwjSb%iT1e2BAnAUwjcV;&)&TL!r6^qIVm1SiFe)9Z1B`uSq)F6hQ4>lihq)P9^7`XTzo;6$UEEW-&Bhl5ha}-E9<%gccpm3$7vXIf_E==*r5QavkQ9E#nyINz!l=K}U``sr-?QW6dlbaAbX=<}p?!)<9h=Du7j|a9<X%xPI@g_vq3*P}7GxN+<DcptUdzQybp{EoxII*@qJAh*xiJ3}UEd(Bv+lMTO_X($J8b$tu!xRDSW+Jj2MdJrl1yhOyl>KPe!j8fM>#MTfl&Rb|B^py$Ac5n<PBJ9d8R8!V?MpzHS$4;OLleesBo_!y*Dk7f!Cqs_TX^m?_HF5DesBN>kdvbLZ`yi$z>JO<bk(lBrqnAOjD@Rc5H{F0z%kAv`0667wy4zZYT38Xu$xVDj(aWx73HC#khoKu2K{%Gx5E)YdOqh5!N;UZYmIkMdynb(BGXi!pblR^vUCsuvQ2zu(%3XIMRejfm15>YZOR@%<F40e$}>mib;cH=;S&wZ_~=My5DM!Q!-Iu0bAW)0`Ds9eNq?O83xSf9}g4*py>?fX0zSKe0i9ulEdM!Z-`We(GlX%VK}Zv6jvMCjfFm^WYP)%J#!uLR{9t*#`tF2PAiJMWBX!z`SSzkdIFNyA}9=R#7WhRNVxQX!G&kjlvzN6oHIPc~-eFv=*gUeW52CZ~H&YVPbwp+Ru<z8b266M$5>GJ(V9cWF;vWhUrDosBLhXz-5@L-7T#6uDNoA&4b7CM1(+pctScm*bIxzeIn)s=ho(_3~W#^UG5$V}18(Cahc!l~S11?B=FDX6SP+>*`~oOF~zr=|s{Sb<xi=)qI8F7Sy+5z4E+#>bwx<>7Rxh6*YyFc<PIi-k@RHODS9O2eAs@4;Y*7qzjES;S<({nvp*^Zzg?&mqb-%nO`5TCsN<>l4%<xK4z6gr1)p;?UACP`WB+2XAa?sZY)a5mUTShEn({aLcG5y?XmQY+TFE53vqp$bp7>+t^^7&_`y^>KrC2Vi$ngehL}JpNnpgwm@ib}q_NK}_n+Sn{L8PFN|I#?nG36ft@1@b8|J2PsNjH(=_7-v6&@D97LL}Z5WMFd*qo|M{M79^rnG2}A#NBHsDqS+h@ONc@bQ7W7#>cm%N^tmsDZCtB{JmvgrfA-=#KHL8<#0j)CVVRxpiy6&&{yg#Y28AgyT6w&S6U{UXmkuscsU>Aazd?wG_eRTwUk7`k!qZX_0Gxr)>Y$j*8*gl!>l`Jp+4>D+{JsM!VGooXzN4+Am>HVQw60v>&<LBPz%>HrM3$l2%=osX%@0&P7F9y5MbCXTL>~x>5sxtnfG)eZk<sTp!GH|1WmFi<9ud=xREH%d)UXzc%%$iVH^UA(KC)&*B*O<}=L+!$NLg7PjzAzO5B#?C;%|bYQ$RT68BKmo8Ss23BavaL-xNC#k?7y?>B}Iux=(P==)^S+I-2o^oVrbI0X}rmn-V4f)!k7QGNN+mLEiUQT87yF`K5tfG%U;x=EvczA#EV}z661Oi#nIYeeyUQ@S}_{Xr#4K@$P&2BWc=Zgmg7B|r$hAr;HlFsMGQ1Yk1X#G|(UeB1Oj~irxvmMN^ZT~)c)G>lRSywb08YYaj5`2(WKw=o~BrOz53E+&X6)Fv;3Xh)J8hM1r!B~_`rdu*!R%U&#k&OM=Vk9QkmgqtwFv(-ElNc<^)&rD=wzk#uv9UK2p|wK~JZqX(tk3*-K=-Ti0i;LUuc`Yd9&3{%rVkm=Cogs2#(v9Z_o!Y4_*k4E1mkI_Sp;#4=#m-=0GhtfI2iGYsoTq){h+6yC&XvZCQ+FWU~+%2nVbU;OD@}WoRb;s%anomw3a4xr)z^QN*5WjFH+GUQ-Nv=EqDaAGYe3MXqg*pTX_1K<yR6aAJC;C_CM#tT{|upyaE(!B+nvMK<U@;7BiwP=gYDXVEU`ey`YuHoY`G-0K&U2TSMtrqyG|e+@#vLUd7#3n}UT}Es8jx(kEQiqI%LZzYo^&xkg{2a(;J<6qS?25?PaBN@}Ul3}fYnci1D{6eu?mvuCI@!lmcQI4VL7uf0fAz}tKp^O_6pvvIN~#LjL@!iX*SF`BtSdpL;%h1d`nP9QmRt=juvM196lhE=eKJlC*AX};!0v9D{hv%X+fYsq8Qo2fhU;wIC(mGXFmC|#p}J~rWX2*ZH>GM1pcuFDwI5_~14+Rl)vvw@pbSV#L*RYe0|Pe5|7>129W{nTuaLZ3m%YpD;TsT8z17Aq+z@evV|U-|_?K&DPz6Ir?AyN;2|3Q@9bFT(;cUZR4_t}dv$*&efK(Hr$gW!3gZ(RNWL8rcRqIE>al0@Ol<G_g-QEsf&5EKXw>0We`!*>;RF0bf#d^}+iX-?vq=(l*8kK%YeXL<DIVSM3<&p0Zrkk&nCI8ANvO8toK1U=(BrW+VNMkW(@i^^~HXG>q(6nQ}WZ2(0sPr_P2#DXQ0HE|@Or9F?Uek{X6?40aX5z-68qjTZz?)cZ21RbX_g1R@Z&jr-jAic0{Nv^Un`y1Ar<3s=pGo5ohRo`Y7*l2?I9z7<g7DbZ<BnX#w{^yP(Z8Vbb%d`8MR?=~X^{3Nv6`h_`%dUQo~Uz&69@!N0j?7edgs&@n2N*{ppN({vwiA1caIxnEgG%kbh5ds=brKuCg;dAcI*5$dzXv=u3H6YVgfZ6F9eBk0l$0(<UQ3AOSz$+q}z_mvQMmEdK;mcnTqf=xj=|+U92QY4<b`PHVud_*bI4q51t@CPox4=7Kf=0N;+4R_`NRMEUEtar0UT>?Qqc(D)i31tCbojepB=K;_DLJcueWb9l674@au3bZ#v3ME{O}oHm*QBMjPgDy$toBy5>8JE^ZBg#iSk^&<Xs>Jg>tBA0&tk{G3Cqm&``6JXe4|bHIvv82Znt{EcvSi$0S3NK8AIbuW^6Wr(P|59iCMFr;9dnk0^Eb=i>Kh{<ypS~)cxcwdq1;vU;Ltt2hr%Y+#8dEc8pA_<57RTru?<KHZgn#2E8SXqs5YW7O1xF2oza&U+Ix|q+p=E7It>8{jIaL2D~D$dehY)I!SKZ<L)6b@wKvW3+ADkM%KH{uGU8gZ1#Z+{^8+@>Ew1PV1oc~gq@MaQ=rowTM!zD%7igTU0YP|>5Y>`V{B&lNm|NdQM|^GBPq6OO6PJUqmbs7j%3#i<U9>Wm&Vl@dr=NWh&JUOL0<&ob+4H9m{BHKDYMHsG-gae!++Q|TL%CvhHew7PTaVm5L?$hp&xB}Scz^pZ7aQZZhwPBxT;dDb>I!$!O&0J`zD=k!a63NaOm=VsPKDFT(Csa5Be~^#}KxL0RtZW`WEhM`2+!;fS1{~yX6`CWUF{5c*R%H0pqwY?kfy9twi22iP;wE;Y4y;$Y7Sqx?Mmf?yl{Nbi1d}`5=ka=xV6ZWxp=1w0746-&ZYos^vC-a(POqKN;g4L*GO-gB*fT#|N`|-8*p0SiN1I@XIlgU6ec+y@o~c8?DgM<cz87QuPmvl2c$%TX-Z8gp@!BG;jEgV5U>o+votpY?+i&Dm1+jabJ@;%Q?Z}Z5u{xUIZ{<BtpQy;wC-b>bfps?O*@vqQ%smH}KS{o|U{np)&2u&kO?AqP2W2+H&t+oLo0e0+OAH5yf$<^_!^_j*F%Jgr)IIe@NXpQehKxo3N77EwXys3M_8lFL}t}^TB2ykzU?Lt=#>ESlxvHFD}^4?v{Z4+n;??br6H%C=&JQt><-HUNOognDADGX93eJZaY8e)}*%OlB!xeMj&!q!V>=Y=6`;#-~Rpk-+uZ1yEcw*mA3iy-5(F4{N!hz@%rX7PNrshEPj7Sjf*lcpJ`O}_?{mmD^7OV%XE)9-Bf9MIZl28EI*C+beaU%E|8q~fUxNu!(CK<D(aAdKh4xX$R<aCZ?g5{F~ZsI{~Pyw!?Z_3<(Ud(gTXHEt+6MK-?EZSiqkO_HO>$Y4tjs6n2Uz)97`+RPcZJSg&{2R8Ke}C^|Ol?chAdKpmdO59(DzH1Wp<2g-(;*(grUsv0T<}oyf3&G4?-e+cns;vauE-AE&ZTlP{JJcN=_O`1-n<6$*W=A~ZUi(cLTbwy|sE{NhGkYcmz)%9aWd#N1ULhxTm{>Y!xp=l6;hQQ<?zqGk<FZ9UOra_c#{B_TYQkxby4iV|iAX9F;-)!h~FtY`Fmdgx|xYMMhF;vC{s)J>C}!_!ez4nQ5#^mK@x%>LfnhjAO2dN`kDcw)_+T^3qj;f;`3c#DMOYWDyu%CN0~M&5~@R$VM(mSsy$MRhg_dSb$deb*V-KcE-}1uFnjna)X#xqSl4M#BMuO<%{PX4gG-qtw5AInB0ns?%-BU?Q9bMh|SSenA9!iacUjDqdqzyw+CXOcnrKzxafwQc6%~;&5L6Pa{!f$#|lQ*M*}PYHgOiimGO3sUivsc%R5cP4jlEEErb5G8wjNT&#l2kLr%e>@Ab2Q7%vLJnkP``vyzbQ21L@ofvsAM<{T{+&7j4Cn!rySl6rzI|e)CIcJIC#SA(|t4YS|%h*yMnJOR^^&SoCBG183xyneA#p1lAWKBmP?gT8z?~8-jh<<eyHFX0SaZ26;wXPbKg{|s@eSc(2t<Up0QT%Qbrc9lu%G5)g&1g?P-6?s0P=FBW3p7{OX1B0k!sipJHIQQi2*Ji(Gfn8ehVx9((w&(S6l;_0oYuaz&J*|ws?-4xsz`xDW)_L^0+BT00F2^68As?k`sKjBsYARlNak2-nv5`fuKJX-9qvE)3{9e5FXR0oqKpLZpP}dK-_S(k->p85i`v}tv?zxMeEPUeAuD-yq*9zsuqG>s$#CmsZp;7_v`3@L_L$7|Mm`RA$oT>D9CFQ@+Ewla2kXbyKE0))qMpzQ3%1cWG1~=y)KPt_t!V4sfd}<J3lG{9j4iazmJA>if5RK7ig(oCCfOM_DrTWj2-l9FObkB#W6<L*iH6!?i?SuO4U_J)H*J$Y(71FmkY}(}4Wdf~jCU5BB8EE>;0a-;2$4FX%pT&J*YcL4NWRYda8+19fP;&`v6pwKU{6JHyh`?PAr&k{VrgPa`Vla*EfJUnD-2L!?u4ciso@6MQ*BBZ;1F*mQz5iGoaDAR_droe@5w4C)PZ-cqEhAj8PyMASNJnIrRqQcZVbuYvbD)#&}&zfc0%W|<|imkHoy1$hrTh`*|cYB<<cZ2QP)kbmx$rI=gQY2NE8u%K0*lx)Ee(2e?!0$i}N|mZO4_?`BJA_3?zfqUu>yL*3RvrlnfpXeR+Z#Ojg7@$_i|;6j2Ps^U<AEl)wY3LEJ-t=1*#-NJp?Dj;^rIB?0z>F`<<cd3Sk%o0*xw+N{nKpap?-^Ua;?-9dXxwXob9fQ)FP&$4(PYy@8H|A>1`nhiRGyR9a@r1V<mqKqEZnxPFesS7dHl-8pBu-lO#YjdYsMyf%*{y#&ij2HGb8EM2mC#9fdN8ddu4)=3mpwD^rD(>~PEUBi&9Wj^su;_E~Eu&wD8RFvPMld&qgHDwKV9c%hk4IQA4@+}gZ+9uC1XkLeP?ieKgI(kvl7Z0&)B$3tSNc-XbJ1Kn4PV#C`cFWy=V7<$W5|q68e{Y6RC7its?Xr+_({oz)ljt+9pkh#h$6`v3!Y`qY6`l<AX_BJNwMp=3}#Mtv8IfOl(%w2i_er-g&lj3S&zBeT+!MN&`O&7{u~3G{J<rlQZF9`qq#&w&14Y}ibarJ#=B*WCIVi;F<|8!G1x(*X#$poDK-Zdw;lHP8}WF}He&I5f8{s7uJ*1`6ZHbG?aj3V_%VS!Ve2#*YNudgX+;yBon^&2?LSz>-Zw?pp&Mv|=yS3#oolou=p6*+VHE4&#GHmi7Ye=;t;VcgZ^NK=X_n>V&@TrjABJ4qDX7kSx1MZ1CotJbVAgHX>MQL+ZNMnzX49@K^D_NMM5I;U-cCZ|8$FN~yF&Jh+jNBBz(Qn7Pp6#J`7@p)S4HEfZZM9y-AZY~^po=0v#bMukS-Cnd;)Dxnkv?vll9*ZQZ~u8H&_|6+{m(aa<Agx5lG5?PI;C>aS%TqQ`MqeRY9Z2WUfOzv;JP?>kX*eAAV1C6z2J+=GrCxoZkZk%}qjU;Ovy>PYgqE?!YB2ah@l)c{&XZ^L=Hlzc4i8k7=lE0;+6?pM>45B0b>qq@7hNUbBDVD2q`bLY{biea&kF-uL|4(`_KlS)HJztx4BBZ?5b_pmggd1PN~g$H_m|#jq2ayO{Sj0SX>LDoL5@*#aMh*7C&@T8t9124!}Jby-|XtvWL=eFOLS%b8)@Bmv5lR%d9L&`4n!Z0RzqmvAC3QX+>g>Udrb%jn86DJ}F@BnAi66s6}~)ycD%-XY7x!W?k5ajueI<un&%-4gNlFbRR;soR8Skz19&=0#oW4Dfv(1sc9%@e|C#Wh$)sxo~v~KF7?D1$W>)Hi`|?zGJqs-0m_IE(B`v%>+jPY?=V4EDFIv?AD^}!CLSoXy*y`1>_BWPBGCpy@`5Zsx8XC9@lVnBb&ddB9FI*LUhzMB>KG&tdQa(xudSAHg|Rto|nKCnNtxe0@OQd9UFI)ph^<WZYF*~wLTA`oCdL;dud8A1e^5EM5fQ=tg?HjHcD!J^NkK4X>B)i`BZsrG#4S}N<|2M65pvTO%Lj7As9@N9ZR2W2uYmAmnrmYX~oLcvc=sO;UK8vjn&L3ZzY767+SMRmw^(MYL%g=?34$pE)aI+^*3LaO}acxW2mX^XyBbyCGF4<DEyghK1Y*m1iu7hmwb8QXF%CLXer6w4M7@X8xuu{;K8@H`4jFv1?%JRX`39=X(~rxvWS3@g#6#!RA%3jDAhgnfTBXcmU<1aI~Myi5`nHSdVg)fxhFFJ#{&^O_ic4dt_~8dKH0=4XU{y!m~WeJQ#Qmyjw;GE;Ux#hfOnNXR=r^GEb6)knIJb0+1OC_uI+4s=cAYFN2pc4spPXq+BuOvVMS$X<oL2+i%0fTN1`rb{p#|vISp<}pS;;TF@B=c__9QyJGzf4A1R>E`VuF0e#F|=MV&gepztEq;K7R)VnwpDZJxrlMPa%)pP<1P{bDON>Weiz97>g@p1N~ejMC32u?IaTG$l`3qqrGeA8KE1(PTTTBlo{4yX2PKRcY&xmk9lPT{X{Gau1`mk#mq(6xY`biH+{;yWN&J*%ebTwQ5t>x{y7cTJ+8mPk-rne6KC4zk_2fUGEcK5tzi3^f`j%br`5xqormVV^+gC|G}?jqH=MTzKY<!CA$eMMd3RgJPTWhg7!N&UC-f8zW#^r*$^!T-61Y0Sqy-!rJFOt=G06Y0WRKr;PH`(1D3KMdU?CUL)eb%d5KHNn*^)A+BYDR1xV_!i!^#%OSW{IuQM&eTc(Qb3$dH54#uxLP3O6nWnBM&NKWZKN)7os%Bt4}5h%{zus1{B`A>~GxET#@7E{AW+Mi9IBNf#f$LrbwBY{ZL+8@0N=E2YBcT!UqkBp>#A7U1<KiC`f-<&#xb{9~P;u7|<`x|?DzSyi#7Ni_X;FyKx$-<TS9JX<}KjIx>_CKxhqZaDsbE%iaRdfDejioe|<=oOlIOF_|!+J}fY*j8$fS0|2=~J9<OaI2?s#N_1*z9HRGdi&gGFj$O9-i95!x2gf+VC#@E(QEob;Z^jrim);v*7H_H*|iJ;kZ`*n`{*Qklm5&qj~6WyQYeyCa$E6d*qOi=U@wwMlj1DtK8#6o;*P~M#?%6I(Js3FzM<@EyLJKGZ#8nX0O$>;}p$fdSB7w@)k1KGR$u7i31LPd(KQX<nj)*jEWX5!_uJs1^hRv$<3w5O@J{qtJT+p`7k)Grigm!9p9hFmT#2Z+RczZt`L=R>q3`8XO2h)h}0s}%%B4{HSSNKq!5L(B?EXm3T>2_gp$;?UkZgd;C7k{T>wnWTroLX!!i!8o##4+NK=QHiG}mXxjoHjv}IGNx4{w?V3Y>T1I%Y?6z*f{W)yN`bGH?Fze3Vu8nfT9*Hnq-g*pV0@$F=3k^%GaG(W%=#4eBO`cnJAc79Ha&ZIF=!=&Y(rkd5O;PWyjvId?LWrBWYcHm?sh$pezGDNK5tcr-K^q&X*y*8;_q!69Dz;e5VD{92T$M!LhRHJh!Y`PTHiTcIKmZny)NO2l$ZUe)+Abs-+sP#j1hEU#9dbT_MOftnFW%KBXRy%jpW~e@UsuHuB#?;V34d2f))x!=RFHwBsam|f;nQchF^Gr}|`|;nur&w^wEuCs|u@tg>2q&+E2&rhLqb<uw+_h6YNfLtxsD14^+7&fcX>Dtsy-ez2F8~-5rwQINxKl&$r;MO@K@>JnF8u8Zo&EV6pZ`p7?AhdzBQjdPfaze!YFios`fSI3qZd#lX7KsR$!MW{?)T3{StXn8@sV!`U^ln5T#_+*u=VLQyi$6<{IhScumQr#vZTQldOy9Ds}g<J6+5izo=tQiQmEuJwC>QN{CNAA^4KCS4?9X6Q~t4!&mRUSt572&@T4)RZzZwynC0B0Jo#)*RNn#hi|2>oSJ|KlvKTu5uk}9W>VxP`zmGOm$Y)Y;EJwQCZwXS%FpT{8Ukmaa-`K9qi@!t(b^JQTc)%5F-{o7V)q`dPrKbfyUvPaeQEmYwTPn8srSf7eICRG_RCSh8XJb?#2ow@I-_P;R)TySFda*#~zM0Pz+1gb_z8b4tL+9i82;RM?p&J~=_~d%|<v(Wo$M^O9pWoF_qT$K)mv3j2`)N)p=nj%updD;$E1d(}#lENw8<3Xk|Fe=){oHA8=!+#SlKiQd$6AVrH1(-&j<THfkGS#I=LPf2-~aNrUw#<w_s7d;@V#@~CjC6PodW;-*5C-L{h{KW8hQkCItC=2xc~UqU;g^*|NQX_zyB*xWQQk5{b$Ir9gxvX?!i?ZQ=w#2q-Sf^HukfWXgWr4J~oO|Jm0Dpsd&fDwfD1XpvPo#2n_Y&VI&v;r*$opT`#EgdWoPS&qf_1S@Z)$6DBgeu%0n&pZ@y2{`#$c>2?DL%<Wrr8reeI8@}_JJ$_F$D$J;|>zfn@63=t~zyIsw;~!s(v39wu-v`vP0Gt-o>et?-TNp)zxgLgnA+AW|%bCMbr!u6rRbQY~wz9ZtMOeoGW(g!1;?$;{|C-Q2S6`YDfWAa)JV)NnH+rnCv#TtuYC%8uq}E|^f@yT(tqUZ@6S4wYN8m8M5_!;{M8JR?`6$n=Re5|UUG^*ocrLiKnIyvkF2CRDFWLw7X>m`=t<<&w#RmQAH`wJ(*h1P@;iW_xV121bXzH5P+l$KU_!KBsP*b?vTi>vV?AZou=jR3cjNTw0Q<igo{QhXyLe%R1&qv8b5A66T^B|`dCzk1Z^Iec=+<`XlrXZa<=&emCw6S>ifXBV~3#W8|!S6yk7QpAolT|@yy;LM5uVbe&-369tYX~qwB*cTd7_>x}*0#5=3QKiS5H)^Cg%X~n*;6O2Y{e!r%40nJ8`Q2D|04Y1A%hLq3p%%iu)9&oRbs?rU;s>;#!K<C16<5wvNa?i!=8ne?9w@YeTHuMQ?ZE$lhwzCSeRE8{o3;nLaZZM@lnDIXRK5R!>Je+m<_cgbQ?sxixE8;`wH+%2iSO`G7PZt$KW=1a8w)Qc^F>Z;?FGw+gjmbi(bm7LHepMNW>OTvj}1%jxW2R>s;J9aE&NukNzi$_^LG@I(6M7#ItLfaqV4G>bq~dLuYHGvcR!Ms-%aSEAkr(U9%Vy4nuTG?(`K<+4fUwJ>BSZMX9EbG@Iub(X2okUY5X!WnZ=moTmmuRaH!W00#<WAj{%8+S*o+G5J!Yyq-4homJ+EuC8-t{kD8i?`v<<_{X%x-QM9G*C3RlN6>`69*yKODFv8R(C60>pXXt(I%t|{!<WAARn~lr9JX7O!Qq_2V7JPJ@-;RR0saQ?H;zG#cLMzybqK@i3@aTGvX0@Z#5enWe+9wcLUZw-{cJBYK`|W8YY?Rm{gTkweP5HPp_ge97^atRP6-Noit1Szo=8$VbfiIRR@E*q?IBKM%?y2w-<gG06&N>57lYuRW;}F`dl)*mXS**Nn{Mu8<!OtR9QwBWbL7jDpd>wTa?1`wgKS}3EoxQpWL9+)!=oxXNXozT+@tzqV49O{noKv}*L-!FQVddt>9)QH#kg2&ierF_FMtg#n+ebcKHv)#c9Rl?9CHk^|HH7h2xmf(zS>%%K*dZ|41?X&n8$}rQIYP@BN5(mI=b<(xD{BtAregyI`HSn3ZWx#f9lT3^^T*4o~J0Ehw0ec1(-Cq2?UPn6^oLwLm^lMHx63~d9Tw&%h8pGLV<{Eq%gb`8w&wUU^SDRrC>IbZ2<N@N*Zl!7P2Q1qoI1lhbYnlqgi~F(;799()2pS=3T*#L9YHFL-Pa)VA_Ap8D|kQe-741=SUJCO=G^Zq}MMxhAOX{ue61qM^6RsQ}@4BeieoBQ}0s7$qP?!5FKuKOhcZ=o*UE+cmZC$=gB*bcL@0ID@HVrVBl%8`;6Qf8fqM9su3LHoHH!~IDL?Nt%gQ(iQ$0ZE{SSqrScVh7ALpErmbL<tDCP=!tu<V2CC-tUIgRZ^5dlPb*vtli6bPu?P<3!JTMxSgOpOWPp;0#U%&tTAnOzl0<&?>&wtLMC_Jsd6<*-rx2Y~4`UCXd*DoLl*(zQ=zHXS#3HC;}fodRJ9uEBl0nJp7-k1(yPxT7Pm<JoMN1!f$*u*(tb<)O^mm4$Aa|tD{pMzL3Xss8cTGyM4fkVQ1YH4W|3Q%1>Ru|=>Z6oWlinTTyR}XWhv(4vF|MUCze_(19-utApEh@zmC0LMSADqfdo<xcgrD2gRYSCMpsN|atvw@P<X2QnIeLQ=2C2^p6BfQx^4_|O!fI<;a%U=LfvH`c4V1{TLOx5OAR`>l&O<QbYs_B)u3~60|Jgdc~SAu*4Fg_%74jM`$Oz*l__5>9D5|CF2dw}h8yzq6VEh3FA<BjIVsTaxO+@Rc}x-++P&8K{u;)_Fu60hUUhQ6=SazKh7lmoQ*rCG?)c5st-AXKVM?o&qQn9`C+E2uKis4mXKix6wIb{M0S(qhNt9MAxY=&!z|H$Rbfyc*px{>^3-)Q5f@9MFbgw+8&&47*(ug9tB*fh#g8#ox8GLe?6$i>|dEqzp$Q8Js9lKp{aLT(WH=EpqMeRFU7>Q856TGSPLgXJGGfWx<q?!t4CiY)0SG0BoGwIMQf|f~0s-L4*yb?4#@n!T_j~ko<a2Yv365?6*WxS89NhMPaj3X6+&Y>td5LxzH{cUESx*?Ga{=er@VYWdtXWk8H9(Cbj|N*?d*w>d;EWC9K+=cdoH90AqjezQO=F(->{ij*OWO0yD5e+Jt+~l1oeV?&$r4EYx8@ZhoOiJxmI8fsv`r9cw1Jf&MnwHFVjMBz1YTb$=&Q8U5}~l6+^ipn!QiW;;G33R+Q(-sSGJM!}F_n;UE%jGNtPYR?xB3M_7-K?z%2uPn<2;=<ATy-K{MB~2f9I3*`fY1{tYB%y6jsy5Gx_#xCef)7^CrLKQC^S`@H)7KPfgCWABr?$Ql;c+k)m4>(4Lx0NOzt2d<erz$4o7}8CCV32Y5`(QB8Q^qtP$QxFIrc^tvv%l#XHE6-LhcI~(9+0wH_O7!W~Sa)n-m&-$bde1DN8o?n_JxEDtZus4IS=}{q7bx(dr!00yWg5Gku?NFya+cMwUDKK?6OnaLS%d-@r8Y!3O3j-@uY_zqGNaCS;c>1Mz9cz0jSm4SE=DWM5QifJ}u~$z}GROLhJmhpD;Nv@ikXve?D6!lfbhKj*_;J1!S;6*0ZU9=+0RLHDn<IcbY|Df2KE<%zXcIC$E;p#igZS>7`i6^T+oT%F%FZTAs(iK2V}x{B&XB+V3BizdrRm91(~e(0Is2h%%>O3$3aL%c<bszYLlOwNwo&NZ50tlaPpd!(C!DSXsB$7%?FjD55<Q4$5{v5Kv+gKo819Y<f}g`M4&6bxJNV>EMvLQWD13b7$DoCU}BuQB`dzlr*cqYSHHWA~6#G-vfQ=Zpx-&e~L2oeqy#D+zs|j!N%V%Ht7bK8^nQ*o4<13<LVhxM=0NE@SbZRa>{lGi2&);3gH;(Y8qyP5kQ#$S&`6GK#SRaBbSHgFb_h*HRxwQz>b7ES7g$#w9SIeH{d3>eO{CDtCU@G4fHt`()W(h6Q50L{(NoMYd?Jddy<7^_n2w<~|tRLj%~xY}p1nIE+p+0@R8mjj>NUNllst9TA2RQ1OgF<#?PD@Fhi8AH1#ZeOo0fZDX7O^vTF4)bYWG(UayF<esuz)gUgdViXQ+?i%eBI$#uJ2WBJvj*wF_>Z(f8t%9kUC&KIy@C55T<dJw9@%gP6bHQ{`xu`5Pk<@_UU(98wz8Ng))M&i1oar|qI2^{_V05Yk0-u$n?TTNm1z<^gV=b<mOQdkAHIp8u)vf2C74vF~cd-U4v??yfE5U=8LNw|=Y><_u_kX}h8Ry+*q=27<R@*F%TaT`+sLOH=zFM@u5t>+fH^8kFggCE+7hrhmTx+V%>x%Cj(;Iw`5YT8U@uxTrpL1`vF3&YaTgF?R>cG>gixw~2g7%aT6m!O9Y8WLDy#Tx-q6ArcbYNt&yd1v#1u;5Bc9L#Hn0f%?UNYJ~FQ@+NY!V(0OCwq9yxQI^@D4;bCtEGYOERAhq+pO0ReSIeMC)x8bks&pbT^6KPB8r4FOqmT<dmG%!%Frj!l~_P?HbaI#nbq3jRk6OIGmhgtbL+d!0Yl-o4!P>TuA%^_@BnI4jM#zUE5#(@>_fsI|fczX0G4AjyB;NZNk^-5SDbi)f2`e!N7Ndffprn!q8e2-HiUl7JRhMn)L+tD)<rL9z0+C`p@tGpjy}cWJ-X{Y~2^XXantUV`B&@`|#+IdZ*N12fe%+U7Hv_1B2d@#?fNQJPYKgi64XA{hm7?u}YpPcNW``d+l$24<oM#tlo4rh)$B*_PBdUOnj{@+=6*vN*nTERHF=roi|o|joaWK9-NJ+HJt<{h4NY08Cg69I^D4ap>e287<1IMMfINEI9W8tW`?(6GX+Rf6t6MlNNSv#;<OyeNcMQ@yJjHgX*i3mI{B5c7v)fdXu~DvKEEhdVYFiSEt9O2*<~CWGp3;7KWw*}G-=^ob>ae)#aP!pA!BHISc%q1Z7aQZZqL<3xT;dDb>I!$!O&0J`@}jkyRDC=+I~Zq??Z*(d*Xs64Z@)h<9iH`h*S>-Jo@!5+}H960z3gPvu}6HGjxl0zd-~x{|Lu@abID;X(jTGNzD1quaHPi3mMEZS@xI;9xQX$iWcz5?VdvCgCtg?tD#1h{kpW$+FcKPSMqU|+y<3V&u~ssXWJ&K8RQUzTs@f8>)wG|rknstW1ThSV^LFHmpnYkh`i!ATA`!K8B@`2Ue!M^N=|`6ZQ+qX5K;mi(7fR{f|*WTZ=(YYvt?3BjlK2K5%)Ejvz!wg-nL=H=0yOL>SvXaS1OWKUDrjd{p){Sw3xc{2A;Z6Y(QS1P;2$R{>&g?En3UhqAmCC#hFrDpMYd%VnlJ=YWZy{h2vsrKVfP7(jQVcj#Su`VsTVS=@wbNZ3Pyw0zPpFpAR+z2{ivo)XLpoh}B()xOezVk}D-^{qK7AP1QjRila!B-{L#QD3@TuTNR!KOtZKR+*B0T`b5Fna!FOK9U~CAEnx|NeDgoQ*Khy+{cpeg{#_f#w@Tam`tFYhQGW6>&v<?F87EUSJr=*eqsB!Un9nq-dVJ3hn*9Ug=^k^ssnYavocsh>ej4xTGzqX>AUW{?VbeQ?yQutB)FA_ZnyG(~P3|@aQryF-s7<aNTY~w00T__>XsA3>fow3?<-Ilbq*Ys1l1XtorqO6zg8^y3kp^9IOGOEcduw3`i+lzt#jCG48UP-H-1D*(C>`XNhh4!Pfm6nMp;J+1OvEhk;u6b!5TP0(!ve<G|Ez7-V9(0NT8Mm{$~sNHSpGwG)fCmLSF=K)uT_LbXEVBch2A!HjhtWHsB3MeqFmWhA!2=2olzO!mA&@!dqs<=@F8PSLzs>f|FYm<JtvoGah{P(;F^jOW*1#okHN3*u7GDfqvz8@H;Yr#9O4k?5T~MUn(Q2&j-qk^>X@ddL-b_!_uf8?+rZSr`7FZ|Ywql_(E18*gv7#IBqUe62Ut;tZ3Q&)PV}_uVi~h6TXHI@vq{hs6F%&_&cOZw#V{yX0g%dcPHN2U6Hqo94iIemIwqwUiFc#azkE5(wsWe}ZOLG#ry;ey`UMf_De{PAsd$Y=@mgDjGg$y|{o)gzN-05|iNksMKaE6{CF6-IUKfsHsI^)4Dyo{DrHUvl;C&(&HO<?tvS3*K%4FE8aj^<6KdL(>v$ss9M!7t}b8xPPf%Vcg6#mv!Cq^F35el3!_l+gN3Ca=^)-~(Gj=>Ij&RJr3F@uiLYLfB#GPcx5rV2<!y+?z($aC;ht}>Eju{bX&S<?}SI{^#w`{E!rqF-G_P2E67oRarIt*b_5VXHb}-yhjh>+^h06u;YqDO2aEGW8H=V{YAePUJ?(`-1|6NME42vNpSg{SrQ(P_2O+8$bv)?wV;r_cfenik9xol%QCfWaqT@t#zKjS5T!6fKWvW95S;=loyDk5eHxt56U<~*U>Kr_Dvn)g+VgMQqyFF;d9leob7P`!Dna^^?Dia4-sV~c>fGNSO1148vkzfaa`2qo~K1QJmAyEZ3<b*vm=$_Y=SjeNlb=YFLPrCprAb(Rkp`ut~c^=xI@klnCFmd-qfygFF05~w)W{Q6&3Y_Mp&?ozKPi`_@j>MTWv*K_YOR$|5<p@reJKLb+%*xq4*o#Kvle>{x-?Zuu(A!g+jP?1Z86I=^uk0Z%H)N4qKEhp>3FSr@d*L{DH=$lYu;gt!fZmB4E6;*c36`kpNE!J4J}p5oPue*Swau6h-oN-iNEg0s<Ud1dhGDLj`*(isMzXhYP7-AreayThfn!nQe){ELdTH3Uenkl}HUY$ewCb!T^VOE13$R<>4f^#kmKHN_tOLL7@)3YZa9$=g+8q2)n|c$thI_0&rtU?v|}h7K2{9s<ab2k2OC*X|nmf=Rfp~!Oo^VQ!AGyDT%sna=k<h*F9Ih7D1wj@beK$IH1;eANd;smROw6VQxFFw9c10-C`gato~w4RkC(&52a-AXz0rm++eaI-ceRyi=~KSAfAuztfB-SPz~Z90yKY8GetUr6>)ThbuJ08AB+jDoXES&3*5}i1lDGCo&YTftebD{WbY2zTdIZS-T-7o8-13=^I#+JV*f|nW72HU8Qg6(=_RGtG8bj^sMZW^ph;bbsiw3R<%ivl3|X5y)iP2I>h=E_Qf0ibugORw_Bkm9B|G}=NpZNJ3j=-5t5<Qar)5bsE$)c9)Q3f%i*FhII?NCkCpUt*F&uQN6aZsx)qgy~f_Ye)<9fSGF(t6l?u4>bXddh$_mB*XKA;W|OTE&Uik^$+(rNg*KGuH%iaigzO&>#MY|<E;Pp6tQN>P0VSI19EHmruKt>_r1r9l)))>!Z?dsb7>B?j3dK~9QY$7L{cvWqokM5Mfx8(Ms(yejP2d(3*w)#i%Uc7RsW-1p}g;N%A`36*;JC>YHp8fqqsfKV)g>@wahYcvt?3XTCQ=ZL`$B25#pEKIREu(<88zu$<*Yqk-K*ZV8K`E|8-jhd(zcx`X49l(zX><L?^$xu556H6<a@a!xr&T0R_D)zo9x(?ky6GWerh3Q<QEkW-fFb|_x2Pft<B)U-WooF>?^?DlywM(-sABTQ9F!?a#;!Z(z-n;c=^ErXZP6D%Ti&kH07it4WF*lobU745ZKO!Qn`u27b65r^7wAdB0U)-i61P2x(Q+hh(q|Tr59JwkQM|Fd7%<Wc66Q-Y(&z@x+_=9wbu;mkId(u>~?wqXuevq<BuD!v^kmW{}wUc`l2aiBf?sLks6pDlR@tCR><*EuAJtlJ<;+ggLDqn9v-Tv@<qN6a+H#OHT@#p*=AZTt9S_5aNM1Nu!dUFRZX^HbZxy{pQXqfLSYyE|x8GlSeWfM?kL;NJ{ZWZYPpC|3CQt_Jo6GvH$0ul1W>+5S?8}Pp8*Pd<zY0l~dEp1J@=6Q2vCjzBgHz7!P8#qq>u`Y(4(A>qmw+T@22vSMPRL>UpD72O@p3q{HkToc?Gpx(vT58pqdFdOt$6wA2+a?K6rnEXk%Y;S>%V0~FS-pf4agh=^bWz9ia#%)Jj!9{uzalX>pr$B2@2XCo#q<tYCKl#^tBrG&{3@rpDC?GpzlTW(6i?kIJd50_{53D?T4#Xo^C-~p9gCk}7A{j^&Ci9aQ}8)vek`~H=dn?2nD!mBmF0Gqp>QEki*F`40$|ewIAu`?4q~?!We?VZFF`v`urDBQ@N<fZzUfWW3sY@T_Vu`is~g$;MHP9xH58(wt|8Iyg<ypgAITkcMYXxJoAA5@uE?B<P!XWsQR~>aqXbowXm&I43##>b5al$8_1sHSf+5(XcP27@CTEr1JGD_#>zi+M_(*HJnaiijYooacF;^-=@RRsXWodd)R|~;litJeWY(q%mG`>usXG<$qww5jKz6b|F9dE2=MtLhCyu{F&Rk{q6s8p*AMP;WvP<4T@Gq1n-x@^+rVH!hCZASy|tSV`ThCt!ZWb-+iWFz<`7`x=l3qJ$O_CZTY_HGE$7~7aALIe-KwauS!?<rUxhfmw&m`+nU0+U4qj3ng$=B6_HmPD!UsRtAl0=Cp^fZeg!r;!MBebM`C3(h@}`9B_r;JI(BV{&znX!Xe^J~?~lQO11Re4DZ%9&%Jst_d$WI0n3{^s(v%gJ)6KJ;(&PdC10wvUhD~6FeWiTt7ms@=Yb5J<`sJ^a(2}QzOTh1zS9_pE?qC5$ji%m(6K#OZw!^=85qWoyM0X3f<9tO!-Iweb$#avGXI=wl3<_sRe}>sRj>Tv=A$jm2LABt}P1F#rXsczUUWQu~A>F;o(rKH1*V-+hUY{Mu|P>IiV?e(i+9h==xCmYKtb@Ssl6mP1z;4<gQ9vhrC4S-|MP*#*%v&t&N<6#G<&qW=L#wXW#9%#L2Fhf~i%Ty4Hp4>C~ckmU#M0$K!i#QT-hpYw3EQ@QT1BrlijiEU&{r)fz1|(-^ZF&iN01H4~MKv-DL2_bu5?U?~dU>EKz|LKL*$!RdMqck=Z=e9wkxG3X9)LCInOY%SfK5jLl0(g<+z<^zw9OdPP3{m{$X9Uj7VT+d5fLf#};_0_%snJhq3hh3!6<65$%+kBmA5#BOYY+s1oWOXop-Dx_{y)5JU2Sjp8_fcxd*HKozHi$rR{)W97`p$o9%)!lQaI=^iM$-Ol`W&gK-Z);@4j2hUlGgs{RWJ{JKEIQix_D$H_4^RBi2cFdsQ>2FA+)=If)tmqm)+mk)APk<g|Z;!Py)v+G*1?;%;&I;%l#4W2($lbjUTm8H=j$rB(9qC2Wu>)sVwJ~Cc+u#cO2GR`edtefdahj4NRZnd|UcACRe5EC%|Sed!NyXU69E#hw|{$79NgJQqYEX>31pMzp5*?-Y`v6X`cmWZ@!`Pn+(Ub^50~m=!fi%WFO5#f7>-xBsFm*W!xi&ggggZh%|y(23h4EC-US8!ZA|TfzY|LDuqc`M`{_yR+_ocxiWjLrX8nf9@G1Z9+$U}!Ioimb59&_@Y{1{sv(zmpk-9FXc?9U^)KMRQB7_xJ#GSwsadVQCd`MyX*EUEOYivpJhptJ?AC6E{Beb-j9VAF6gqQ6GC-sjnPvtZu&Hr>0wskgoGlr^(@|)n#3YoYuKiLd!~wU{ROkX=TIPz$(HfR<aP2(TF+`d=#7r!lN6zhOMx!m8O1%x1umGbpU>;yTQ=@PnQ#YfK8=JeW$omzN9@CiphP|drG%wU4h>UM1OOp(kkEi(owjg$SRM(f<2e$KbT689jff^<)|1{OCUIm|*F_AU!oG26YGqVFHD?vPo-IgI@4QEwEOr`%k@b9%r<syaX)CHE?EnHC}7CyF*futIpLt)dUs7}-`PPR0)f<=ncU~?N7-UaEKS3s>FqBDf@p3<}3@n@1L1}U3IPqf;(qc%hJ*;AF6)ikDt4r=&*mZ=_g@OX*h8;@&l+{<i3`kiNjV%v}Z{yoKlOK#~@lZ&O0?L#<uB}7O?D;;fFM&hoW;z^PiJV5Ph*U_%1u}W)O^Xz3(7kdG~m^e-FmcgAGf<I*h#S5aafpXz*SLp1|-}wAzf@9Anj~tQF@&!x>Lsr|;2+(Ic_8Yx`A~A!{PfkV)?Q_3>F3Kv|Y>$t8Ljb$Et>u!8(Sxl|r{R^-`{kc~gM|$cR+c3Vw$S_Otz4DpyRO(_RrhS73z0%4pP_Y!7Ujp=$CSqwae3HL;+XP}eSH2fI9Y`n8G$E_NqsAct;a0qCgsUzYohuNs9!uk48O_-MUchN`G2kVF;^c%clv#_sX{)Jf@3+-?S4y;T83fd$NyT8=lI5UWnTOxN~q)4DaHe?So<#DLaiP&BPcyB`1ykCgNbqrAlXu}%`cS~Yr&yAhM}snlsX%u0zsgV$oYPbccxA?rPPZBI`_?duE^G|D)QA>?HW2C$4Bt)Jq_L9Fvcg>%P;>i+dsaqAO8HVzKAn<g~sh<azDvdsmx!l;~bn(zJAQxNS9GAnoUst@+d|MytH_j$1qj}n$~3t^$*L>fFCwGhl2(cwH!WBQj@>Ew`t>opGb&@lz&r?tB?vWcQh0Y*FQ&57&fSCW3w~xdcBu5$A1L(zdpTyU;h4=zy0##ru?sukAJ*;3cCl#rQgqeo)t@a5=Ec!nf+q8fwuVPr12I5zBJL+d&9tk+VE=;X<tZ^SH~yXvR_Hja^kvXA$*Wv%gAP0DuYl5GT<q2N(o%k5{0{zL7oeHy{i7>Uw`@QumAJMfPVj1+)}Rm5PHXv(a?%d{k&lIDY4)>e*Aju^*Dn=jBsFX82NZEx>6mlaQT-<7qaEyJkIkL&b<2SbWIUsLSLmfdMQ`#i#pVkVUbFhcms#fF;vwFlLWOSbTjOI=^FA~b#+!@E>_biSJnLWk#iy4eo>Xlwbuq7tq1Z)dPn^t)xbcenw`XtC=3L@t%SW)4kvh^(Q`(n^!yt#Pg~s4%X9}37gQ|gLC^l)iCevEHXiup#ci-tH|v%Us`m5mJ=~b+*e`y9D`+WEuVVHgZl<G0!i0fE3C%(MVqXeo&MeKI!iZ)W%beSB?Q&gw8Zvvyk4M4q+PkDrr&}@v1Hzvk+#V{djJxJjNP*41NR`>`O;kVR8i+J?7{CyxHjv`gdqqxZnwH361{41>WBhe?+Z!lB3J`4q;J78eI7W(NBf^+O>qr@f!Qb2%{1_iwEV4Leyf)OR)KQ>{m69A#_-y=M#L*@pG8lOoL?cdn4t-a6Mu>BY(Q>b~K7UHxWTsH}l}aGnk5huepls{?hgo}T^}`C*!D1q{KGlpSGMI$h+f|WV(XKAkk69__YunReQ`5G^taH>ch^b+HsG^*V(UNV%p39}e9I#qtx7cD#tXE3BJC~N-SqBJ|sZu8fj~`z}7LzPOTs4EBNtBu-4`+Le5*{s!3D;zAUz6tx$is=#w4c9RsOsisEn8L8NR=E<|BlWXKpa;7u#lo_O*9|hKIRj)u=seomBbk`L!Rv9Zc>_$FloSZnW3xvly*dc7C78#J#6)*X2ZhQCc#10K!^NcUh|m$LJRIEQR)75%K`|ExjZ3yYJl-)vZP?20e=)^L#`wLG+EC}OEk&o_sz}Y&>y-#9JPddTk56uvb@n-uT04K#|(z9Yph4Du!yg@n4%ZN^|c;>3~Pt3&s#XJM}4?08Uo86y1}*VhVVwplGt$~xBH<R@1`7~Pf8$8Gu9=M$k63VuK{X96>WL=_L0t+KG)jD&1y-jmp%EBNu#=rv*D&wRn}H>%Lr+vsjbYat_=lp53AOT*R2Mg)KCbcD(eJJU3SDOryY(|%9~O_FP7w>d^Q=V4SlB$S4re^jv!v%es$l>_;x7x)J(g@!P?%imP1xH0Ow&1k!;<^)8R8;fBBfiXe;qjr$yz4DE-42`;{g$6e6K$ZKoXDp++I<a2#qQCKrA>XOOGQ^~yULu;WnQkD>99Z46O`dh;%dwDrEGnnS<lvd#q-VRqO|Rl~#-01`3}y}=G=I^M)H01;CeYA0NTUQ)?P5E{umN5#p}Wjiq5_>{ZcY$^15s21Q-sGx)kJE*je%uG`x3q;WQHWKH(JPfWki|%0BwY$o>ysTg0z~5W$(WPRr?$eCIS5#~OE1udo840mk;S8yOCSJYyE(_gt*tG8n7FGpYv`rf|&oJSfS-dsR0`hFn#J`SVY<JC1N<JzDrfPnfRoriqUBUW+CTm1kKJIomjGdiiM#~9)%#q;>Q@i0Jj=e9wMFdOf&7*0U!)Wv3P+P?WsuuA^z>pdVb5xEqV5eqQs*xneIE_&=SsdLntDo}~-t=JeRNENuxQHP~Nn*Q%I-F^wt~getVG_Ka`&1DS490h>5}_aNJV$F6eS^pw&1A;p`I=+lN{!6tNOyOntpT^Orwb{;x25qL;_0bRmm)bY`27HkOT^5STDzTd8EkbRChbfG_%pD9Pgbq2>(g8d9X{6AEFBh-+IGX4EF=$jE1YENR`p_H-4YF7=()J^_A({r`K{%*6@g+x$5@w46Jq6t<NwDaGWE^*O0Twetb8X<(4jHBBWyktstr%p&BtHA|Gnh$Fy6fo-XcZsZoiqYA*t1ujB7SP<{HX)uwK6EVez?r&S38B8K7T817;d%aI{dWq-@Z`CPOG#b5K!sKEc^&)rSWE_%IaT@Jf-}fj@Pz1jmGAdmTUydwf{W82lyr16FY50j17!J*JyyR>mEGs+q7dIYeMrWmP3CVeQb#PBD_|dYPw*(xr*xM{iVpqRvJiz~&Cn95;7+790az2r>0f!;MOYl$6Fd+P(Gb4|Px4Hqz?zNvy($V+H?gp7q})e7l<D8~JsGY6Wa#8dstK$zMa01Tc>Th7=_<W|bSHxMuCO5khEv4bjmvZ0JNS7NzLV`foxXj_@cEeJ)C8EULbM?KsiFsP;F5hyV3ZPGlpp;X(leVpS!|ZtX$#e>FH%OP)q%TgYZu^(36w__@jc^ZWOZ`WxL)uH5P|@{G_Ra#K)_H&R)lO7sz-dTc?V7Qf~U0Wc1|;~kiq)OquS(}rbfEVOaxtF)@ZNh-AoHia3dWp(ce)-W09FJj)NK~4tSYI8X*NQ@B=ge8cyN0><IExTWQlu|}WI6TNF=SZ1tZRb2bC%YyZ>QWLYlW|MPA2q0eozHFU@6@g-<u;Esz%o9^C<mM7O!IlP<2eUfwW;uRmh9K&ZSmo%Jj%n$(}XxCG}&lsrf{6q^zdSi?1{Bj<&qaS&$}D4J;_@Rp<}p<t?))UGxi;3<-;@Uy|yuDT=h5*?(-~k*y|50{^r_>NPwv;3_zbWZQjnoS!HV%+U=NB0)YqOU+v*}vni#o+WZN@3J{kWBc!@6!1#LCL|VJInj>WTgR1+}pcPbjOF#*R*7`4T1!Z_>f^m<wEFw4n2o*^XS#s?HpE`HPpeNU4!0`<*0p!&9d^MTzdkMDBX#SkYY@a1HJ_5is<t0E!OzOT5g_ui?xF|KbN@5077V(agc4dszevw1WmGOLaz3_)fRV@F+3hd&Y?PoEIHbvJ}9Z^iE$0Mz{?RVB=^t-B|yjCkL)s{f5=$HyVJky<ayJuv)tI|m<-Eu(-#soDX&MnjCXvbiOwld#3D{XT!>wy1d&gIi|)?4-=s}Uf|pT3S^w8?YFaHh7RU+FfmBxGiMvoKFud_mEJcwk@{%TG2uGiHtlxVChS0qdh1gi)5mF`BKh=0ZhVgp1C?8E445T?+)fNd}t{#Gx9-*md~|2Om$OYVk2)f9(3UA_p28)Q{=u(loTe9lN0suQLm4thkx+^)G$l$8JmueV}migZ7xDy)}<<OnHNohySmC`7M!?O{&<A-F({`<4)FOfDZ6976L8}M;{7&DIiXBKx`dXyQm9|qz-JI@SHh&SHNrRwp*qd$E~JB>@kYH&DuaVQbmBI^^)x)k9~L7025)bcc~TfEUbY;i%Nx~$TIdrW9*AI(2S`M`+H;Iu^*GF6?38`b)wVt%u}r4@TYX29R$?HFT%jsZ7nkPbBo&y`~=TCl;xyjp|9fCZL$-PX0%l#91nH1+YZ8u>>edRL(F5LmBIuJX{Mj&)ncw97}1^&{=tQd#T9E8K{8mzLLdd(zn{-8`@^B=>~-+cm;<WgIP^_?4*Pd4x6m-bVeq@ivQ!;QErvDkU^ru5a2UrEpyt6Zxpx+>v3{KsTT!0*!>!jgf~lcRZPH7=CPXrv^6^VY@=;2mP_2<LOqV803bK!ZtuEWTMe>1RhHMp&u_cvJe0ZiFgs&@Ltf}JPbBfhH2s2@vN43T{cCK-fJP%Q3|3;ed)vGu5u0^7)Vj-A}{6m-RlCB=d;T`q~Wu(Za0}iOfug7t;o7tE(y7H)MxtuD!kn{W;Cwu%bi37Cvi1vjpv>ywNl>=H8RqCyo=a!CC-zYXU20D|Gil@p~?>{MDJ&tQjjE)LYoE8~_4pu^yjAOi;tK7jQ++khwgJ+Ip3DY<>igkh9EQrvMuMG|cV|Pt9w+$x20rB%A>~i-qC%48Q?Wn15l;XoSC_<?gkkLv-O{Jvd@%{R?NQlfZND2%|s;wB=%$FSg92hMVn5k`r0__*~s?cO%s02wN(=-bfP!G8i$GB$aQ8$v1o9$LYm0%*02~NqNJjaKUG_ps`IIV5*HDNXo&z6rdj8mrVt%hm!2&CiLe}X~EqHsmPi4=T{1)jSG!=%Jqs^v~1o-oEPG)zbxAdR*lc1m%8$VHh4+bDhUXo8&bB&?B6?YrZgpT`^Kn~^bS!3PXXaGn}e&1iiFbk3OgCSB)qnP;2XjuwJ>tInP>^0D*cCgw=D837OzJ_8ugYL`kS04W<Hj-<FnJp1ZwsdtWfZjBrzet~+eItx+(cnZ6nrZ>TZjPfPr&IVw?x#bE;j`EU_ZDjhNg$|UG90<mk6DfT`bB9E0LyvNh8u<l>Eu3$QmL^=4Q8>d)_s(K`j>~ALhpG6C`M}_SoniXX8ANQDlv2bZpTwDCS_^_9mODY6p9GO!3^tqVG7Q<tpR|%C5%9fY{a<U5>)BvU#H@I`PQ+H>00snn8ji?|&EjV>m(4P=ILXE~4H~c1NC*3RZ$ynAcIhTo9VzRNPO@(7V%39RJ6pm13nxffx5dR!D`d8gKHal~v<9Q+c#STp+YDBGNSQQ~jAcy{oQ~EuyKZd-Mqd~8)oPTT3-w%`dj;!0*|~Bpk8DRIW<wW(ldsl2i~8GA*)*<R?Q;_nJsH;1nwDEZBGCllZ66bf_tVK)*Ty)HazVU8hKJiUbzHZsiQEsq;SPO$^Q#8PI1~2<K*-}7v&@Aowf!8O)W3k3B=9NRHXQLrmc>(BpU!83gDDn5j<ORLZ{SiuJ8nZ;oxM+ehSv#t2xzy47}qwsEkqb<qDn}bTZ|x9Ke*bTPhq-+xcn^nfN`^13uTp)1@sbt7-WgJ3MPzkm2GjSg~nmiOXg_Hat?r{%Y;#Z?iE}*zWcmI3q=ipe6X%0i981B1EI73+m;daCh18C-qkWJ3CXu{+Z$HU?}Xaw1X!yZg!a2pvA|F0FxNw<wi)58aaSozao<}tATKXoBP7X1C?qtZo)BRqpcHpl`ID202P4*P$XgnnCTfz>035XASH0uywB}4ZqzFM#Y!HRTJUrbP&|zdO<T>cGK@H(6{rQ(QngXwlviJx>asBRN{D!15gHn8m4x|<XHage=YeUB=-)P^{CzzbwPir@>N>0<S@wezSk6?8Z&cC#S9QR%K+zzQN<XmS5Ej9Bv?3_9kb_URtYtjj8iJCzf#s5=hcNomW!n{kF&t8dgc2l86!kz)weQ-1=B&q_&hwA`ePspOn3mP2LnPwfKjLH{zfS)a)rfzCjYSSYjWrLd3GIg@|Ifjnk0I11AS6?7zDg;eXO5ajmBKwsVG!VA)!A6*<-p+Iy%9^?A&33~kA*TE`m$W~1TcUBt|6bM0)a@-*FN}*{*TEE}KDUJ`LS{m!U@=K8l^n@NNH}T|c)dIO-UXSB?&gW+wIC6y36kr%tSuDU`W_K2i-_Olwtqe^nSN}RNubq}RFSB$u)&?%dBpUSj}iIE56=K3vR?+l`Ds>$m`jcDMFGU?YY{?F7YOXx#cdnI)3Z!BoQ59~bD8?Ju~-aP2x-Cg9=C@$y8V{GS73wM5}|k@&RngcK}0{5s53dSXZ)n4X5cw0_b9bBCwi?Pk|jdJa^L9OVDt}g{s#6l^lfEQLZRn%o$$$Wv*nW3iJOKY<qwC9RT-tG<V?t3F2rj%tZteq!1e?QQKix+tz4r3Kth13fv^Tn(=^PjQmHH({%~4d#*Hw*FjLLz&9YOwi>c0U$gOs|kHczrIYv;-q6i->s1S965@=b@IczpJl2!`R2p^o)PUD<ogu4c{I-AtuR;%d*t;VvB0=;SM8uw+b7g|e+BB3igmlvpNPGf_zr-`FHiC}8Zg&c{_Ay4D*wmuj?4PhDYX@4dPg%Cs+VlcMqM9pqlsyd<#m(GcWQ5_4|J@k5-ROJd}L0X`hQME4M<b$xlXmGm(un>{rG$%ALE<asAsvcl+^+WGrTyx3BM1eC+%?bS+f&?Gq-h4bpl`{b*iIoG>Wogr>`X#77JuMwEuFkNk{3Ef%)a5nYV78Q0C~~r{c7Lb-y}Qru+|C^yyy~tp(e)yp23s{vCj13!Hwv_kmhx`>&@_#SJwYRCyHJFZ1CqqGX`1p{n^%}L?zYPFlxAX;0@>62UNufCq|Fph)m|}@uG8Y{8>dQ*IR7o;5|Yb}ITG{vWL8(dle2#)cQ>&UqDx_G>-4}ihiQ`vFb|IT%LXLVw7>V-V9lavlW>mIhKJB>=UjcdgDAQT?*?1Aa4KYqST7afD)Q&gQ7#Zhheu4IV??^EP!!m^lB&3?q#8kSQy-;z(t!$%qlVr<@d%M1Plc|SCzHuj{hHU*n2<v?L{(>;*lw`SOsP!^tZTl}^2P3ynrjmC2^49b>NGF^_wk|G)}1Z1W_*x{(RIvnA_9(oHqr5PWu|${2MWAqiTy#PLsMs4QZ(YHUG4c<)OZm45UUkV_Qsi(E<f5D4X2}R%)BI}Z>Oc#x$r=oLLlUBIeZFQF+`N^G4^9>3lpD<$7`^ofPcs%N9I>p8gyWslN7VTRsecK&<+we<>(^Mewlr{K>QzHS}vL72SZ&FCV@M&Flay;ZN8%EdFrh`Yul#$Fkr-ErOWK+vN`0pOL$L0Jf_-yi>qa%A4aYqHZ8kd4TZ|D61+8CZs%;?fPb@gi7oOqk!QBkEf+CtPi~HrVY&9H6EqR&QsEP;T~Giq7rIHzS|B=W63>*BL`HjIKZh<0UnzYlLrn{^9Et*{injX`0W4IMAbSMzoPxf_j#lqCs<d05plhO*m3519<=Qip;dQYYM0M20M9b;}tcfpN2p%B}Wy8_GRUu2|u%y&fObY=e*;Ap4R5#A~0v5n5ha@}x4H7SMW~XaY$R!a4n!3Xeo%Dh~dbubASjOO=eT&ix8*dBtZ-ZvG-iD7<1j%t32v{s{D3uyR^&h5neVqA23#(@v>}Il5@Iv%-1M~yk+GzHXhHbQ`VLUu=Lj;p&%o^IxsCam%>#<GtJRY&jByMNFe!P5a^#!!eiPU?rh%z+rw7|Sx4@N**!=uMs7m>N~^Z|7d0z_y`#UK{pot)xA;7KKGTy=A$SO~&)f>uUojHCPX0^Z_S65z?a-$yFL8EzJZLrcv^w8Ilw)HvF!2U`d}c^Xz~5Aqfqw6`*U7GV4md?rT|rw90)sBrg#@ebl7b;&(}@jE?C{PL5?#0$|VPQGP`9#bOEwz_3jKBbPr2nw|notZZTZgbDd=S*=HyEUn*AW&s+TJ;=bTx=<4tBf@|)NS<*Lqud>+fNVO(%PU+g&K5VteY*z>t{`(GM-R)n=yh*N}pevv+s6e6OlamFi%B>;PN`r>dDNV?DNLGjux(VZGB)<MRq@TeM5_Kv&Cv%8<j|n1;#&@l23r}f-HGb_fIgiSSH&g%D6i~G<2CeTh~|}H4a|%ElnH(k7{b9_n=kyfo=RuaOoJcx?eflg^Rhw#FuyANMd4(?QU8XLQHrp^frKvtS&n;3+@X_tsYq;OBuC|;YBXKBCzh5Jr{~m83~<~K1H_kvoZ9(%lKd%9C9j{V(yKXKtHmzbKkpLi%U;);yQCK;|i>~x9P{M<A7B0NtmvpSj>GRw>Yz>qL{*K5o4N3jnf}*b*JUDvW-n$2+C7<G&Ns+4qXAkS_qtLM1kWoY#OgStj)awb1{iN0x&rjBz3?#mBP<i<d9ri(PQn5Aj6Co;OuZF4knfso-HLQ9zoO1;dQA>wNLtCXX82k-43W|wJxjOLyQ53_;sPZ%tNON{6#qrDbCHaTZkuCh(py|w}wKU-*+1NJZJ-h(32@NpVT(R3(d>jIAQdK$9C>Xt>QKhqhAdPqwFPdBLM*hI^iD^(P8v*TKAMlRAvRx=~g^fT>3eK60YFSnEbSjjQV0$^RP505&fL4SMJNRO7X5-ZER7(B`Qn7F_w6dxBv4MQS;ze*TCX7uLP<Y5qY}U-zrZqn$g(VzU&M5hkziDfc}U*V7ug|3*kci{(0+<y?yAAvL~~u3xhE95UBUWn7d4A%#+ecY#`7?F3dRQL~=mzBYLiKw^A16&*S9RV<eiN;FYDh4+AM0=UbOMh~m&IYWC3yJ}$1a%A|vx`uTL@axmC-Vi2%Uo7csKZ@eZIpRvq*o@$A@)w{;)m5he=k<oJkw&*&h)K3i4fl{vtST9L`yCF=g_%!J}a7BNT8c!n8?MpVZbTcc#G}RbhS;66dT#KXULJNvkyNHeY;q-O(6sp;iy!42g+r*y$!V*GTjA2weG4WoSZ2Bk<?>>ygoJ$x0!<_GL4L^hB6JqKs9E^R*_YJ9H<V5L{Ca{4quC9(I%R*oh4m(~*#SxQ7IR?G?)MwH>Qx(MnYcH=nFt)vs!pA&2_bF@L{z6#S2EJ!0N(>>lF;FII(k0W>7Fr3)2`sa(VGPD0$>uhHrM9Sfcq36|{z*iYW2jAX#Y~^w!QuVkY3YYp?5eJ}f|ASG&l3r#tL*YIh1u;ZV6Cd_jaaR1;kH`ZT4%IYZ86o`Fkqk8288W#f&OI+Z{I(Uhq<+|k!Zz>U2&z<hU5GPw2J*_Xe|0VwdW(t_ZOI#?hQ@ANPz)5f|4~$kuSaU4dr}Q_C~0KBmj(Lo9uZRTmez2ikR=r<adyd*L-xvbQODyHV4@xpGL!N%k&P(P^+&eWBC{|OJMAC4#DOMhu{tk;sSA$JFw815?<6z$&kwD=G0IXiyB}i$sw6sx;D>+deZ0M_&#+<$m4?e@B}8am0}f_ALT0WR-uLo^UIQ@xzIWSu}R`6vnn!FCS0;&3SA>cG!o0CEt+~~h9wkwMzOgnz(5`0>i;85p-?Y!OcnnW@T%eJ6Lg04H9AH+Hd-sA4m15HA)jWjFuNMiJ&)x=v#2g?@oQIHMFdm&1+4t+7U8@@cDGsB<)n>1msrRe{oA-CrJhls)d4%W(V*Yb%^kv`9OKK}98DKbek;7~uk0?GFsU78QY_@p9-d_5NgY|`B?%lnfGNDZ!1^Fgn4c0fdVT-Ic{X2mSKj?hQ8F)k62cQ(?--1be`xD=ExIk`6-pwO)(K#2vd+_?j>Pf^clk2gn9%kBX2BGIg#d^#xXCdTJ6`AYzO<b)S!QMzrxDD)5wOr9XY`TOLykNaxe%MmSpuqu=<Xx>uv~la?QU{)TXUxkNm#?afYSeUa;%ZzY`)>h(_PDhv66_KjLlqV28m|8=Fh;|9GS|Dq(Z36d0wx3*LDPndS4H@;Kcdxe%<?o)x?N7*te0r0Ml|+oUg?^yOiIxnIf#vlwm8mzHUI)@Wm;In|dEp{&HK1^~1&>+4*aS!ms^M=0co^+Mn6;S(M71r-Ei(>~rQ^f?MEtt0LgAJ<VW1Qpr?E5Mh;&DA5Dm9k2gVNU>ont2)d`OZ0`kJW!sR514?jM}#y=j24P18G?|Iw<}#<#wNL`|3X)A>3A%vv%Wpnf_*`9{ustTzyE&ZvB3NN9UNyVWQ1xPe%a9oOflLD1qhC(@0p&@;8#O1#ZE#DNwHmWDyGNU4CiD<e=@AXhxGrTN}CTeLAP|1FQ4UK<V2xLH-e=_O>VQ3r8APJE+Eke;82aU$Rl0~)u8jDISvg4$Ed*6<wogVPD`O9#O|OvMyn%kyW6Qwel3NN5bbv$g16fJW|~VZ$qt6O+pQ03sO4!8ZM{0q`SIp2=a1OiuJ8Pn=VCw~ey$WONd$OD7E=fcNv;K#>4|}rmm%u9129Y#bAsPV8#vm42@-a^#$&6_2(@{@F*jAr$=2*AH@Fv=Af<1yayDxVUfV`2y>aO=#;MsCT$IYj2CyWK<@1~Cc)5vZe~Y+xRUcalNg#b#@EG#Nsxg5?=wXqjQw+E5&)ICsSEU2Jpiw30{qPEr<I)z|(BWHE!gkDPmo0l%bthT(K4Srjr|~(`q4VWm;-XS@ct2zdLNK!awhJZcIu(xjdOIfW|3owGC`G*n@1JqfAWJyogp1tY;Db_k$nn&Up^vRJQlYH^-xT`sp}GA`;o1{$*@O4Tw=ip#VRn56DcoW9pf*tpATZtv+8i^ri$IwO98JU5+bPzw8T|NUSly9(JUrHW28`S+jCziCa|>A$p1Lp2MyC>}g)fCxk)?K{P>Y<*ZKe^?=a>sMq87aygeB1cS164+nd@cj{kF+m{N61NV7v%_xd+#*bM368WlK%n5J>=QKicZ1bZas3ComiXNC0D<8tt1Y=di>ZojVD;3zZ?3m~(q0a3(%M3c%==CLtLglko#Z!AQ|Cb5e(hnjLR?!>q$=+<c5U0qQbRt_n+Nof@-UB{Re$Uq)yZ-1<0>sz)j1h8oMmY(kT#9}EWGHwwJ6;PT-XC&ut2t8@WU-Zc@W$|CO7hmIPwr4SNgIbXCVfe&6>elPe<KbWq$EyRAT!8w=F)<U-Q+t_1fRfgdSnGh3Fcaic-ZWenjNkJjc#QhdurbN3B(@Yh#x1UzCNU-vWL{M>lPq`oM9pj8$@5rto!lMbflvf8Bn!2Ql*gEVeQGX1)h0>R_AgJP_X|QLjPRmy;@8h@MK7Zr$-w*WXAOHOR>%ac<cTMM5xV+AyOIX>#BWvtv*pR?HH?W(&1)>Dsh$`IWORG<wi?WWk<a0uK*lzuu=D>!eF2Ef<0mC8S&_b|G@`P=f+4F+FsuFiyvBOULp{*#~{e^Du<0p$n`SJEK<*`NC7D`tiQ~t4!&mRVdCYUR47o3??ah#lKrd^iGz-|5PZ&x4h^^50+;aAzUU&vzU{J+-wnCohn?m5EZNhz%2p^RfW(hVF(5XCmn|HJF1$N$>xiH+?_1Ncjn!2H*0c@F;j+IMcwf<g1()Ox=4wBY9pR!k=a)<Ma_u_?b*;&{gt3nhflVW{#2C62?C4I%bKv<W}QJ7XJ;CYe~EbKlJ8ifrvA_0OkRM2pAq5xm=^6sQhkd~&_~@*lJP|Nh_q`~LwF={Ju')))
_ROUTES={0:_PAYLOAD['base']}
for _rid,_patch in _PAYLOAD['patches'].items():
    _tape=list(_ROUTES[0])
    for _t,_a in _patch: _tape[_t]=_a
    _ROUTES[int(_rid)]=_tape
del _PAYLOAD
_SETTINGS={'hand_align': True, 'weed_repair': True, 'sell_lead': True, 'budget_guard': False, 'room_guard': False, 'clamp_sells': False, 'dead_stock': False, 'terminal_liquidation': False, 'front_run': False}

def _router(observation,step,state):
    if step>=144 and not state.get('day6'):
        shops=_get(_get(observation,'town',{}),'unlocked_shops',[]) or []
        state['route']={('BAKERY', 'YARN_STORE'): 3, ('BRUNCH_SPOT', 'YARN_STORE'): 4, ('FARMERS_MARKET', 'YARN_STORE'): 5, ('ICE_CREAM_SHOP', 'YARN_STORE'): 6, ('PET_CAFE', 'YARN_STORE'): 5, ('PIZZA_SHOP', 'YARN_STORE'): 7, ('SMOOTHIE_SHOP', 'YARN_STORE'): 8, ('YARN_STORE', 'BAKERY'): 9, ('YARN_STORE', 'BRUNCH_SPOT'): 9, ('YARN_STORE', 'FARMERS_MARKET'): 1, ('YARN_STORE', 'ICE_CREAM_SHOP'): 9, ('YARN_STORE', 'PET_CAFE'): 10, ('YARN_STORE', 'PIZZA_SHOP'): 6, ('YARN_STORE', 'SMOOTHIE_SHOP'): 11, ('YARN_STORE', 'YARN_STORE'): 12}.get(tuple(shops[:2]),0)
        state['day6']=True
    if step>=648 and not state.get('day27'):
        state['route']=2
        state['day27']=True
    return state.get('route',0)

_R42_OPENING=[['BUY_PRODUCT', 'WHEAT', 5], ['BUY_PRODUCT', 'WHEAT', 10], ['SELL', 'WHEAT', 60]]
for _r42_tape in _ROUTES.values():
    _r42_tape[0]=dict(_r42_tape[0],market=[list(o) for o in _R42_OPENING])
del _r42_tape
_IMPL=make_agent(_ROUTES,router=_router,**_SETTINGS)
_IMPL.chassis.diagnostics['terminal_rescue_errors']=0

def agent(observation,configuration=None):
    try:
        action=_IMPL(observation,configuration)
        pass
        return action
    except Exception:
        return {'farmer':['PASS'],'hands':[],'market':[]}

_SHOP_PARENT=agent
del agent

def agent(observation,configuration=None):
    action=_SHOP_PARENT(observation,configuration)
    try:
        if _step_of(observation)>=718:
            view=_View(observation,_int(_get(observation,'player',0)),_IMPL.chassis.cfg)
            units=[]
            for i,pos in enumerate(view.positions):
                units.append(['DROP'] if _shed_adjacent(pos,view.board) and view.inv(i) else ['PASS'])
            action={'farmer':units[0],'hands':units[1:],'market':[]}
            projected=_IMPL.chassis._projected_shed(action,view)
            action['market']=[['SELL',item,projected.get(item,0)] for item in PRODUCTS if projected.get(item,0)>0]
            action['market'].sort(key=lambda o:-view.prices.get(o[1],0)*o[2])
    except Exception:
        _IMPL.chassis.diagnostics['terminal_rescue_errors'] += 1
    return action

# EXP-154 modifications: Ahmed Berat Ozer; public capabilities credited below.
# Dmitrii Gluzdov Seven Turn Rescue and Kaggle engine contributors, Apache-2.0.

_UNIT_NS={"__name__":"v28_own_unit_model"}
exec('# SPDX-License-Identifier: Apache-2.0\n# Extracted Kaggle / kaggle-environments contributor code; see NOTICE.txt.\n"""Exact deterministic unit/decay semantics extracted from kaggle-environments 1.32.7.\nSource kaggriculture.py SHA256 bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e.\nNo interpreter, market RNG, policy controls, or replay content is included.\n"""\n\nENGINE_VERSION = "1.32.7"\nSOURCE_SHA256 = "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e"\n\nCROPS = {\n    "WHEAT":      {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "interval": 0, "max_yield": 6, "ongoing": False},\n    "CARROT":     {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "interval": 0, "max_yield": 4, "ongoing": False},\n    "TOMATO":     {"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "interval": 1, "max_yield": 4, "ongoing": True},\n    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},\n    "MELON":      {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},\n}\n\nANIMALS = {\n    "GOOSE": {"cost": 300, "structure": "COOP",    "first_yield_day": 4, "interval": 1, "max_held": 4, "product": "EGG"},\n    "COW":   {"cost": 400, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "max_held": 6, "product": "MILK"},\n    "SHEEP": {"cost": 500, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "max_held": 6, "product": "WOOL"},\n}\n\nPRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]\n\nFARMER_MOVES = {\n    "NORTH": (0, -1),\n    "SOUTH": (0, 1),\n    "EAST":  (1, 0),\n    "WEST":  (-1, 0),\n}\n\ndef _shed_access_tiles(board_size):\n    """Four inner-corner tiles around the shed, in NWSE order."""\n    half = board_size // 2\n    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]\n\ndef _is_shed_adjacent(pos, board_size):\n    return tuple(pos) in {(x, y) for (x, y) in _shed_access_tiles(board_size)}\n\ndef _new_plant(crop, day, turns_per_day):\n    cd = CROPS[crop]\n    return {\n        "kind": "PLANT",\n        "crop": crop,\n        "planted_day": day,\n        "watered_today": False,\n        "consecutive_unwatered": 1,  # planting day counts as unwatered\n        "yield_units": 0 if cd["ongoing"] else 1,\n        "max_lifespan_step": (-1 if cd["ongoing"] else (day + cd["max_yield_day"] + 1) * turns_per_day),\n        "fertilized_until_day": -1,\n    }\n\ndef _new_animal(animal, day):\n    a = ANIMALS[animal]\n    return {\n        "kind": a["structure"],\n        "animal": animal,\n        "placed_day": day,\n        "yield_units": 0,\n        "consecutive_unfed": 0,\n        "fed_today": False,\n        "cared_today": False,\n        "fertilizer_available": False,\n        "pending_care_bonus": 0,\n    }\n\ndef _farmer_position(farm, idx):\n    """idx 0 = main farmer, 1+ = hand index."""\n    if idx == 0:\n        return farm["farmer"]\n    return farm["hands"][idx - 1] if idx - 1 < len(farm["hands"]) else None\n\ndef _set_farmer_position(farm, idx, pos):\n    if idx == 0:\n        farm["farmer"] = list(pos)\n    else:\n        farm["hands"][idx - 1] = list(pos)\n\ndef _farmer_inventory(private, idx):\n    """Inventories list is [main_farmer, *hands]; grow it if idx is past the end."""\n    while len(private["inventories"]) <= idx:\n        private["inventories"].append({})\n    return private["inventories"][idx]\n\ndef _inv_add(inv, item, n=1):\n    inv[item] = inv.get(item, 0) + n\n\ndef _inv_take(inv, item, n=1):\n    if inv.get(item, 0) < n:\n        return False\n    inv[item] -= n\n    if inv[item] == 0:\n        del inv[item]\n    return True\n\ndef _apply_unit_action(farm, private, idx, action, board_size, day, turns_per_day, shed_capacity=100):\n    """Process one farmer/hand\'s action. Invalid / illegal actions are silent no-ops."""\n    if not isinstance(action, list) or not action:\n        return\n    op = action[0]\n    pos = _farmer_position(farm, idx)\n    if pos is None:\n        return\n    fx, fy = pos[0], pos[1]\n    inv = _farmer_inventory(private, idx)\n\n    if op in FARMER_MOVES:\n        dx, dy = FARMER_MOVES[op]\n        nx, ny = fx + dx, fy + dy\n        if not (0 <= nx < board_size and 0 <= ny < board_size):\n            return\n        # Movement onto LOCKED tiles is allowed: a hand can spawn on a locked\n        # shed-access tile, and blocking movement would strand it there forever.\n        # Tile operations (PLANT, WATER, etc.) still no-op on LOCKED tiles.\n        _set_farmer_position(farm, idx, (nx, ny))\n        return\n\n    if op == "PASS":\n        return\n\n    tile = farm["tiles"][fy][fx]\n\n    # Shed operations resolve before the LOCKED guard. They use the tile only as\n    # a standing position -- the shed itself is always owned -- and three of the\n    # four shed-access tiles start LOCKED, so guarding them first would make the\n    # shed unreachable from those tiles.\n    if op == "DROP":\n        if not _is_shed_adjacent((fx, fy), board_size):\n            return\n        shed = private["shed"]\n        for item, n in list(inv.items()):\n            if n <= 0:\n                del inv[item]\n                continue\n            room = max(0, shed_capacity - sum(shed.values()))\n            take = min(n, room)\n            if take > 0:\n                shed[item] = shed.get(item, 0) + take\n            del inv[item]\n        return\n\n    if op == "PICKUP":\n        if not _is_shed_adjacent((fx, fy), board_size):\n            return\n        if len(action) < 2:\n            return\n        item = action[1]\n        n = int(action[2]) if len(action) >= 3 else 1\n        if n <= 0:\n            return\n        # Seeds live in private["seeds"] and are consumed directly by PLANT;\n        # they never pass through farmer inventory or the shed.\n        available = private["shed"].get(item, 0)\n        n = min(n, available)\n        if n <= 0:\n            return\n        private["shed"][item] -= n\n        _inv_add(inv, item, n)\n        return\n\n    if op == "PLACE":\n        if len(action) < 2:\n            return\n        item = action[1]\n        # Animal placement: standing on a matching unoccupied structure. A LOCKED\n        # tile is the string "LOCKED", never a dict, so this branch cannot match\n        # there and PLACE falls through to the shed path below.\n        if (\n            item in ANIMALS\n            and isinstance(tile, dict)\n            and tile.get("kind") == ANIMALS[item]["structure"]\n            and "animal" not in tile\n        ):\n            if _inv_take(inv, item, 1):\n                farm["tiles"][fy][fx] = _new_animal(item, day)\n            return\n        # Shed drop: orthogonally adjacent to the shed; obeys shedCapacity.\n        if _is_shed_adjacent((fx, fy), board_size):\n            n = int(action[2]) if len(action) >= 3 else 1\n            if n <= 0:\n                return\n            n = min(n, inv.get(item, 0))\n            if n <= 0:\n                return\n            current = sum(private["shed"].values())\n            room = max(0, shed_capacity - current)\n            n = min(n, room)\n            if n <= 0:\n                return\n            inv[item] -= n\n            if inv[item] == 0:\n                del inv[item]\n            private["shed"][item] = private["shed"].get(item, 0) + n\n        return\n\n    # Everything below mutates the tile the unit stands on, so it requires that\n    # tile to be owned.\n    if tile == "LOCKED":\n        return\n\n    if op == "PLANT":\n        if len(action) < 2:\n            return\n        crop = action[1]\n        if crop not in CROPS:\n            return\n        if tile is not None:\n            return\n        if private["seeds"].get(crop, 0) <= 0:\n            return\n        private["seeds"][crop] -= 1\n        farm["tiles"][fy][fx] = _new_plant(crop, day, turns_per_day)\n        return\n\n    if op == "WATER":\n        if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):\n            return\n        if tile["watered_today"]:\n            return\n        tile["watered_today"] = True\n        crop_data = CROPS[tile["crop"]]\n        if not crop_data["ongoing"]:\n            age_days = day - tile["planted_day"]\n            window_start = (crop_data["max_yield_day"] + 1) // 2\n            if window_start <= age_days <= crop_data["max_yield_day"]:\n                bonus = 2 if tile["fertilized_until_day"] >= day else 1\n                tile["yield_units"] = min(crop_data["max_yield"], tile["yield_units"] + bonus)\n        return\n\n    if op == "HARVEST":\n        if not isinstance(tile, dict):\n            return\n        if tile.get("yield_units", 0) <= 0:\n            return\n        if tile.get("kind") == "PLANT":\n            crop_data = CROPS[tile["crop"]]\n            if day - tile["planted_day"] < crop_data["first_yield_day"]:\n                # Ongoing crops only accumulate yield_units after first_yield_day,\n                # so reaching here with yield_units > 0 indicates a bug.\n                if crop_data["ongoing"]:\n                    print(\n                        f"WARNING: HARVEST on immature ongoing {tile[\'crop\']} "\n                        f"(planted day {tile[\'planted_day\']}, current day {day}, "\n                        f"first_yield_day {crop_data[\'first_yield_day\']}, "\n                        f"yield_units {tile[\'yield_units\']}); should never happen"\n                    )\n                return\n            units = tile["yield_units"]\n            tile["yield_units"] = 0\n            _inv_add(inv, tile["crop"], units)\n            if not crop_data["ongoing"]:\n                farm["tiles"][fy][fx] = None\n        elif "animal" in tile:\n            units = tile["yield_units"]\n            tile["yield_units"] = 0\n            _inv_add(inv, ANIMALS[tile["animal"]]["product"], units)\n        return\n\n    if op == "FERTILIZE":\n        if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):\n            return\n        if not _inv_take(inv, "FERTILIZER", 1):\n            return\n        # Active for `day`, `day+1`, `day+2` (3 days inclusive).\n        tile["fertilized_until_day"] = max(tile.get("fertilized_until_day", -1), day + 2)\n        return\n\n    if op == "DIG":\n        if tile is None:\n            return\n        # Removes plants, weeds, empty coop/pasture. Does NOT remove a placed animal.\n        if isinstance(tile, dict) and "animal" in tile:\n            return\n        farm["tiles"][fy][fx] = None\n        return\n\n    if op == "BUILD_COOP":\n        if tile is not None:\n            return\n        farm["tiles"][fy][fx] = {"kind": "COOP"}\n        return\n\n    if op == "BUILD_PASTURE":\n        if tile is not None:\n            return\n        farm["tiles"][fy][fx] = {"kind": "PASTURE"}\n        return\n\n    if op == "FEED":\n        if not (isinstance(tile, dict) and "animal" in tile):\n            return\n        if tile["fed_today"]:\n            return\n        if not _inv_take(inv, "WHEAT", 1):\n            return\n        tile["fed_today"] = True\n        return\n\n    if op == "COLLECT_FERTILIZER":\n        if not (isinstance(tile, dict) and "animal" in tile):\n            return\n        if not tile["fertilizer_available"]:\n            return\n        tile["fertilizer_available"] = False\n        _inv_add(inv, "FERTILIZER", 1)\n        return\n\n    if op == "CARE":\n        if not (isinstance(tile, dict) and "animal" in tile):\n            return\n        if tile["cared_today"]:\n            return\n        tile["cared_today"] = True\n        return\n\ndef _decay_plants(farm, step):\n    board_size = len(farm["tiles"])\n    for y in range(board_size):\n        for x in range(board_size):\n            tile = farm["tiles"][y][x]\n            if not isinstance(tile, dict) or tile.get("kind") != "PLANT":\n                continue\n            mls = tile["max_lifespan_step"]\n            if mls < 0 or step < mls:\n                continue\n            if (step - mls) % 2 != 0:\n                continue\n            tile["yield_units"] -= 1\n            if tile["yield_units"] <= 0:\n                farm["tiles"][y][x] = {"kind": "WEED"}\n\n',_UNIT_NS)
_PLANNER_NS=dict(_UNIT_NS)
exec('"""E182 modification: Shop0909 last-seven-turn physical closure planner.\n\nNo engine imports, policy tapes, replay fixtures, RNG or remote calls.\nThe only supported market continuation is SELL; unknown execution abstains.\n"""\nfrom copy import deepcopy\nfrom time import perf_counter\nSTART, FINAL = (712, 718)\nOPS = set(FARMER_MOVES) | {\'PASS\', \'DROP\', \'PICKUP\', \'PLACE\', \'PLANT\', \'WATER\', \'HARVEST\', \'FERTILIZE\', \'DIG\', \'BUILD_COOP\', \'BUILD_PASTURE\', \'FEED\', \'CARE\', \'COLLECT_FERTILIZER\'}\nITEMS = tuple(PRODUCTS) + tuple(ANIMALS)\n\nclass Unsupported(ValueError):\n    pass\n\ndef _get(obj, key, default=None):\n    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)\n\ndef _settings(config):\n    size, turns, last = (_get(config, k, d) for k, d in [(\'boardSize\', 10), (\'turnsPerDay\', 24), (\'episodeSteps\', 720)])\n    if (size, turns, last) != (10, 24, 720):\n        raise Unsupported(\'requires pinned 10x10/24/720 terminal window\')\n    cap = int(_get(config, \'shedCapacity\', 100))\n    orders = min(10, int(_get(config, \'maxMarketOrdersPerTurn\', 10)))\n    if cap < 1 or orders < 1:\n        raise Unsupported(\'invalid capacity/order limit\')\n    return (size, turns, cap, orders)\n\ndef physical_state(obs):\n    """Comparable own physical state; market prices and bank are intentionally excluded."""\n    seat = int(_get(obs, \'player\', 0))\n    farm = _get(obs, \'farms\')[seat]\n    return ({k: v for k, v in farm.items() if k != \'money\'}, _get(obs, \'private\'))\n\ndef _commands(action, n):\n    return [action.get(\'farmer\', [\'PASS\']), *action.get(\'hands\', [])][:n] + [[\'PASS\'] for _ in range(max(0, n - 1 - len(action.get(\'hands\', []))))]\n\ndef _clone_state(farm, private):\n    f = dict(farm)\n    f[\'tiles\'] = [[dict(tile) if isinstance(tile, dict) else tile for tile in row] for row in farm[\'tiles\']]\n    f[\'farmer\'] = list(farm[\'farmer\'])\n    f[\'hands\'] = [list(pos) for pos in farm[\'hands\']]\n    f[\'unlocked_quadrants\'] = list(farm[\'unlocked_quadrants\'])\n    pr = dict(private)\n    pr[\'shed\'], pr[\'seeds\'] = (dict(private[\'shed\']), dict(private[\'seeds\']))\n    pr[\'inventories\'] = [dict(inv) for inv in private[\'inventories\']]\n    return (f, pr)\n\ndef _clone_schedule(schedule):\n    result = []\n    for action in schedule:\n        value = dict(action)\n        if \'farmer\' in action:\n            value[\'farmer\'] = list(action[\'farmer\'])\n        for key in [\'hands\', \'market\']:\n            if key in action:\n                value[key] = [list(command) for command in action[key]]\n        result.append(value)\n    return result\n\ndef _validate(schedule, n, orders):\n    for action in schedule:\n        if not isinstance(action, dict) or set(action) - {\'farmer\', \'hands\', \'market\'}:\n            raise Unsupported(\'unknown action shape\')\n        if not isinstance(action.get(\'hands\', []), list):\n            raise Unsupported(\'hands must be a list\')\n        for command in [action.get(\'farmer\', [\'PASS\']), *action.get(\'hands\', [])]:\n            if not isinstance(command, list) or not command or command[0] not in OPS:\n                raise Unsupported(\'unknown/malformed unit operation\')\n            if command[0] in {\'PICKUP\', \'PLACE\', \'PLANT\'}:\n                if len(command) < 2 or command[1] not in ITEMS:\n                    raise Unsupported(\'unknown unit item\')\n                if len(command) > 2 and (not isinstance(command[2], int)):\n                    raise Unsupported(\'noninteger unit quantity\')\n        market = action.get(\'market\', [])\n        if not isinstance(market, list) or len(market) > orders:\n            raise Unsupported(\'market order shape/cap\')\n        for order in market:\n            if not isinstance(order, list) or len(order) != 3 or order[0] != \'SELL\' or (order[1] not in PRODUCTS) or (not isinstance(order[2], int)) or (order[2] <= 0):\n                raise Unsupported(\'baseline market must contain positive integer SELL only\')\n\ndef liquidation(shed, inherited_market, max_orders=10):\n    """Use actual post-unit stock; retain first parent item ordering, then stable product order."""\n    items = []\n    for order in inherited_market:\n        if order[1] not in items:\n            items.append(order[1])\n    items += [item for item in PRODUCTS if item not in items]\n    orders = [[\'SELL\', item, int(shed.get(item, 0))] for item in items if shed.get(item, 0) > 0]\n    if len(orders) > min(10, max_orders):\n        raise Unsupported(\'actual final stock exceeds order slots\')\n    return orders\n\ndef shop_liquidation(farm, private, prices):\n    """Exact original final worker/drop and market rule, with current stock/prices."""\n    return liquidate(FarmView({\'player\': 0, \'farms\': [farm], \'private\': private, \'market\': {\'prices\': prices}}))\n\ndef simulate(obs, config, schedule, *, final_liquidate=False, detailed=False, preserve_final_commands=False):\n    """Exact own unit/decay and SELL-stock transitions. No claim to simulate shared prices."""\n    size, turns, cap, order_cap = _settings(config)\n    step = int(_get(obs, \'step\', -1))\n    if step < START or step + len(schedule) - 1 > FINAL or (not schedule):\n        raise Unsupported(\'outside 712..718; no day boundary or terminal auto-drop\')\n    if any(((t + 1) % turns == 0 for t in range(step, step + len(schedule)))):\n        raise Unsupported(\'day boundary\')\n    farm0, private0 = physical_state(obs)\n    farm, private = _clone_state(farm0, private0)\n    n = 1 + len(farm[\'hands\'])\n    if len(private[\'inventories\']) != n or n > 32:\n        raise Unsupported(\'invalid/unbounded worker inventory shape\')\n    _validate(schedule, n, order_cap)\n    deposited = [dict() for _ in range(n)]\n    sold = {}\n    snapshots, rows, events = ([], [], [])\n    executed = _clone_schedule(schedule)\n    overflow = 0\n    for offset, action in enumerate(executed):\n        t = step + offset\n        if detailed:\n            snapshots.append(_clone_state(farm, private))\n        if t == FINAL and (not preserve_final_commands):\n            action = shop_liquidation(farm, private, _get(obs, \'market\')[\'prices\'])\n            executed[offset] = action\n        all_commands = [action.get(\'farmer\', [\'PASS\']), *action.get(\'hands\', [])]\n        demand = {}\n        for command in all_commands:\n            if command[0] == \'PLANT\':\n                demand[command[1]] = demand.get(command[1], 0) + 1\n        blocked = {item for item, count in demand.items() if count > private[\'seeds\'].get(item, 0)}\n        for actor, command in enumerate(_commands(action, n)):\n            if command[0] == \'PLANT\' and command[1] in blocked:\n                command = [\'PASS\']\n            pos = farm[\'farmer\'] if actor == 0 else farm[\'hands\'][actor - 1]\n            xy = tuple(pos)\n            inv = private[\'inventories\'][actor]\n            before_inv = dict(inv) if command[0] in {\'DROP\', \'HARVEST\', \'COLLECT_FERTILIZER\'} else None\n            before_shed = dict(private[\'shed\']) if command[0] in {\'DROP\', \'PLACE\'} else None\n            _apply_unit_action(farm, private, actor, command, size, t // turns, turns, cap)\n            if before_shed is not None:\n                delta = {item: amount - before_shed.get(item, 0) for item, amount in private[\'shed\'].items() if amount > before_shed.get(item, 0)}\n                for item, amount in delta.items():\n                    deposited[actor][item] = deposited[actor].get(item, 0) + amount\n                if delta:\n                    events.append({\'offset\': offset, \'actor\': actor, \'op\': command[0], \'xy\': xy, \'deposited\': delta})\n                if command[0] == \'DROP\':\n                    overflow += sum((max(0, amount - inv.get(item, 0) - delta.get(item, 0)) for item, amount in before_inv.items()))\n            if command[0] in {\'HARVEST\', \'COLLECT_FERTILIZER\'}:\n                delta = {item: amount - before_inv.get(item, 0) for item, amount in inv.items() if amount > before_inv.get(item, 0)}\n                if delta:\n                    events.append({\'offset\': offset, \'actor\': actor, \'op\': command[0], \'xy\': xy, \'acquired\': delta})\n        pre_market = dict(private[\'shed\'])\n        if t == FINAL and preserve_final_commands and final_liquidate:\n            action[\'market\'] = liquidation(pre_market, [], order_cap)\n            prices = _get(obs, \'market\')[\'prices\']\n            action[\'market\'].sort(key=lambda order: -int(prices.get(order[1], 0)) * order[2])\n        for _, item, requested in action.get(\'market\', []):\n            quantity = min(requested, private[\'shed\'].get(item, 0), 99999)\n            if quantity > 0:\n                private[\'shed\'][item] -= quantity\n                sold[item] = sold.get(item, 0) + quantity\n        _decay_plants(farm, t)\n        rows.append({\'pre_market_shed\': pre_market, \'post_market_shed\': dict(private[\'shed\']), \'deposited_by_actor\': [dict(v) for v in deposited], \'sold\': dict(sold)})\n    if detailed:\n        snapshots.append(_clone_state(farm, private))\n    return {\'rows\': rows, \'states\': snapshots, \'events\': events, \'actions\': executed, \'overflow_units\': overflow, \'farm\': farm, \'private\': private, \'sold\': sold}\n\ndef _ge(left, right):\n    return all((left.get(item, 0) >= value for item, value in right.items()))\n\ndef dominates(candidate, baseline):\n    """Preserve every baseline worker\'s actual deposit prefixes and shed availability."""\n    if candidate[\'overflow_units\']:\n        return False\n    for new, old in zip(candidate[\'rows\'], baseline[\'rows\']):\n        if not _ge(new[\'pre_market_shed\'], old[\'pre_market_shed\']):\n            return False\n        if not _ge(new[\'sold\'], old[\'sold\']):\n            return False\n        if any((not _ge(a, b) for a, b in zip(new[\'deposited_by_actor\'], old[\'deposited_by_actor\']))):\n            return False\n    return True\n\ndef _value(run, prices):\n    shed = run[\'private\'][\'shed\']\n    return sum(((run[\'sold\'].get(item, 0) + shed.get(item, 0)) * prices[item] for item in PRODUCTS))\n\ndef _walk(start, end):\n    x, y = start\n    tx, ty = end\n    return [[\'EAST\']] * max(0, tx - x) + [[\'WEST\']] * max(0, x - tx) + [[\'SOUTH\']] * max(0, ty - y) + [[\'NORTH\']] * max(0, y - ty)\n\ndef _return(pos):\n    targets = _shed_access_tiles(10)\n    target = min(targets, key=lambda xy: (abs(pos[0] - xy[0]) + abs(pos[1] - xy[1]), targets.index(xy)))\n    return _walk(pos, target) + [[\'DROP\']]\n\ndef _proposals(run, actor, prices, max_per_actor):\n    """One/two resource bundles plus direct carry closure, replacing a baseline suffix."""\n    owners = {}\n    for event in run[\'events\']:\n        if \'acquired\' in event:\n            owners.setdefault((tuple(event[\'xy\']), event[\'op\']), set()).add(event[\'actor\'])\n    proposals = []\n    seen = set()\n    horizon = len(run[\'rows\'])\n    for offset in range(horizon):\n        farm, private = run[\'states\'][offset]\n        pos = tuple(farm[\'farmer\'] if actor == 0 else farm[\'hands\'][actor - 1])\n        inventory = private[\'inventories\'][actor]\n        carried = sum((prices.get(item, 0) * count for item, count in inventory.items()))\n        prefix_deposits = run[\'rows\'][offset - 1][\'deposited_by_actor\'][actor] if offset else {}\n        future_deposits = run[\'rows\'][-1][\'deposited_by_actor\'][actor]\n        obligation = sum((prices.get(item, 0) * (count - prefix_deposits.get(item, 0)) for item, count in future_deposits.items()))\n        bundles = []\n        for y, row in enumerate(farm[\'tiles\']):\n            for x, tile in enumerate(row):\n                if not isinstance(tile, dict):\n                    continue\n                xy, operations, value = ((x, y), [], 0)\n                if tile.get(\'yield_units\', 0) > 0:\n                    item = tile.get(\'crop\') if tile.get(\'kind\') == \'PLANT\' else ANIMALS.get(tile.get(\'animal\'), {}).get(\'product\')\n                    mature = item and (\'animal\' in tile or (START + offset) // 24 - tile[\'planted_day\'] >= CROPS[item][\'first_yield_day\'])\n                    if mature and (not owners.get((xy, \'HARVEST\'), set()) - {actor}):\n                        operations.append([\'HARVEST\'])\n                        value += prices[item] * tile[\'yield_units\']\n                if tile.get(\'fertilizer_available\') and \'animal\' in tile and (not owners.get((xy, \'COLLECT_FERTILIZER\'), set()) - {actor}):\n                    operations.append([\'COLLECT_FERTILIZER\'])\n                    value += prices[\'FERTILIZER\']\n                if operations:\n                    distance = len(_walk(pos, xy)) + len(operations) + len(_return(xy))\n                    if distance <= horizon - offset:\n                        bundles.append((xy, operations, value, distance))\n        bundles.sort(key=lambda b: (-b[2] / b[3], -b[2], b[0]))\n        variants = [([], carried)] if carried else []\n        for xy, ops, value, _ in bundles[:6]:\n            variants.append(([(xy, ops)], carried + value))\n        for first in bundles[:3]:\n            for second in bundles[:3]:\n                if first[0] != second[0]:\n                    variants.append(([(first[0], first[1]), (second[0], second[1])], carried + first[2] + second[2]))\n        for stops, value in variants:\n            route, cursor = ([], pos)\n            for xy, ops in stops:\n                route += _walk(cursor, xy) + ops\n                cursor = xy\n            route += _return(cursor)\n            if len(route) > horizon - offset:\n                continue\n            route += [[\'PASS\']] * (horizon - offset - len(route))\n            key = (offset, tuple((tuple(c) for c in route)))\n            if key not in seen:\n                seen.add(key)\n                proposals.append((value - obligation, offset, route, len(stops)))\n    proposals.sort(key=lambda p: (-p[0], p[1], p[2]))\n    direct = [p for p in proposals if p[3] == 0 and p[0] > 0][:2]\n    chosen = direct + [p for p in proposals if p not in direct]\n    return chosen[:max_per_actor]\n\ndef plan_terminal(obs, config, baseline_remaining, *, max_simulations=64, passes=1, proposals_per_actor=4):\n    """At 712 accept seven actions; positive physical delivery is mandatory."""\n    begun = perf_counter()\n    fallback = {\'accepted\': False, \'reason\': \'\', \'actions\': None, \'simulations\': 0}\n    try:\n        if int(_get(obs, \'step\', -1)) != START or len(baseline_remaining) != FINAL - START + 1:\n            raise Unsupported(\'planning requires step 712 and exactly seven actions through 718\')\n        max_simulations = min(256, max(1, int(max_simulations)))\n        passes = min(2, max(1, int(passes)))\n        proposals_per_actor = min(16, max(1, int(proposals_per_actor)))\n        baseline = simulate(obs, config, baseline_remaining, detailed=True)\n        prices = {item: max(1, float(_get(obs, \'market\', {}).get(\'prices\', {}).get(item, 1))) for item in PRODUCTS}\n        current, best = (_clone_schedule(baseline_remaining), baseline)\n        baseline_value = best_value = _value(baseline, prices)\n        changes, simulations = ([], 0)\n        n = len(baseline[\'private\'][\'inventories\'])\n        for sweep in range(passes):\n            improved = False\n            for actor in range(n):\n                winner = None\n                for _, offset, route, bundle_count in _proposals(best, actor, prices, proposals_per_actor):\n                    if simulations >= max_simulations:\n                        break\n                    trial = _clone_schedule(current)\n                    for i, command in enumerate(route, offset):\n                        if actor == 0:\n                            trial[i][\'farmer\'] = command\n                        else:\n                            trial[i].setdefault(\'hands\', [])\n                            while len(trial[i][\'hands\']) < n - 1:\n                                trial[i][\'hands\'].append([\'PASS\'])\n                            trial[i][\'hands\'][actor - 1] = command\n                    evaluated = simulate(obs, config, trial)\n                    simulations += 1\n                    score = _value(evaluated, prices)\n                    if score > best_value and dominates(evaluated, baseline):\n                        required = {(tuple(e[\'xy\']), e[\'op\'], e[\'actor\']): e[\'acquired\'] for e in best[\'events\'] if \'acquired\' in e and e[\'actor\'] != actor}\n                        acquired = {}\n                        for e in evaluated[\'events\']:\n                            if \'acquired\' in e:\n                                key = (tuple(e[\'xy\']), e[\'op\'], e[\'actor\'])\n                                dst = acquired.setdefault(key, {})\n                                for item, amount in e[\'acquired\'].items():\n                                    dst[item] = dst.get(item, 0) + amount\n                        if all((_ge(acquired.get(k, {}), v) for k, v in required.items())):\n                            winner, best_value = ((trial, offset, bundle_count), score)\n                if winner:\n                    current, offset, bundle_count = winner\n                    best = simulate(obs, config, current, detailed=True)\n                    changes.append({\'pass\': sweep, \'actor\': actor, \'from_step\': START + offset, \'resource_bundles\': bundle_count, \'estimated_stock_value\': best_value})\n                    improved = True\n                if simulations >= max_simulations:\n                    break\n            if not improved or simulations >= max_simulations:\n                break\n        if not changes or best_value <= baseline_value:\n            return {**fallback, \'reason\': \'no positive physical delivery gain\', \'simulations\': simulations, \'changed_workers\': [], \'changes\': [], \'certificate\': {\'stock_value_gain_at_initial_prices\': 0, \'sold_unit_delta\': dict.fromkeys(PRODUCTS, 0)}, \'planning_ms\': (perf_counter() - begun) * 1000}\n        final = simulate(obs, config, current, final_liquidate=True, detailed=True)\n        physical = simulate(obs, config, current)\n        if not dominates(physical, baseline):\n            raise Unsupported(\'no zero-overflow dominating continuation\')\n        delta = {item: final[\'sold\'].get(item, 0) - baseline[\'sold\'].get(item, 0) for item in PRODUCTS}\n        deposited_gain = any((final[\'rows\'][-1][\'deposited_by_actor\'][actor].get(item, 0) > baseline[\'rows\'][-1][\'deposited_by_actor\'][actor].get(item, 0) for actor in range(n) for item in PRODUCTS))\n        worker_change = any((_commands(new, n) != _commands(old, n) for new, old in zip(final[\'actions\'], baseline[\'actions\'])))\n        accepted = worker_change and deposited_gain and any((v > 0 for v in delta.values())) and all((v >= 0 for v in delta.values()))\n        plan = {\'accepted\': accepted, \'reason\': \'joint physical dominance\' if accepted else \'no improvement\', \'baseline\': _clone_schedule(baseline_remaining), \'actions\': final[\'actions\'], \'expected_states\': final[\'states\'][:-1], \'simulations\': simulations, \'changes\': changes, \'abandoned\': False, \'changed_workers\': sorted({c[\'actor\'] for c in changes}), \'certificate\': {\'baseline_rows\': baseline[\'rows\'], \'physical_rows\': physical[\'rows\'], \'baseline_overflow\': baseline[\'overflow_units\'], \'candidate_overflow\': final[\'overflow_units\'], \'sold_unit_delta\': delta, \'stock_value_gain_at_initial_prices\': best_value - baseline_value, \'baseline_final_shed\': baseline[\'private\'][\'shed\'], \'final_shed\': final[\'private\'][\'shed\'], \'positive_physical_deposit_gain\': deposited_gain, \'markets_712_717_unchanged\': all((final[\'actions\'][i].get(\'market\', []) == baseline_remaining[i].get(\'market\', []) for i in range(FINAL - START)))}}\n    except (Unsupported, KeyError, TypeError, ValueError, IndexError) as exc:\n        plan = {**fallback, \'reason\': str(exc)}\n    plan[\'planning_ms\'] = (perf_counter() - begun) * 1000\n    return plan\n\ndef _effective_action(action, n):\n    return (_commands(action, n), action.get(\'market\', []))\n\ndef _recover_observed(obs, config, parent_action, plan):\n    """Bounded cargo salvage after deviation; never resume old positional commands."""\n    farm, private = physical_state(obs)\n    positions = [farm[\'farmer\'], *farm[\'hands\']]\n    remaining = FINAL - int(_get(obs, \'step\')) + 1\n    room = max(0, int(_get(config, \'shedCapacity\', 100)) - sum(private[\'shed\'].values()))\n    commands = []\n    problems = []\n    prices = _get(obs, \'market\', {}).get(\'prices\', {})\n    for actor, (pos, inv) in enumerate(zip(positions, private[\'inventories\'])):\n        command = [\'PASS\']\n        if any((v > 0 for v in inv.values())):\n            route = _return(pos)\n            if len(route) > remaining:\n                problems.append({\'actor\': actor, \'reason\': \'unreachable cargo\'})\n            elif len(route) > 1:\n                command = route[0]\n            elif sum((max(0, q) for q in inv.values())) <= room:\n                command = [\'DROP\']\n                room -= sum((max(0, q) for q in inv.values()))\n            else:\n                items = [item for item in PRODUCTS if inv.get(item, 0) > 0]\n                if room and items:\n                    item = max(items, key=lambda i: (prices.get(i, 1) * min(inv[i], room), -PRODUCTS.index(i)))\n                    quantity = min(inv[item], room)\n                    command = [\'PLACE\', item, quantity]\n                    room -= quantity\n                else:\n                    problems.append({\'actor\': actor, \'reason\': \'no shed capacity\'})\n        commands.append(command)\n    action = {\'farmer\': commands[0], \'hands\': commands[1:], \'market\': deepcopy(parent_action.get(\'market\', []))}\n    if int(_get(obs, \'step\')) == FINAL:\n        action[\'market\'] = []\n        action = simulate(obs, config, [action], final_liquidate=True, preserve_final_commands=True)[\'actions\'][0]\n    plan[\'recovery_steps\'] = plan.get(\'recovery_steps\', 0) + 1\n    if problems:\n        plan.setdefault(\'recovery_failures\', []).append({\'step\': int(_get(obs, \'step\')), \'problems\': problems})\n    return action\n\ndef terminal_action(obs, config, parent_action, plan):\n    """Canonical guard, pre-deviation abstention, observed recovery after deviation."""\n    step = int(_get(obs, \'step\', -1))\n    if not plan or not plan.get(\'accepted\') or (not START <= step <= FINAL):\n        return parent_action\n    if plan.get(\'abandoned\'):\n        return _recover_observed(obs, config, parent_action, plan) if plan.get(\'deviated\') else parent_action\n    index = step - START\n    n = 1 + len(physical_state(obs)[0][\'hands\'])\n    mismatch = physical_state(obs) != plan[\'expected_states\'][index] or _effective_action(parent_action, n) != _effective_action(plan[\'baseline\'][index], n)\n    if mismatch:\n        plan[\'abandoned\'] = True\n        plan[\'abandon_step\'] = step\n        plan[\'reason\'] = \'physical observation or effective baseline action diverged\'\n        plan[\'safety_failure\'] = True\n        return _recover_observed(obs, config, parent_action, plan) if plan.get(\'deviated\') else parent_action\n    result = deepcopy(plan[\'actions\'][index])\n    if step == FINAL:\n        farm, private = physical_state(obs)\n        result = shop_liquidation(farm, private, _get(obs, \'market\')[\'prices\'])\n    if _commands(result, n) != _commands(parent_action, n):\n        plan[\'deviated\'] = True\n    return result',_PLANNER_NS)
# EXP-154 integration by Ahmed Berat Ozer, derived from Dmitrii Gluzdov E182.
# The preserved v27 parent is simulated on a private shadow only at step 712.
_PRE_TERMINAL_AGENT=agent
del agent
_TERMINAL_PLANS={}
_TERMINAL_PREVIOUS={}
_UPGRADE_STATS={'planning_calls':0,'accepted':0,'changed_steps':0,'aborted':0,'shadow_declines':0,'errors':0,'max_planning_ms':0.0}

def _parent_liquidate(farm, private, prices):
    # Exactly v27's final projected DROP ordering, in the planner's private state.
    view=_View({'player':0,'farms':[farm],'private':private,'market':{'prices':prices}},0,_IMPL.chassis.cfg)
    commands=[['DROP'] if _shed_adjacent(pos,view.board) and view.inv(i) else ['PASS'] for i,pos in enumerate(view.positions)]
    action={'farmer':commands[0],'hands':commands[1:],'market':[]}
    stock=_IMPL.chassis._projected_shed(action,view)
    action['market']=[['SELL',item,stock.get(item,0)] for item in PRODUCTS if stock.get(item,0)>0]
    action['market'].sort(key=lambda o:-view.prices.get(o[1],0)*o[2])
    return action

_PLANNER_NS['shop_liquidation']=_parent_liquidate

def _shadow_terminal(obs,config):
    seat=int(obs['player']);chassis=_IMPL.chassis
    state=chassis.players.get(seat)
    if not state or state.get('last_step')!=711 or state.get('route')!=2:
        return None
    # No delayed weed/structure intervention may depend on an unmodeled future.
    if state.get('pending'):
        return None
    shadow=copy.copy(chassis);shadow.players=copy.deepcopy(chassis.players)
    shadow.diagnostics={k:0 for k in chassis.diagnostics}
    projected=copy.deepcopy(obs);baseline=[];states=[]
    for step in range(712,719):
        projected['step']=step;projected['day']=step//24;projected['hour']=step%24
        states.append(copy.deepcopy(shadow.players[seat]))
        action=shadow.act(projected,config)
        if step==718:
            action=_parent_liquidate(projected['farms'][seat],projected['private'],projected['market']['prices'])
        else:
            market=action.get('market',[])
            if len(market)!=9 or {o[1] for o in market}!=set(PRODUCTS) or any(o[0]!='SELL' or len(o)!=3 or type(o[2]) is not int or o[2]<100 for o in market):
                return None
        if any(shadow.diagnostics.values()):return None
        run=_PLANNER_NS['simulate'](projected,config,[action])
        if run['actions'][0]!=action:return None
        baseline.append(action)
        run['farm']['money']=projected['farms'][seat]['money']
        projected['farms'][seat]=run['farm'];projected['private']=run['private']
    return baseline,states

def agent(observation,configuration=None):
    try:
        step=int(observation['step']);seat=int(observation['player'])
    except Exception:
        return _PRE_TERMINAL_AGENT(observation,configuration)
    previous=_TERMINAL_PREVIOUS.get(seat)
    if step==0 or (previous is not None and step<=previous):_TERMINAL_PLANS.pop(seat,None)
    _TERMINAL_PREVIOUS[seat]=step
    plan=_TERMINAL_PLANS.get(seat)
    if plan and plan.get('accepted') and 712<=step<=718:
        if previous!=step-1:plan.update(abandoned=True,reason='nonconsecutive callback')
        try:
            result=_PLANNER_NS['terminal_action'](observation,configuration,plan['baseline'][step-712],plan)
            if plan.get('abandoned'):
                if not plan.get('abort_counted'):
                    plan['abort_counted']=True;_UPGRADE_STATS['aborted']+=1
                if not plan.get('deviated'):
                    _IMPL.chassis.players[seat]=copy.deepcopy(plan['parent_states_before'][step-712])
                    _TERMINAL_PLANS.pop(seat,None)
                    return _PRE_TERMINAL_AGENT(observation,configuration)
            _UPGRADE_STATS['changed_steps']+=int(result!=plan['baseline'][step-712])
            return result
        except Exception:
            _UPGRADE_STATS['errors']+=1
            if plan.get('deviated'):
                try:return _PLANNER_NS['_recover_observed'](observation,configuration,plan['baseline'][step-712],plan)
                except Exception:return _parent_liquidate(observation['farms'][seat],observation['private'],observation['market']['prices'])
    if step!=712:return _PRE_TERMINAL_AGENT(observation,configuration)
    _UPGRADE_STATS['planning_calls']+=1
    try:shadow=_shadow_terminal(observation,configuration)
    except (ValueError,KeyError,TypeError,IndexError):shadow=None
    if shadow is None:
        _UPGRADE_STATS['shadow_declines']+=1
        return _PRE_TERMINAL_AGENT(observation,configuration)
    baseline,states=shadow
    actual=_PRE_TERMINAL_AGENT(observation,configuration)
    if actual!=baseline[0]:
        _UPGRADE_STATS['shadow_declines']+=1;return actual
    try:
        plan=_PLANNER_NS['plan_terminal'](observation,configuration,baseline,max_simulations=64,passes=1,proposals_per_actor=4)
        _UPGRADE_STATS['max_planning_ms']=max(_UPGRADE_STATS['max_planning_ms'],plan.get('planning_ms',0.0))
        if not plan.get('accepted'):return actual
        plan['parent_states_before']=states;_TERMINAL_PLANS[seat]=plan
        _UPGRADE_STATS['accepted']+=1
        result=_PLANNER_NS['terminal_action'](observation,configuration,actual,plan)
        _UPGRADE_STATS['changed_steps']+=int(result!=actual)
        return result
    except Exception:
        _UPGRADE_STATS['errors']+=1;return actual

agent.telemetry=_UPGRADE_STATS

# EXP-154: aurax7 Reactive v2 day-end storage guard, adapted to our v27 view.
_PRE_ROOM_AGENT=agent
del agent
_ROOM_STATS={'changed_turns':0,'added_units':0,'errors':0}
def agent(observation,configuration=None):
    action=_PRE_ROOM_AGENT(observation,configuration)
    try:
        step=_step_of(observation)
        if step%24!=23:return action
        view=_View(observation,_int(_get(observation,'player',0)),_IMPL.chassis.cfg)
        carried=sum(max(0,int(n)) for inv in view.invs for n in inv.values())
        needed=sum(view.shed.values())+carried-99
        if needed<=0:return action
        planned={}
        for o in action.get('market',[]):
            if o and o[0]=='SELL' and len(o)>=3:planned[o[1]]=planned.get(o[1],0)+max(0,int(o[2]))
        result=copy.deepcopy(action);added=0
        for item in sorted(PRODUCTS,key=lambda it:-int(view.prices.get(it,0))):
            qty=min(needed,max(0,view.shed.get(item,0)-planned.get(item,0)))
            if qty<=0:continue
            if len(result['market'])>=10:break
            result['market'].append(['SELL',item,qty]);needed-=qty;added+=qty
            if needed<=0:break
        if added:_ROOM_STATS['changed_turns']+=1;_ROOM_STATS['added_units']+=added
        return result
    except Exception:
        _ROOM_STATS['errors']+=1;return action

agent.telemetry=_ROOM_STATS

# Incorporated upstream attribution and change notice:
# E182 Shop0909 + terminal physical closure (modified 2026-09-09)
# 
# The active public parent is Yusuke Hayashi's yhay81/shop-router-0909 v3.
# router_parent.py and actions.json are exact original bytes, not newly authored
# routes. The parent credits aurax7's Reactive Router for sale timing and shed
# projection; that attribution remains in router_parent.py. Original payload
# LICENSE.txt is preserved unchanged (Apache License 2.0 text); it contains no
# named copyright grantor and no separate NOTICE was supplied. No additional
# ownership, endorsement, or upstream replay-data rights claim is made.
# 
# Local changes: separate main.py/policy.py adapter; bounded start712 planner
# copied from frozen E180/S78 and modified for seven callbacks, exact Shop final
# liquidation, strict positive physical delivery/sale gain, and observation guards.
# unit_model.py is an unchanged frozen E180 copy of Kaggle's extracted semantics.
# The following original E180 notice is retained verbatim for attribution history.
# Its references to Thomas files describe E180, not files supplied in this Shop
# package: no Thomas tapes, trees or policy are included here.
# 
# ----- Original E180 notice -----
# Kaggriculture: Last-Mile Harvest Planner
# Attribution and change notice
# 
# Thomas Tschinkel is the author of the parent public state-router policy and its
# published decision trees and action-route data. Source: Kaggriculture: 93.8% Win
# Rate Public State Router, notebook version 3, scriptVersionId 347936183:
# https://www.kaggle.com/code/thomastschinkel/kaggriculture-93-8-win-rate-public-state-router?scriptVersionId=347936183
# The public notebook identifies its license as Apache License, Version 2.0.
# Original published main.py SHA-256:
# b87a27ed614a33329be85f1b662e51cf4078a019fee937afcebbbbf2f51f8522
# 
# Changes to that source for this distribution: compressed route/tree literals
# were decoded into readable tapes.json and trees.json; a read-only planned_action
# helper was added; descriptive headers and local data loading were adapted.
# The original parent feature extraction, tree traversal and agent behavior are
# retained. These public routes are not claimed as newly authored or trained by
# the notebook distributor.
# 
# unit_model.py contains deterministic unit-action and crop-decay definitions
# extracted from Kaggle's kaggle-environments 1.32.7 Kaggriculture engine, licensed
# under Apache License, Version 2.0. Credit: Kaggle and the kaggle-environments
# contributors. Project: https://github.com/Kaggle/kaggle-environments
# Source file: kaggle_environments/envs/kaggriculture/kaggriculture.py
# Source SHA-256:
# bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e
# The extracted unit/decay definitions are not a newly authored game engine;
# market price dynamics and the full interpreter are not part of this module.
# 
# Additional work in this distribution: a bounded last-nine-action collection
# and delivery planner, observation guards and recovery, a settings-consuming
# factory and entry point, standalone examples, and deterministic packaging.
# The full Apache License, Version 2.0 is included as LICENSE.txt.
# No endorsement by Thomas Tschinkel or Kaggle is implied.
# 
# Data provenance limitation: Thomas's source refers to public replay data and
# an upstream provenance.json. That original episode-level manifest, replay IDs
# and individual replay-author identities were not supplied with the public
# notebook/output used here. No names or episode lineage have been invented.
# Notebook-level licensing does not independently establish the missing underlying
# replay-data rights chain. The package supplies usable readable routes, not a
# reproducible reconstruction of their original collection or training process.
# 
# Packaging note: source inputs described as byte-exact above are
# normalized to UTF-8/LF text with a final newline in this standalone
# notebook package. Route JSON values and parent policy behavior are unchanged.

# Final public-entry guard; measured separately and compared on captured observations.
_V28_CORE=agent
del agent
_IMPL.chassis.diagnostics['v28_entry_errors']=0
def agent(observation,configuration=None):
    try:
        return _V28_CORE(observation,configuration)
    except Exception:
        _IMPL.chassis.diagnostics['v28_entry_errors']+=1
        return {'farmer':['PASS'],'hands':[],'market':[]}
agent.telemetry=_ROOM_STATS

# EXP-155: prvsiyan V221B finite tomato investment, adapted by Ahmed Berat Ozer.
# Original public source is retained under research24/public; Apache-2.0.
MAX_ORDERS=10
class FarmView(_View):
    def __init__(self,obs):super().__init__(obs,int(obs['player']),_IMPL.chassis.cfg)
    def inventory(self,actor):return self.inv(actor)
def projected_shed(action,view):return _IMPL.chassis._projected_shed(action,view)

CROP_MIN_PRICE=70

# V219: a finite late tomato investment with dedicated, observed workers.
_V219_PARENT = agent
del agent
_V219_FERTILIZE = True  # Builder changes only this flag for the ablation.
_V219_STATES = {}
_V219_REPORT = {'commitments': 0, 'hire_requests': 0, 'confirmed_workers': 0,
                'hire_shortfalls': 0, 'plant_requests': 0, 'confirmed_plants': 0,
                'water_requests': 0, 'fertilize_requests': 0, 'harvest_requests': 0,
                'confirmed_harvest_units': 0, 'drop_requests': 0,
                'tomato_sale_requests': 0, 'budget_declines': 0, 'lost_plants': 0}


def _v219_fib(n):
    a, b = 1, 1
    for _ in range(n): a, b = b, a+b
    return a


def _v219_native_day(native, day):
    tape = _IMPL.chassis.routes[native['route']]
    return tape[day*24:min((day+1)*24,719)]


def _v219_qualifies(obs, native):
    farm=obs['farms'][obs['player']]
    if len(farm['tiles']) != 10 or set(farm['unlocked_quadrants']) != {'NW','NE','SW'}:
        return False
    if farm['money'] < 12000 or obs['market']['prices']['TOMATO'] < CROP_MIN_PRICE:
        return False
    if sum(s in ('PIZZA_SHOP','FARMERS_MARKET') for s in obs['town']['unlocked_shops']) < 3:
        return False
    if any(farm['tiles'][y][x] != 'LOCKED' for y in (5,6) for x in range(5,10)):
        return False
    if obs['private']['seeds'].get('TOMATO',0) or obs['private']['shed'].get('TOMATO',0):
        return False
    if any(isinstance(t,dict) and t.get('crop')=='TOMATO' for row in farm['tiles'] for t in row):
        return False
    # The investment uses spare land and new worker indices. Avoid taking over
    # any native tomato or land purchase obligation on the known own schedule.
    for tape in _IMPL.chassis.routes.values():
        for a in tape[432:719]:
            if any(o and o[0]=='BUY_LAND' for o in a.get('market',[])):return False
            if any(c==['PLANT','TOMATO'] for c in [a.get('farmer')]+a.get('hands',[])):return False
    return True


def _v219_walk(pos, target):
    x,y=pos;tx,ty=target
    if x != tx:return ['EAST' if x < tx else 'WEST']
    if y != ty:return ['SOUTH' if y < ty else 'NORTH']
    return None


def _v219_home(pos):
    return min(((4,4),(5,4),(4,5),(5,5)),key=lambda p:abs(pos[0]-p[0])+abs(pos[1]-p[1]))


def _v219_request(obs, action, state, native):
    step=int(obs['step']);day=step//24;offset=step%24
    farm=obs['farms'][obs['player']];private=obs['private']
    # If the planting-day transaction could not complete, abandon investment.
    # Later purchases would miss the finite day26..29 production window.
    if not state.get('committed') and day!=18:return action
    if state.get('requested_day')==day or offset>3:return action
    planned=_v219_native_day(native,day)
    remaining=planned[offset+1:]
    if any(o and o[0]=='HIRE' for a in remaining for o in a.get('market',[])):
        return action
    parent_hires=sum(bool(o) and o[0]=='HIRE' for o in action['market'])
    expected=max(len(a.get('hands',[])) for a in planned)
    if len(farm['hands'])+parent_hires != expected:return action
    fertilizer=bool(_V219_FERTILIZE and day in (24,27) and _r79_tomato_fertilizer_worthwhile(obs,action))
    # One watering tour: at most 2 entry moves + 9 between tiles + 10 waters.
    # A hire request by hour2 leaves at least21 callbacks after confirmation.
    crop_workers=1 if day in (19,20,21,22,23,25) and offset<=2 else (3 if 26<=day<=28 else 2)
    labor=_r53_labor_assignment(obs,action,fertilizer)
    if labor is not None:crop_workers=labor['workers']
    count=crop_workers+int(fertilizer and day==27 and labor is None)
    extra=[]
    if not state.get('committed'):
        extra += [['BUY_LAND'],['BUY_SEED','TOMATO',10]]
    fertilizer_quantity=_r70_parent_fert_qty(obs,action,planned,offset) if fertilizer else 0
    if fertilizer:extra.append(['BUY_PRODUCT','FERTILIZER',fertilizer_quantity])
    extra += [['HIRE'] for _ in range(count)]
    if len(action['market'])+len(extra)>MAX_ORDERS:return action
    # No assumed sale proceeds. Reserve 3,000 for parent obligations and price
    # movement; the qualification separately requires 12,000 initial liquidity.
    budget=sum(_v219_fib(n) for n in range(farm['hires_today'],farm['hires_today']+parent_hires+count))
    if not state.get('committed'):budget+=4500
    if fertilizer:budget+=fertilizer_quantity*(obs['market']['prices']['FERTILIZER']+5)
    for order in action['market']:
        if not order:continue
        if order[0]=='BUY_PRODUCT':budget+=int(order[2])*(int(obs['market']['prices'][order[1]])+10)
        elif order[0]=='BUY_ANIMAL':budget+=int(order[2])*{'COW':400,'SHEEP':500,'GOOSE':300}[order[1]]
        elif order[0]=='BUY_SEED':budget+=int(order[2])*{'WHEAT':10,'CARROT':20,'TOMATO':50,'STRAWBERRY':100,'MELON':80}[order[1]]
    if farm['money']<budget+3000:
        _V219_REPORT['budget_declines']+=1;return action
    state['pending']={'step':step,'first_actor':expected+1,'count':count,'crop_workers':crop_workers,'fertilizer':fertilizer,'labor':labor}
    if labor is not None:
        _R53_LABOR_REPORT['labor_requests']+=1;_R53_LABOR_REPORT['labor_hires_avoided']+=1;_R53_LABOR_REPORT['labor_day'+str(day)]+=1
    state['requested_day']=day
    _V219_REPORT['hire_requests']+=count
    if not state.get('committed'):
        state['committed']=True;_V219_REPORT['commitments']+=1
    changed=copy.deepcopy(action);changed['market']+=extra
    return changed


def _v219_worker(obs, state, actor, role):
    day=int(obs['step'])//24;step=int(obs['step']);view=FarmView(obs)
    pos=tuple(view.positions[actor]);inv=view.inventory(actor)
    targets=role['targets']
    # Actual cargo differences, observed on the next callback, verify harvests.
    previous=state['last_work'].get(actor)
    if previous and previous['step']==step-1 and previous['command']==['HARVEST']:
        _V219_REPORT['confirmed_harvest_units']+=max(0,int(inv.get('TOMATO',0))-previous['tomatoes'])
    if role.get('needs_fertilizer') and not role.get('loaded'):
        home=_v219_home(pos)
        walk=_v219_walk(pos,home)
        if walk:return walk
        desired=role.get('fertilizer_quantity',10 if role['kind']=='fertilizer' else 5)
        if inv.get('FERTILIZER',0)>=desired:role['loaded']=True
        elif role.get('pickup_requested'):
            # Never spend repeated turns waiting for stock that was not bought.
            role['loaded']=True;role['fertilizer_available']=int(inv.get('FERTILIZER',0))
        elif view.shed.get('FERTILIZER',0)>=desired:
            role['pickup_requested']=True;return ['PICKUP','FERTILIZER',desired]
        else:role['loaded']=True
    todo=[]
    for target in targets:
        x,y=target;tile=view.tiles[y][x]
        tomato=isinstance(tile,dict) and tile.get('crop')=='TOMATO'
        if tomato and target not in state['seen_plants']:
            state['seen_plants'].add(target);_V219_REPORT['confirmed_plants']+=1
        if target in state['seen_plants'] and not tomato and target not in state['lost']:
            state['lost'].add(target);_V219_REPORT['lost_plants']+=1
        command=None
        if role['kind']=='fertilizer':
            if tomato and tile.get('fertilized_until_day',-1)<day+2 and inv.get('FERTILIZER',0)>0:
                command=['FERTILIZE']
        elif day==18 and not tomato:
            if tile is None and obs['private']['seeds'].get('TOMATO',0)>0:command=['PLANT','TOMATO']
            elif isinstance(tile,dict) and tile.get('kind')=='WEED':command=['DIG']
        elif tomato:
            # No later production follows the final day, so watering then would
            # consume time needed to harvest and deliver the final cargo.
            if day<29 and not tile.get('watered_today'):command=['WATER']
            elif role.get('needs_fertilizer') and tile.get('fertilized_until_day',-1)<day+2 and inv.get('FERTILIZER',0)>0:
                command=['FERTILIZE']
            elif tile.get('yield_units',0)>0:command=['HARVEST']
        if command:todo.append((target,command))
    # Final return has priority once only the exact distance plus DROP remains.
    home=_v219_home(pos);distance=abs(pos[0]-home[0])+abs(pos[1]-home[1])
    if step>=718-distance and inv.get('TOMATO',0):
        return _v219_walk(pos,home) or ['PLACE','TOMATO',int(inv.get('TOMATO',0))]
    if todo:
        target,command=min(todo,key=lambda v:(abs(pos[0]-v[0][0])+abs(pos[1]-v[0][1]),targets.index(v[0])))
        return _v219_walk(pos,target) or command
    if inv.get('TOMATO',0):return _v219_walk(pos,home) or ['PLACE','TOMATO',int(inv['TOMATO'])]
    if any(inv.values()):return _v219_walk(pos,home) or ['DROP']
    return ['PASS']


def agent(observation, configuration=None):
    action=_V219_PARENT(observation,configuration)
    step=int(observation['step']);player=int(observation['player']);day=step//24
    state=_V219_STATES.get(player)
    if state is None or step<=state['last_step']:
        state={'last_step':step,'day':-1,'workers':{},'last_work':{},'seen_plants':set(),'lost':set(),
               'targets':[(x,y) for y in (5,6) for x in range(5,10)]}
        _V219_STATES[player]=state
    state['last_step']=step
    native=_IMPL.chassis.players[player]
    if step==432:state['eligible']=_v219_qualifies(observation,native)
    if not state.get('eligible') or day<18:return action
    if state['day']!=day:
        state['day']=day;state['workers']={};state['last_work']={}
    farm=observation['farms'][player]
    pending=state.pop('pending',None)
    if pending:
        if len(farm['hands'])+1 >= pending['first_actor']+pending['count'] and 'SE' in farm['unlocked_quadrants']:
            for index in range(pending['count']):
                fertilizer_worker=index==pending['crop_workers']
                if fertilizer_worker:targets=state['targets']
                elif pending['crop_workers']==1:targets=state['targets']
                elif pending['crop_workers']==2:targets=state['targets'][index*5:index*5+5]
                else:targets=[[(5,5),(6,5),(7,5)],[(8,5),(9,5),(9,6),(8,6)],[(5,6),(6,6),(7,6)]][index]
                state['workers'][pending['first_actor']+index]={'kind':'fertilizer' if fertilizer_worker else 'crop','targets':targets,
                    'needs_fertilizer':pending['fertilizer'] and (day==24 or fertilizer_worker)}
                if pending.get('labor') is not None:
                    role=state['workers'][pending['first_actor']+index]
                    role['targets']=[tuple(p) for p in pending['labor']['paths'][index]]
                    role['needs_fertilizer']=pending['labor']['fertilizer'];role['fertilizer_quantity']=len(role['targets'])
                    if tuple(farm['hands'][pending['first_actor']+index-1])!=tuple(pending['labor']['spawns'][index]):_R53_LABOR_REPORT['labor_spawn_errors']+=1
                    if index==0:_R53_LABOR_REPORT['labor_confirmed']+=1
            _V219_REPORT['confirmed_workers']+=pending['count']
        else:_V219_REPORT['hire_shortfalls']+=pending['count']
    action=_v219_request(observation,action,state,native)
    if state['workers']:
        commands=[action.get('farmer') or ['PASS']]+list(action.get('hands') or [])
        commands += [['PASS'] for _ in range(len(farm['hands'])+1-len(commands))]
        for actor,role in state['workers'].items():
            if actor>=len(commands):continue
            command=_v219_worker(observation,state,actor,role)
            commands[actor]=command
            name={'PLANT':'plant_requests','WATER':'water_requests','FERTILIZE':'fertilize_requests',
                  'HARVEST':'harvest_requests','DROP':'drop_requests'}.get(command[0])
            if name:_V219_REPORT[name]+=1
            state['last_work'][actor]={'step':step,'command':command,'tomatoes':observation['private']['inventories'][actor].get('TOMATO',0)}
        action=copy.deepcopy(action);action['farmer'],action['hands']=commands[0],commands[1:]
    if state.get('committed') and len(action['market'])<MAX_ORDERS and not any(o[:2]==['SELL','TOMATO'] for o in action['market']):
        quantity=projected_shed(action,FarmView(observation)).get('TOMATO',0)
        if quantity>0:
            action=copy.deepcopy(action);action['market'].append(['SELL','TOMATO',quantity])
            _V219_REPORT['tomato_sale_requests']+=quantity
    return action


agent.telemetry=_V219_REPORT

# V221B: labor-only ablation of frozen V219G; not yet publicly scored.


# Crop workers own their final routes after commitment. A private parent shadow
# does not contain these obligations, so terminal rescue must abstain there.
_ORIGINAL_SHADOW_TERMINAL=_shadow_terminal
def _shadow_terminal(obs,config):
    if _V219_STATES.get(int(obs['player']),{}).get('committed'):return None
    return _ORIGINAL_SHADOW_TERMINAL(obs,config)

APPLY_TIMING=False

_EXPERIMENT_PARENT=agent
del agent
_V219_REPORT['extra_fertilizer_days']=0
_V219_REPORT['reordered_market_turns']=0
_V219_REPORT['errors']=0
def agent(observation,configuration=None):
    try:
        action=_EXPERIMENT_PARENT(observation,configuration)
        if APPLY_TIMING and int(observation['step'])>=144:action=_v224_sales_first(action)
        return action
    except Exception:
        _V219_REPORT['errors']+=1
        return {'farmer':['PASS'],'hands':[],'market':[]}
agent.telemetry=_V219_REPORT

def _v224_sales_first(action):
    original=action.get('market',[])[:MAX_ORDERS]
    orders=[list(o) for o in original if o and (o[0] in ('HIRE','BUY_LAND') or (len(o)>=3 and int(o[2])>0))]
    for index in range(len(orders)):
        order=orders[index]
        if order[0]!='SELL':continue
        cursor=index
        while cursor>0:
            previous=orders[cursor-1]
            if previous[0]=='SELL':break
            if previous[0] in ('BUY_PRODUCT','BUY_ANIMAL') and previous[1]==order[1]:break
            orders[cursor-1],orders[cursor]=orders[cursor],orders[cursor-1]
            cursor-=1
    if orders==original:return action
    _V219_REPORT['reordered_market_turns']+=1
    changed=copy.deepcopy(action);changed['market']=orders
    return changed
_ORDER_PARENT=agent
del agent

def agent(observation,configuration=None):
    try:
        action=_ORDER_PARENT(observation,configuration)
        if int(observation["step"])>=144:action=_v224_sales_first(action)
        return action
    except Exception:
        _V219_REPORT["errors"]+=1
        return {"farmer":["PASS"],"hands":[],"market":[]}
agent.telemetry=_V219_REPORT

_V31_CORE=agent
del agent
_IMPL.chassis.diagnostics['production_errors']=0
_IMPL.chassis.diagnostics['v31_entry_errors']=0
def agent(observation,configuration=None):
    before=_V219_REPORT['errors']
    try:
        action=_V31_CORE(observation,configuration)
        _IMPL.chassis.diagnostics['production_errors']+=_V219_REPORT['errors']-before
        return action
    except Exception:
        _IMPL.chassis.diagnostics['v31_entry_errors']+=1
        return {'farmer':['PASS'],'hands':[],'market':[]}
agent.telemetry=_V219_REPORT

# Apache-2.0; later cattle transfer from prvsiyan, Moon (2026-09-10).
# Bounded livestock substitution; confirm owned animals before redirecting workers.
_V231_PARENT=agent
_V231_CAP=4
_V231_STATES={}
_V231_REPORT={}

def _v231_new_state():
    return {'last':-1,'confirmed':0,'reserved':0,'pending_buy':None,
            'carrying':{},'pending_places':[],'sites':{},'milk_credit':0,
            'requested':0,'failed_purchase_units':0,'picked':0,'placed':0,
            'failed_placements':0,'extra_milk_harvested':0,'extra_milk_sale_requests':0}

def _v231_controller(obs,action,state,cap):
    step=int(obs['step']);seat=int(obs['player']);farm=obs['farms'][seat]
    private=obs['private'];shed=private['shed'];inventories=private['inventories']
    positions=[farm['farmer'],*farm['hands']]
    pending=state['pending_buy']
    if pending is not None:
        gained=max(0,int(shed.get('COW',0))-pending['before'])
        confirmed=min(pending['quantity'],gained)
        state['confirmed']+=confirmed;state['reserved']+=confirmed
        state['failed_purchase_units']+=pending['quantity']-confirmed
        state['pending_buy']=None
    for pending in state['pending_places']:
        x,y=pending['site'];tile=farm['tiles'][y][x]
        if (isinstance(tile,dict) and tile.get('animal')=='COW'
                and tile.get('placed_day')==pending['day']):
            state['sites'][(x,y)]=pending['day'];state['placed']+=1
            actor=pending['actor'];state['carrying'][actor]=max(0,state['carrying'].get(actor,0)-1)
        else:state['failed_placements']+=1
    state['pending_places']=[]
    state['last']=step
    result=copy.deepcopy(action)
    workers=[result.get('farmer') or ['PASS'],*(result.get('hands') or [])]
    seen_harvest=set();cow_available=int(shed.get('COW',0));occupied=set()
    for actor,work in enumerate(workers[:len(positions)]):
        inventory=inventories[actor] if actor<len(inventories) else {}
        x,y=positions[actor];tile=farm['tiles'][y][x];site=(x,y)
        if (work==['HARVEST'] and site in state['sites'] and site not in seen_harvest
                and isinstance(tile,dict) and tile.get('animal')=='COW'
                and tile.get('placed_day')==state['sites'][site]):
            units=max(0,int(tile.get('yield_units',0)))
            state['milk_credit']+=units;state['extra_milk_harvested']+=units
            seen_harvest.add(site)
        if len(work)>=2 and work[:2]==['PICKUP','SHEEP']:
            quantity=max(0,int(work[2]) if len(work)>2 else 1)
            center=len(farm['tiles'])//2
            if (quantity and state['reserved']>=quantity and cow_available>=quantity
                    and x in (center-1,center) and y in (center-1,center)
                    and not any(inventory.get(a,0) for a in ('COW','SHEEP','GOOSE'))):
                work[1]='COW';state['reserved']-=quantity;cow_available-=quantity
                state['carrying'][actor]=state['carrying'].get(actor,0)+quantity
                state['picked']+=quantity
        if (len(work)>=2 and work[:2]==['PLACE','SHEEP']
                and state['carrying'].get(actor,0)>0 and inventory.get('COW',0)>0
                and isinstance(tile,dict) and tile.get('kind')=='PASTURE'
                and 'animal' not in tile and site not in occupied):
            work[1]='COW'
            state['pending_places'].append({'actor':actor,'site':site,'day':step//24})
        if (len(work)>=2 and work[0]=='PLACE' and work[1] in ('COW','SHEEP','GOOSE')
                and inventory.get(work[1],0)>0):occupied.add(site)
    result['farmer'],result['hands']=workers[0],workers[1:]
    market=result.get('market',[])
    animal_orders=[o for o in market if len(o)>=3 and o[0]=='BUY_ANIMAL']
    shops=obs['town']['unlocked_shops'];prices=obs['market']['prices']
    counts={'COW':0,'SHEEP':0}
    for line in farm['tiles']:
        for tile in line:
            if isinstance(tile,dict) and tile.get('animal') in counts:counts[tile['animal']]+=1
    cargo=sum(int(inv.get(a,0)) for inv in inventories for a in ('COW','SHEEP','GOOSE'))
    stock_animals=sum(int(shed.get(a,0)) for a in ('COW','SHEEP','GOOSE'))
    milk_shops=sum(shop in ('PIZZA_SHOP','ICE_CREAM_SHOP','SMOOTHIE_SHOP') for shop in shops)
    if (216<=step<=227 and len(shops)>=3 and state['confirmed']<cap and not state['reserved']
            and not any(state['carrying'].values()) and not state['pending_places']
            and not cargo and not stock_animals and len(animal_orders)==1
            and animal_orders[0][1]=='SHEEP' and milk_shops>=2 and 'YARN_STORE' not in shops
            and int(prices.get('MILK',0))>=int(prices.get('WOOL',0))
            and counts['COW']>=4 and counts['SHEEP']>=2):
        order=animal_orders[0];quantity=int(order[2])
        if 1<=quantity<=2 and quantity<=cap-state['confirmed']:
            order[1]='COW';state['requested']+=quantity
            state['pending_buy']={'before':int(shed.get('COW',0)),'quantity':quantity}
    # Sell only additional physically harvested production at an existing sale slot.
    if state['milk_credit']>0:
        stock=projected_shed(result,FarmView(obs))
        total_planned=sum(max(0,int(o[2])) for o in market if len(o)>=3 and o[:2]==['SELL','MILK'])
        extra=min(state['milk_credit'],max(0,int(stock.get('MILK',0))-total_planned))
        if extra:
            for order in market:
                if len(order)>=3 and order[:2]==['SELL','MILK'] and int(order[2])>0:
                    order[2]=int(order[2])+extra
                    state['milk_credit']-=extra;state['extra_milk_sale_requests']+=extra
                    break
    result['market']=market
    return result

def agent(observation,configuration=None):
    step=int(observation['step']);seat=int(observation['player'])
    state=_V231_STATES.get(seat)
    if state is None or step<=state['last']:
        state=_V231_STATES[seat]=_v231_new_state()
    action=_V231_PARENT(observation,configuration)
    action=_v231_controller(observation,action,state,_V231_CAP)
    _V231_REPORT.clear();_V231_REPORT.update(_V231_PARENT.telemetry)
    for name in ('confirmed','reserved','requested','failed_purchase_units','picked','placed',
                 'failed_placements','extra_milk_harvested','extra_milk_sale_requests','milk_credit'):
        _V231_REPORT['cattle_'+name]=state[name]
    _V231_REPORT['cattle_carried_pending']=sum(state['carrying'].values())
    return action

agent.telemetry=_V231_REPORT


# EXP-167, adapted from Dmitrii Gluzdov's Two Coins, One Sheep (Apache-2.0).
# Reserve only physically available stock after the final parent worker actions.
_R36_SALE_PARENT=agent
_R36_NATIVE_LEAD=Chassis._sell_lead
_R36_NATIVE_SUPPRESS=Chassis._apply_suppression
_R36_SALE_REPORT={}

def _r36_native_lead(self,action,view,projected,route,step,next_sup):
    if step<288 or step>=696:
        return _R36_NATIVE_LEAD(self,action,view,projected,route,step,next_sup)

def _r36_suppress(action,state,step):
    _R36_NATIVE_SUPPRESS(action,state,step)
    due=state.get('r36_debts',{}).pop(step,{})
    for order in action.get('market',[]):
        if len(order)>=3 and order[0]=='SELL':
            removed=min(max(0,int(order[2])),due.get(order[1],0))
            order[2]-=removed
            due[order[1]]=due.get(order[1],0)-removed

Chassis._sell_lead=_r36_native_lead
Chassis._apply_suppression=staticmethod(_r36_suppress)

def _r36_reserve(obs,action):
    step=int(obs['step'])
    # The final planner forecasts its own parent, so keep its full window native.
    if not 288<=step<696:return action
    native=_IMPL.chassis.players[int(obs['player'])]
    tape=_IMPL.chassis.routes[native['route']]
    end=min(695,step+_R37_HORIZONS.get(int(obs['player']),2),(step//72+1)*72-1)
    if end<=step:return action
    commands=[action.get('farmer') or ['PASS'],*(action.get('hands') or [])]
    view=FarmView(obs)
    # This projection intentionally abstains on ambiguous animal depot returns.
    if any(len(c)>1 and c[0]=='PLACE' and c[1] in ANIMAL_STRUCTURE
           and view.inv(i).get(c[1],0)>0 for i,c in enumerate(commands[:len(view.positions)])):
        return action
    stock=projected_shed(action,view)
    market=action.get('market',[])
    blocked={o[1] for o in market if len(o)>1 and o[0] in ('SELL','BUY_PRODUCT')}
    blocked.update(c[1] for c in commands if len(c)>1 and c[0]=='PICKUP')
    blocked.update(c[1] for queue in native['pending'].values() for pos,c in queue
                   if len(c)>1 and c[0]=='PICKUP')
    debts=native['sell_state'].setdefault('r36_debts',{})
    for item in PRODUCTS:
        if item in ('WHEAT','FERTILIZER') or item in blocked or view.prices.get(item,0)<2:continue
        available=max(0,int(stock.get(item,0)))
        if not available or len(market)>=10:continue
        reservations=[]
        for due_step in range(step+1,end+1):
            future=tape[due_step]
            work=[future.get('farmer') or ['PASS'],*(future.get('hands') or [])]
            if any(len(c)>1 and c[:2]==['PICKUP',item] for c in work):break
            if any(len(o)>1 and o[:2]==['BUY_PRODUCT',item] for o in future.get('market',[])):break
            planned=sum(max(0,int(o[2])) for o in future.get('market',[]) if len(o)>=3 and o[:2]==['SELL',item])
            amount=min(available,max(0,planned-debts.get(due_step,{}).get(item,0)))
            if amount:
                reservations.append((due_step,amount));available-=amount
            if not available:break
        qty=sum(q for _,q in reservations)
        if qty:
            market.append(['SELL',item,qty])
            for due,q in reservations:
                debt=debts.setdefault(due,{})
                debt[item]=debt.get(item,0)+q
            _R36_SALE_REPORT['sale_reserved_units']+=qty
            _R36_SALE_REPORT['sale_reservations']+=1
    return action

def agent(observation,configuration=None):
    if int(observation.get('step',0))==0:
        _R36_SALE_REPORT.update(sale_reserved_units=0,sale_reservations=0,sale_errors=0)
    action=_R36_SALE_PARENT(observation,configuration)
    try:
        if configuration is None or all(configuration.get(k,v)==v for k,v in
            [('boardSize',10),('turnsPerDay',24),('shedCapacity',100),('maxMarketOrdersPerTurn',10)]):
            action=_r36_reserve(observation,action)
            if int(observation['step'])>=288:action=_v224_sales_first(action)
    except Exception:
        _R36_SALE_REPORT['sale_errors']=_R36_SALE_REPORT.get('sale_errors',0)+1
    _R36_SALE_REPORT.update(_R36_SALE_PARENT.telemetry)
    return action

agent.telemetry=_R36_SALE_REPORT

# Ensure the Kaggle-selected final callable is the exported policy.
agent = globals().pop("agent")


# Public capability transfer: lucifer19; Flexon is the same Two Coins asset set.
# Apache-2.0; exact functions from Kaggle kaggle-environments 1.32.7.
# https://github.com/Kaggle/kaggle-environments/tree/master/kaggle_environments/envs/kaggriculture
import math
_R37_MARKET_PARAMS = {'WHEAT': {'base': 25, 'I0': 10000, 'T': 400, 'below_func': 'sqrt', 'below_target': 0.8, 'above_func': 'log', 'above_target': 0.2}, 'CARROT': {'base': 35, 'I0': 10000, 'T': 450, 'below_func': 'hinge', 'below_target': 1.0, 'above_func': 'sqrt', 'above_target': 0.7}, 'TOMATO': {'base': 60, 'I0': 10000, 'T': 200, 'below_func': 'hinge', 'below_target': 0.4, 'above_func': 'sqrt', 'above_target': 0.6}, 'STRAWBERRY': {'base': 120, 'I0': 10000, 'T': 100, 'below_func': 'sqrt', 'below_target': 0.7, 'above_func': 'linear', 'above_target': 1.6}, 'MELON': {'base': 250, 'I0': 10000, 'T': 300, 'below_func': 'log', 'below_target': 0.2, 'above_func': 'sq', 'above_target': 3.6}, 'EGG': {'base': 50, 'I0': 10000, 'T': 332, 'below_func': 'hinge', 'below_target': 0.4, 'above_func': 'log', 'above_target': 0.2}, 'MILK': {'base': 160, 'I0': 10000, 'T': 122, 'below_func': 'sqrt', 'below_target': 0.6, 'above_func': 'linear', 'above_target': 1.6}, 'WOOL': {'base': 200, 'I0': 10000, 'T': 105, 'below_func': 'log', 'below_target': 0.2, 'above_func': 'sq', 'above_target': 3.2}, 'FERTILIZER': {'base': 100, 'I0': 10000, 'T': 200, 'below_func': 'linear', 'below_target': 0.4, 'above_func': 'linear', 'above_target': 0.4}}
_R37_PRICE_FLOOR = 1
_R37_HINGE_GAIN = 8.0
def _r37_shape(func, x, T=None):
    x = max(0.0, x)
    if func == "linear": return x
    if func == "sq":     return x * x
    if func == "sqrt":   return math.sqrt(x)
    if func == "log":    return math.log(1.0 + x)
    if func == "log10":  return math.log10(1.0 + x)
    if func == "hinge":
        # Degenerates to linear if T is missing or non-positive.
        if not T or T <= 0:
            return x
        u = x / T
        return u + _R37_HINGE_GAIN * max(0.0, u - 1.0) ** 2
    return x

def _r37_market_price(item, inventory, params=None):
    """Floor at _R37_PRICE_FLOOR."""
    p = (params or _R37_MARKET_PARAMS)[item]
    base = p["base"]
    I0 = p["I0"]
    T = p["T"]
    if inventory < I0:
        f = p["below_func"]
        amp = p["below_target"] * base / _r37_shape(f, T, T)
        price = base + amp * _r37_shape(f, I0 - inventory, T)
    else:
        f = p["above_func"]
        amp = p["above_target"] * base / _r37_shape(f, T, T)
        price = base - amp * _r37_shape(f, inventory - I0, T)
    return max(_R37_PRICE_FLOOR, int(round(price)))

def _r37_similarity(observation):
    """Empty tiles cannot make two unrelated production layouts look alike."""
    farms = observation['farms']
    own, rival = farms[observation['player']], farms[1-observation['player']]
    if own['unlocked_quadrants'] != rival['unlocked_quadrants']:
        return 0.0
    matches = total = 0
    for a, b in zip([t for row in own['tiles'] for t in row],
                    [t for row in rival['tiles'] for t in row]):
        sa = (a.get('crop'), a.get('animal')) if isinstance(a, dict) else (None, None)
        sb = (b.get('crop'), b.get('animal')) if isinstance(b, dict) else (None, None)
        if sa != (None, None) or sb != (None, None):
            total += 1
            matches += sa == sb
    return matches / total if total >= 8 else 0.0


def _r37_quote_priority(observation, order, stock):
    """Revenue exposed to a small rival batch, not nominal headline revenue."""
    item = order[1]
    quantity = min(max(0, int(order[2])), stock.get(item, 0))
    if not quantity or item not in _R37_MARKET_PARAMS:
        return 0.0
    inventory = observation['market']['inventory'][item]
    params = {k: dict(v) for k, v in _R37_MARKET_PARAMS.items()}
    for k, patch in observation['market'].get('params', {}).items():
        if k in params:
            params[k].update(patch)
    rival = observation['farms'][1-observation['player']]
    crop_item = item if item in ('WHEAT','CARROT','TOMATO','STRAWBERRY','MELON') else None
    animal = {'EGG':'GOOSE','MILK':'COW','WOOL':'SHEEP'}.get(item)
    standing = sum(max(0, int(t.get('yield_units', 0))) for row in rival['tiles'] for t in row
                   if isinstance(t, dict) and
                   ((crop_item is not None and t.get('crop') == crop_item) or
                    (animal is not None and t.get('animal') == animal)))
    # Public fields do not reveal the rival shed. Eight units are a scenario,
    # not a recovered hidden quantity; visible ripe yield increases the stress.
    batch = min(24, max(8, standing))
    now = sum(_r37_market_price(item, inventory+j, params) for j in range(quantity))
    later = sum(_r37_market_price(item, inventory+batch+j, params) for j in range(quantity))
    return now-later


def _r37_reorder_sales(observation, action):
    """Keep quantities and purchase barriers; rank distinct contiguous sales."""
    stock = projected_shed(action, FarmView(observation))
    orders = [list(o) for o in action['market']]
    start = 0
    while start < len(orders):
        if orders[start][0] != 'SELL':
            start += 1
            continue
        end = start
        while end < len(orders) and orders[end][0] == 'SELL':
            end += 1
        block = orders[start:end]
        if len({o[1] for o in block}) == len(block):
            orders[start:end] = sorted(block, key=lambda o: _r37_quote_priority(observation, o, stock), reverse=True)
        start = end
    if orders != action['market']:
        _R37_STATS['quote_reordered_turns'] += 1
        action = dict(action, market=orders)
    return action



# EXP175: bounded public cash-response probe inspired by leoprovorov,
# Two Coins Mirror Counter v1 (Apache-2.0). No hidden rival inventory.
_R44_PROBES={}
_R44_REPORT=dict(probe_matches=0,probe_four_turn_calls=0,probe_errors=0)

def _r44_before(obs):
    player=int(obs['player']);step=int(obs['step'])
    st=_R44_PROBES.get(player)
    if st is None or step<=st['step']:
        st=_R44_PROBES[player]={'step':-1,'money':None,'probe':0,'matched':False}
    if step==0:_R44_REPORT.update(probe_matches=0,probe_four_turn_calls=0,probe_errors=0)
    money=tuple(float(obs['farms'][i]['money']) for i in (player,1-player))
    if st['money'] is not None and st['probe']>=100 and _r37_similarity(obs)>=.90:
        own=money[0]-st['money'][0];rival=money[1]-st['money'][1]
        if own>0 and rival>0 and abs(own-rival)<=max(5.0,.05*st['probe']):
            if not st['matched']:_R44_REPORT['probe_matches']+=1
            st['matched']=True
    st.update(step=step,money=money,probe=0)
    return st

def _r44_after(obs,action,st):
    step=int(obs['step']);player=int(obs['player'])
    if not 336<=step<648 or st['matched']:return
    # Positive all-sale probes avoid mistaking equal spending for preemption.
    if not action['market'] or any(o and o[0]!='SELL' for o in action['market']):return
    debts=_IMPL.chassis.players[player]['sell_state'].get('r36_debts',{})
    own=debts.get(step+3,{})
    if own:st['probe']=sum(max(0,int(n))*int(obs['market']['prices'].get(item,0)) for item,n in own.items())

_R37_ADAPTIVE = True
_R37_QUOTE = True
# EXP-168: adapted from lucifer19 / Harvest Nocturne, Apache-2.0.
# All rivalry features use public occupied tiles; no private rival inventory.
_R37_PARENT = agent
_R37_PLAYERS = {}
_R37_HORIZONS = {}
_R37_REPORT = {}
_R37_STATS = dict(quote_reordered_turns=0, three_turn_calls=0, nocturne_errors=0)
del agent

def agent(observation, configuration=None):
    player, step = int(observation['player']), int(observation['step'])
    state = _R37_PLAYERS.get(player)
    if state is None or step <= state['step']:
        state = _R37_PLAYERS[player] = {'step': -1, 'streak': 0}
    if step == 0:
        _R37_STATS.update(quote_reordered_turns=0, three_turn_calls=0, nocturne_errors=0)
    state['step'] = step
    _R37_HORIZONS[player] = 2
    probe_state=_r44_before(observation)
    try:
        if _R37_ADAPTIVE and step < 648:
            state['streak'] = state['streak'] + 1 if _r37_similarity(observation) >= .90 else 0
            if 336 <= step < 648 and state['streak'] >= 6:
                _R37_HORIZONS[player] = 3
                _R37_STATS['three_turn_calls'] += 1
    except Exception:
        _R37_STATS['nocturne_errors'] += 1
    if _R37_HORIZONS[player]==3 and probe_state['matched']:
        _R37_HORIZONS[player]=4
        _R44_REPORT['probe_four_turn_calls']+=1
    # EXP179: four-turn reservation; retain stock, debt and purchase barriers.
    if 288 <= step < 696:_R37_HORIZONS[player] = 4
    action = _R37_PARENT(observation, configuration)
    _r44_after(observation,action,probe_state)
    if _R37_QUOTE and step >= 288:
        try:
            action = _r37_reorder_sales(observation, action)
        except Exception:
            _R37_STATS['nocturne_errors'] += 1
    _R37_REPORT.update(getattr(_R37_PARENT, 'telemetry', {}))
    _R37_REPORT.update(_R37_STATS)
    _R37_REPORT.update(_R44_REPORT)
    return action

agent.telemetry = _R37_REPORT

# Export guard: normal decisions stay identical to the frozen screened policy.
_RELEASE_PARENT=agent
_RELEASE_REPORT={}
_RELEASE_ERRORS=0
del agent

def agent(observation,configuration=None):
    global _RELEASE_ERRORS
    try:
        result=_RELEASE_PARENT(observation,configuration)
    except Exception:
        _RELEASE_ERRORS+=1
        count=0
        try:
            count=min(64,len(observation['farms'][int(observation['player'])]['hands']))
        except Exception:
            pass
        result={'farmer':['PASS'],'hands':[['PASS'] for _ in range(count)],'market':[]}
    _RELEASE_REPORT.update(getattr(_RELEASE_PARENT,'telemetry',{}))
    _RELEASE_REPORT['release_errors']=_RELEASE_ERRORS
    return result

agent.telemetry=_RELEASE_REPORT
agent=globals().pop('agent')

# Adapted from prvsiyan / The Soil Remembers Rain, Apache-2.0.
# V233: bounded, financed six-sheep SE discovery investment.
_V233_PARENT=agent
del agent
_V233_STATES={}
_V233_REPORT=dict(sheep_commit_requests=0,sheep_committed=0,sheep_hire_requests=0,
    sheep_workers_confirmed=0,sheep_hire_shortfalls=0,sheep_budget_declines=0,
    sheep_capacity_declines=0,sheep_purchase_shortfalls=0,sheep_feed_buy_requests=0,
    sheep_wool_harvested=0,sheep_fert_collected=0,sheep_extra_wool_sales=0,
    sheep_extra_fert_sales=0,sheep_rescue_feed_requests=0)

def _v233_eligible(obs,native):
    farm=obs['farms'][obs['player']];prices=obs['market']['prices']
    if len(farm['tiles'])!=10 or set(farm['unlocked_quadrants'])!={'NW','NE','SW'}:return False
    if obs['town']['unlocked_shops'].count('YARN_STORE')<2 or prices['WOOL']<220 or prices['WHEAT']>45:return False
    if any(farm['tiles'][y][x]!='LOCKED' for y in (5,6) for x in range(5,8)):return False
    if obs['private']['shed'].get('SHEEP',0) or any(i.get('SHEEP',0) for i in obs['private']['inventories']):return False
    for day in range(12,30):
        for a in _v219_native_day(native,day):
            if any(o and (o[0]=='BUY_LAND' or o[:2]==['BUY_ANIMAL','SHEEP']) for o in a.get('market',[])):return False
            if any(c and c[0] in ('PICKUP','PLACE') and len(c)>1 and c[1]=='SHEEP' for c in [a.get('farmer')]+a.get('hands',[])):return False
    return True

def _v233_request(obs,action,state,native):
    step=int(obs['step']);day=step//24;hour=step%24
    if hour>(2 if state.get('committed') else 1) or state.get('requested_day')==day:return action
    if not state.get('committed') and (day!=12 or not _v233_eligible(obs,native)):return action
    planned=_v219_native_day(native,day)
    if any(o and o[0]=='HIRE' for a in planned[hour+1:] for o in a.get('market',[])):return action
    farm=obs['farms'][obs['player']];market=action.get('market',[])
    parent_hires=sum(bool(o) and o[0]=='HIRE' for o in market)
    expected=max(len(a.get('hands',[])) for a in planned)
    if len(farm['hands'])+parent_hires!=expected:return action
    initial=not state.get('committed')
    extra=([['BUY_LAND'],['BUY_ANIMAL','SHEEP',6]] if initial else [])+[['BUY_PRODUCT','WHEAT',6],['HIRE'],['HIRE']]
    if len(market)+len(extra)>MAX_ORDERS:return action
    stock=projected_shed(action,FarmView(obs))
    incoming=6+6*initial
    budget=7000*initial+6*(int(obs['market']['prices']['WHEAT'])+10)
    budget+=sum(_v219_fib(n) for n in range(farm['hires_today'],farm['hires_today']+parent_hires+2))
    for o in market:
        if not o:continue
        if o[0]=='BUY_LAND':return action
        if o[0]=='BUY_PRODUCT':
            incoming+=int(o[2]);budget+=int(o[2])*(int(obs['market']['prices'][o[1]])+10)
        elif o[0]=='BUY_ANIMAL':
            incoming+=int(o[2]);budget+=int(o[2])*{'SHEEP':500,'COW':400,'GOOSE':300}[o[1]]
        elif o[0]=='BUY_SEED':budget+=int(o[2])*{'WHEAT':10,'CARROT':20,'TOMATO':50,'STRAWBERRY':100,'MELON':80}[o[1]]
    if sum(stock.values())+incoming>100:
        _V233_REPORT['sheep_capacity_declines']+=1;return action
    if farm['money']<budget+(3000 if initial else 1000):
        _V233_REPORT['sheep_budget_declines']+=1;return action
    state['requested_day']=day
    state['pending']={'first':expected+1,'initial':initial}
    _V233_REPORT['sheep_hire_requests']+=2;_V233_REPORT['sheep_feed_buy_requests']+=6
    if initial:_V233_REPORT['sheep_commit_requests']+=1
    result=copy.deepcopy(action);result['market']=market+extra
    return result

def _v233_worker(obs,actor,targets):
    farm=obs['farms'][obs['player']];private=obs['private'];step=int(obs['step'])
    pos=tuple(farm['hands'][actor-1]);inv=private['inventories'][actor]
    access=((4,4),(5,4),(4,5),(5,5))
    home=min(access,key=lambda p:(abs(pos[0]-p[0])+abs(pos[1]-p[1]),p))
    distance=abs(pos[0]-home[0])+abs(pos[1]-home[1])
    cargo=[item for item in ('WOOL','FERTILIZER') if inv.get(item,0)]
    if cargo and step%24 >= (22 if step//24==29 else 23)-distance:
        return _v219_walk(pos,home) or ['PLACE',cargo[0],inv[cargo[0]]]
    missing=sum(not(isinstance(farm['tiles'][y][x],dict) and farm['tiles'][y][x].get('animal')=='SHEEP') for x,y in targets)
    if missing and not inv.get('SHEEP',0) and private['shed'].get('SHEEP',0):
        return _v219_walk(pos,home) or ['PICKUP','SHEEP',min(missing,private['shed']['SHEEP'])]
    hungry=sum(not(isinstance(farm['tiles'][y][x],dict) and farm['tiles'][y][x].get('fed_today')) for x,y in targets)
    if hungry and not inv.get('WHEAT',0) and private['shed'].get('WHEAT',0):
        return _v219_walk(pos,home) or ['PICKUP','WHEAT',min(hungry,private['shed']['WHEAT'])]
    tasks=[]
    for target in targets:
        x,y=target;tile=farm['tiles'][y][x];command=None
        if tile is None:command=['BUILD_PASTURE']
        elif isinstance(tile,dict) and tile.get('kind')=='WEED':command=['DIG']
        elif isinstance(tile,dict) and tile.get('kind')=='PASTURE' and not tile.get('animal'):
            if inv.get('SHEEP',0):command=['PLACE','SHEEP']
        elif isinstance(tile,dict) and tile.get('animal')=='SHEEP':
            if not tile['fed_today'] and inv.get('WHEAT',0):command=['FEED']
            elif not tile['cared_today']:command=['CARE']
            elif tile['yield_units']:command=['HARVEST']
            elif tile['fertilizer_available']:command=['COLLECT_FERTILIZER']
        if command:tasks.append((abs(pos[0]-x)+abs(pos[1]-y),targets.index(target),target,command))
    if tasks:
        _,_,target,command=min(tasks);return _v219_walk(pos,target) or command
    if cargo:return _v219_walk(pos,home) or ['PLACE',cargo[0],inv[cargo[0]]]
    return ['PASS']

def _v234_rescue(obs,action,state):
    if not state['workers'] or int(obs['step'])%24>14:return action
    orders=action.get('market',[])
    if len(orders)>=MAX_ORDERS:return action
    if any(o and (o[0] in ('HIRE','BUY_LAND','BUY_ANIMAL','BUY_PRODUCT','BUY_SEED') or (len(o)>1 and o[1]=='WHEAT')) for o in orders):return action
    farm=obs['farms'][obs['player']];private=obs['private'];hungry=carried=0
    commands=[action.get('farmer') or ['PASS']]+list(action.get('hands') or [])
    for actor,targets in state['workers'].items():
        command=commands[actor]
        if command==['FEED'] or command[:2]==['PICKUP','WHEAT']:return action
        carried+=private['inventories'][actor].get('WHEAT',0)
        hungry+=sum(isinstance(farm['tiles'][y][x],dict) and farm['tiles'][y][x].get('animal')=='SHEEP' and not farm['tiles'][y][x].get('fed_today') for x,y in targets)
    stock=projected_shed(action,FarmView(obs))
    shortage=hungry-carried-stock.get('WHEAT',0)
    if not 0<shortage<=6 or state.get('rescue_today',0)+shortage>6:return action
    quote=int(obs['market']['prices']['WHEAT'])
    if quote<1 or farm['money']<1000+shortage*(quote+10) or sum(stock.values())+shortage>100:return action
    result=copy.deepcopy(action);result['market'].append(['BUY_PRODUCT','WHEAT',shortage])
    state['rescue_today']=state.get('rescue_today',0)+shortage
    _V233_REPORT['sheep_rescue_feed_requests']+=shortage
    return result

def agent(observation,configuration=None):
    action=_V233_PARENT(observation,configuration)
    step=int(observation['step']);player=int(observation['player']);day=step//24
    state=_V233_STATES.get(player)
    if state is None or step<=state['last_step']:
        state={'last_step':step,'day':-1,'workers':{},'work':{},'credit':{'WOOL':0,'FERTILIZER':0}}
        _V233_STATES[player]=state
    state['last_step']=step
    if configuration is not None and any(configuration.get(k,v)!=v for k,v in
        (('boardSize',10),('turnsPerDay',24),('shedCapacity',100),('maxMarketOrdersPerTurn',10))):return action
    if day<12:return action
    farm=observation['farms'][player];private=observation['private']
    if state['day']!=day:state['day']=day;state['workers']={};state['work']={};state['rescue_today']=0
    for actor,previous in state['work'].items():
        if previous['step']!=step-1 or actor>=len(private['inventories']):continue
        item={'HARVEST':'WOOL','COLLECT_FERTILIZER':'FERTILIZER'}.get(previous['command'][0])
        if item:
            gained=max(0,private['inventories'][actor].get(item,0)-previous['inventory'].get(item,0))
            state['credit'][item]+=gained
            _V233_REPORT['sheep_wool_harvested' if item=='WOOL' else 'sheep_fert_collected']+=gained
    pending=state.pop('pending',None)
    if pending:
        funded='SE' in farm['unlocked_quadrants'] and (not pending['initial'] or private['shed'].get('SHEEP',0)>=6)
        if not funded:_V233_REPORT['sheep_purchase_shortfalls']+=1
        elif len(farm['hands'])<pending['first']+1:_V233_REPORT['sheep_hire_shortfalls']+=1
        else:
            for i in range(2):state['workers'][pending['first']+i]=[(x,5+i) for x in range(5,8)]
            _V233_REPORT['sheep_workers_confirmed']+=2
            if pending['initial']:state['committed']=True;_V233_REPORT['sheep_committed']+=1
    action=_v233_request(observation,action,state,_IMPL.chassis.players[player])
    if not state.get('committed'):return action
    result=copy.deepcopy(action)
    commands=[result.get('farmer') or ['PASS']]+list(result.get('hands') or [])
    commands += [['PASS'] for _ in range(len(farm['hands'])+1-len(commands))]
    state['work']={}
    for actor,targets in state['workers'].items():
        command=_v233_worker(observation,actor,targets);commands[actor]=command
        state['work'][actor]={'step':step,'command':command,'inventory':dict(private['inventories'][actor])}
    result['farmer'],result['hands']=commands[0],commands[1:]
    result=_v234_rescue(observation,result,state)
    stock=projected_shed(result,FarmView(observation))
    for item in ('WOOL','FERTILIZER'):
        scheduled=sum(int(o[2]) for o in result['market'] if o[:2]==['SELL',item])
        count=min(state['credit'][item],max(0,stock.get(item,0)-scheduled))
        if count and len(result['market'])<MAX_ORDERS:
            result['market'].append(['SELL',item,count]);state['credit'][item]-=count
            _V233_REPORT['sheep_extra_wool_sales' if item=='WOOL' else 'sheep_extra_fert_sales']+=count
    return result

_R46_SHEEP_AGENT=agent
_R46_SHADOW_PARENT=_shadow_terminal
_R46_REPORT={}
def _shadow_terminal(obs,config):
    if _V233_STATES.get(int(obs['player']),{}).get('committed'):return None
    return _R46_SHADOW_PARENT(obs,config)
del agent
def agent(observation,configuration=None):
    try:
        if int(observation.get('step',-1))==0:
            for k in _V233_REPORT:_V233_REPORT[k]=0
        result=_R46_SHEEP_AGENT(observation,configuration)
    except Exception:
        _R46_REPORT['sheep_overlay_errors']=_R46_REPORT.get('sheep_overlay_errors',0)+1
        result={'farmer':['PASS'],'hands':[],'market':[]}
    _R46_REPORT.update(getattr(_V233_PARENT,'telemetry',{}))
    _R46_REPORT.update(_V233_REPORT)
    return result
agent.telemetry=_R46_REPORT
agent=globals().pop('agent')

# EXP182: finite-harvest wheat/carrot input planner; original adaptation.
_R51_INPUT_PARENT=agent
_R51_INPUT_STATES={}
_R51_INPUT_REPORT={}
_R51_INPUT_MAX_WORKERS=2
_R51_INPUT_CROPS={'WHEAT':(2,4,6),'CARROT':(2,3,4)}

def _r51_input_forecast(obs,route,expected):
    step=int(obs['step']);day=step//24;farm=obs['farms'][obs['player']]
    pos=[list(farm['farmer'])]+[list(p) for p in farm['hands'][:expected]];targets={}
    for y,line in enumerate(farm['tiles']):
        for x,tile in enumerate(line):
            if not isinstance(tile,dict) or tile.get('crop') not in _R51_INPUT_CROPS:continue
            item=tile['crop'];first,last,cap=_R51_INPUT_CROPS[item]
            if 1<=day-tile['planted_day']<last:
                targets[(x,y)]={'crop':item,'birth':tile['planted_day'],'yield':tile['yield_units'],
                    'until':tile.get('fertilized_until_day',-1),'watered':tile.get('watered_today',False),'water':[],'harvest':None,'first':first,'last':last,'cap':cap}
    access=((4,4),(5,4),(4,5),(5,5));seen=set()
    # Native continuation ends before the reactive terminal closure planner.
    for t in range(step,min(712,(day+4)*24)):
        tape=_IMPL.chassis.routes[2 if t>=648 else route];a=tape[t]
        for actor,c in enumerate([a.get('farmer') or ['PASS'],*(a.get('hands') or [])][:len(pos)]):
            if not c:continue
            xy=tuple(pos[actor]);target=targets.get(xy)
            if target is not None and target['harvest'] is None:
                if c[0]=='WATER' and (t//24,xy) not in seen:
                    seen.add((t//24,xy))
                    if not(t//24==day and target['watered']) and target['first']<=t//24-target['birth']<=target['last']:target['water'].append(t)
                if c[0]=='HARVEST':target['harvest']=t
            if c[0] in MOVES:
                dx,dy=MOVES[c[0]];pos[actor]=[max(0,min(9,pos[actor][0]+dx)),max(0,min(9,pos[actor][1]+dy))]
        for o in a.get('market',[]):
            if o and o[0]=='HIRE':
                counts={p:sum(tuple(q)==p for q in pos) for p in access}
                pos.append(list(min(access,key=lambda p:(counts[p],access.index(p)))))
        if (t+1)%24==0:pos=[[4,4]]
    return targets

def _r51_input_gain(target,arrival,day):
    if target['harvest'] is None or target['harvest']<=arrival:return 0
    extra=sum(arrival<t<=target['harvest'] and day<=t//24<=day+2 and t//24>target['until'] for t in target['water'])
    baseline=target['yield']+sum(2 if t//24<=target['until'] else 1 for t in target['water'])
    return max(0,min(extra,target['cap']-baseline))

def _r51_input_path(obs,targets,action,index):
    step=int(obs['step']);day=step//24
    ready,start=_r62_input_start(obs,action,index)
    prices={p:max(1,int(obs['market']['prices'][p])-2) for p in ('WHEAT','CARROT')}
    fertilizer=max(1,_r37_market_price('FERTILIZER',obs['market']['inventory']['FERTILIZER']-16)+2)
    # Tuple: penalized value, gross value, next free turn, position, path, used,
    # wheat units, carrot units. No state reads from the rival's private farm.
    beam=[(0,0,ready,start,(),frozenset(),0,0)]
    best=None
    for depth in range(8):
        expanded=[]
        for score,gross,now,pos,path,used,wheat,carrot in beam:
            for xy,target in targets.items():
                if xy in used:continue
                arrival=now+abs(pos[0]-xy[0])+abs(pos[1]-xy[1])
                if arrival>=day*24+23:continue
                gain=_r51_input_gain(target,arrival,day)
                if not gain:continue
                item=target['crop'];new_gross=gross+gain*prices[item]
                new_path=path+((xy[0],xy[1],item,target['birth']),)
                expanded.append((new_gross-1.5*fertilizer*len(new_path),new_gross,arrival+1,xy,new_path,
                                 used|{xy},wheat+(gain if item=='WHEAT' else 0),carrot+(gain if item=='CARROT' else 0)))
        if not expanded:break
        expanded.sort(key=lambda s:(-s[0],-s[1],s[2],s[4]))
        beam=expanded[:8]
        if depth>=2:
            candidate=beam[0]
            if best is None or (-candidate[0],-candidate[1],candidate[2],candidate[4])<(-best[0],-best[1],best[2],best[4]):best=candidate
    if best is None:return [],{'WHEAT':0,'CARROT':0}
    return list(best[4]),{'WHEAT':best[6],'CARROT':best[7]}

def _r51_input_control(obs,action,state):
    step=int(obs['step']);day=step//24;hour=step%24;player=int(obs['player']);farm=obs['farms'][player];private=obs['private']
    native=_IMPL.chassis.players[player]
    if state.get('day')!=day:state.update(day=day,workers={},pending=None,placed=[])
    for x,y in state['placed']:
        tile=farm['tiles'][y][x]
        if isinstance(tile,dict) and tile.get('fertilized_until_day',-1)>=day+2:_R51_INPUT_REPORT['input_confirmed_applications']+=1
        else:_R51_INPUT_REPORT['input_application_errors']+=1
    state['placed']=[]
    if state.get('pending'):
        pending=state.pop('pending')
        for actor,plan in pending.items():
            if len(farm['hands'])>=actor:state['workers'][actor]=plan;_R51_INPUT_REPORT['input_confirmed_hires']+=1
            else:_R51_INPUT_REPORT['input_hire_errors']+=1
    if state['workers']:
        changed=copy.deepcopy(action)
        for actor,plan in state['workers'].items():
            inv=private['inventories'][actor];pos=tuple(farm['hands'][actor-1]);cmd=['PASS']
            if not plan['loaded']:
                stock=projected_shed(changed,FarmView(obs));q=min(plan['quantity'],max(0,stock.get('FERTILIZER',0)))
                if q and _shed_adjacent(pos,10):
                    cmd=['PICKUP','FERTILIZER',q];plan['loaded']=True;_R51_INPUT_REPORT['input_loaded_units']+=q
                    if q<plan['quantity']:_R51_INPUT_REPORT['input_stock_shortfalls']+=plan['quantity']-q
            elif inv.get('FERTILIZER',0):
                while plan['path']:
                    x,y,crop,birth=plan['path'][0];tile=farm['tiles'][y][x]
                    if not isinstance(tile,dict) or tile.get('crop')!=crop or tile.get('planted_day')!=birth or tile.get('fertilized_until_day',-1)>=day+2:
                        plan['path'].pop(0);continue
                    cmd=_v219_walk(pos,(x,y)) or ['FERTILIZE']
                    if cmd==['FERTILIZE']:state['placed'].append((x,y));plan['path'].pop(0);_R51_INPUT_REPORT['input_application_requests']+=1
                    break
            changed['hands'][actor-1]=cmd
        return changed
    if hour not in (1,2,3) or not 12<=day<=28:return action
    planned=_v219_native_day(native,day);expected=max(len(a.get('hands',[])) for a in planned)
    if any(o and o[0]=='HIRE' for a in planned[hour:] for o in a.get('market',[])) or native['pending']:return action
    parents=[_V219_STATES.get(player,{}),_V233_STATES.get(player,{})]
    # A parent may retry after a full market queue; its headcount must remain native.
    if day in (12,18) or any(p.get('committed') and p.get('requested_day')!=day for p in parents):return action
    if any(p.get('pending') for p in parents) or any(o and o[0]=='HIRE' for o in action.get('market',[])):return action
    owned=set(range(1,expected+1))
    for p in parents:
        actors=set(p.get('workers',{}))
        if owned&actors:return action
        owned|=actors
    if owned!=set(range(1,len(farm['hands'])+1)):return action
    targets=_r51_input_forecast(obs,native['route'],expected);plans=[];total_q=0;total_cost=0;all_units={'WHEAT':0,'CARROT':0}
    stock=projected_shed(action,FarmView(obs));purchases=sum(max(0,int(o[2])) for o in action.get('market',[]) if len(o)>2 and o[0] in ('BUY_PRODUCT','BUY_ANIMAL'))
    # Units act before market orders. Preserve the native next-turn pickup,
    # after the current parent's actual sales/purchases, before buying tour inputs.
    available=max(0,stock.get('FERTILIZER',0))
    for o in action.get('market',[]):
        if len(o)>=3 and o[:2]==['SELL','FERTILIZER']:available=max(0,available-max(0,int(o[2])))
        elif len(o)>=3 and o[:2]==['BUY_PRODUCT','FERTILIZER']:available+=max(0,int(o[2]))
    next_native=planned[hour+1];native_pickups=sum(max(0,int(c[2]) if len(c)>2 else 1) for c in [next_native.get('farmer') or ['PASS'],*(next_native.get('hands') or [])] if len(c)>1 and c[:2]==['PICKUP','FERTILIZER'])
    topup=max(0,native_pickups-available)
    plans,total_q,total_cost,all_units=_r68_joint_plans(obs,action,targets,stock,purchases,topup)
    if not plans:return action
    state['pending']={len(farm['hands'])+1+i:plan for i,plan in enumerate(plans)}
    _R51_INPUT_REPORT['input_hire_requests']+=len(plans);_R51_INPUT_REPORT['input_purchase_requests']+=total_q+topup
    _R51_INPUT_REPORT['input_forecast_wheat']+=all_units['WHEAT'];_R51_INPUT_REPORT['input_forecast_carrot']+=all_units['CARROT']
    changed=copy.deepcopy(action);changed['market'] += [['BUY_PRODUCT','FERTILIZER',total_q+topup]]+[['HIRE'] for _ in plans];return changed

def agent(observation,configuration=None):
    try:
        step=int(observation['step']);player=int(observation['player']);state=_R51_INPUT_STATES.get(player)
        if state is None or step<=state['step']:
            state=_R51_INPUT_STATES[player]={'step':-1}
            _R51_INPUT_REPORT.update(input_hire_requests=0,input_confirmed_hires=0,input_hire_errors=0,input_purchase_requests=0,
                input_loaded_units=0,input_stock_shortfalls=0,input_application_requests=0,input_confirmed_applications=0,
                input_application_errors=0,input_errors=0,input_forecast_wheat=0,input_forecast_carrot=0)
        state['step']=step;action=_R51_INPUT_PARENT(observation,configuration)
        if configuration is None or all(configuration.get(k,v)==v for k,v in [('boardSize',10),('turnsPerDay',24),('shedCapacity',100),('maxMarketOrdersPerTurn',10)]):
            action=_r51_input_control(observation,action,state)
        _R51_INPUT_REPORT.update(getattr(_R51_INPUT_PARENT,'telemetry',{}));return action
    except Exception:
        _R51_INPUT_REPORT['input_errors']=_R51_INPUT_REPORT.get('input_errors',0)+1
        return {'farmer':['PASS'],'hands':[],'market':[]}
agent.telemetry=_R51_INPUT_REPORT
agent=globals().pop('agent')

# EXP182: project the final hour's actual worker actions before automatic deposit.
_R51_WAREHOUSE_PARENT=agent
_R51_WAREHOUSE_REPORT={}

def _r51_close_warehouse(obs,action):
    step=int(obs['step']);day=step//24
    if step%24!=23 or not 12<=day<=28:return action
    # No speculative product purchase/worker count model: these hours abstain.
    if any(o and o[0] not in ('SELL',) for o in action.get('market',[])):return action
    farm,private=_PLANNER_NS['_clone_state'](obs['farms'][obs['player']],obs['private'])
    commands=[action.get('farmer') or ['PASS'],*(action.get('hands') or [])]
    demand={}
    for c in commands:
        if len(c)>1 and c[0]=='PLANT':demand[c[1]]=demand.get(c[1],0)+1
    blocked={k for k,q in demand.items() if q>private['seeds'].get(k,0)}
    for actor,c in enumerate(commands[:len(private['inventories'])]):
        if len(c)>1 and c[0]=='PLANT' and c[1] in blocked:c=['PASS']
        _PLANNER_NS['_apply_unit_action'](farm,private,actor,c,10,day,24,100)
    post=dict(private['shed'])
    for o in action.get('market',[]):
        if len(o)>=3 and o[0]=='SELL':post[o[1]]=max(0,post.get(o[1],0)-max(0,int(o[2])))
    needed=sum(post.values())+sum(max(0,q) for inv in private['inventories'] for q in inv.values())-100
    if needed<=0:return action
    result=copy.deepcopy(action);orders=result['market']
    # Grain and fertilizer have native input obligations; other products do not.
    # Additional commodity sales are bounded by actual post-action physical stock.
    for item in sorted((p for p in PRODUCTS if p not in ('WHEAT','FERTILIZER')),key=lambda p:-obs['market']['prices'].get(p,0)):
        qty=min(needed,post.get(item,0))
        if not qty:continue
        existing=next((o for o in orders if len(o)>=3 and o[:2]==['SELL',item]),None)
        if existing is not None:existing[2]=max(0,int(existing[2]))+qty
        elif len(orders)<10:orders.append(['SELL',item,qty])
        else:continue
        needed-=qty;post[item]-=qty;_R51_WAREHOUSE_REPORT['warehouse_extra_sales']+=qty
        if needed<=0:break
    if needed>0:
        native=_IMPL.chassis.players[int(obs['player'])];reserve=0
        for t in range(step+1,719):
            future=_IMPL.chassis.routes[2 if t>=648 else native['route']][t]
            for c in [future.get('farmer') or ['PASS'],*(future.get('hands') or [])]:
                if len(c)>1 and c[:2]==['PICKUP','WHEAT']:reserve+=max(0,int(c[2]) if len(c)>2 else 1)
            if any(len(o)>1 and o[:2]==['BUY_PRODUCT','WHEAT'] for o in future.get('market',[])):break
        incoming=sum(max(0,inv.get('WHEAT',0)) for inv in private['inventories'])
        others=sum(q for p,q in post.items() if p!='WHEAT')+sum(max(0,q) for inv in private['inventories'] for p,q in inv.items() if p!='WHEAT')
        # Even if every other carried item deposits first, this grain reserve fits.
        qty=min(needed,post.get('WHEAT',0),max(0,post.get('WHEAT',0)+incoming-reserve)) if 100-others>=reserve else 0
        existing=next((o for o in orders if len(o)>=3 and o[:2]==['SELL','WHEAT']),None)
        if qty and (existing is not None or len(orders)<10):
            if existing is not None:existing[2]=max(0,int(existing[2]))+qty
            else:orders.append(['SELL','WHEAT',qty])
            needed-=qty;_R51_WAREHOUSE_REPORT['warehouse_extra_sales']+=qty
    _R51_WAREHOUSE_REPORT['warehouse_projected_unresolved']+=max(0,needed)
    if result!=action:_R51_WAREHOUSE_REPORT['warehouse_changed_turns']+=1
    return result

def agent(observation,configuration=None):
    result=_R51_WAREHOUSE_PARENT(observation,configuration)
    try:
        if int(observation['step'])==0:_R51_WAREHOUSE_REPORT.update(warehouse_changed_turns=0,warehouse_extra_sales=0,warehouse_projected_unresolved=0,warehouse_errors=0)
        if configuration is None or all(configuration.get(k,v)==v for k,v in [('boardSize',10),('turnsPerDay',24),('shedCapacity',100),('maxMarketOrdersPerTurn',10)]):result=_r51_close_warehouse(observation,result)
    except Exception:_R51_WAREHOUSE_REPORT['warehouse_errors']=_R51_WAREHOUSE_REPORT.get('warehouse_errors',0)+1
    _R51_WAREHOUSE_REPORT.update(getattr(_R51_WAREHOUSE_PARENT,'telemetry',{}));return result
agent.telemetry=_R51_WAREHOUSE_REPORT
agent=globals().pop('agent')

from itertools import permutations as _r53_permutations
_R53_LABOR_REPORT=dict(labor_requests=0,labor_hires_avoided=0,labor_spawn_errors=0,labor_confirmed=0,labor_day26=0,labor_day27=0,labor_day28=0)

def _r53_labor_assignment(obs,action,fertilizer):
    step=int(obs['step']);day=step//24;farm=obs['farms'][obs['player']]
    if day not in (26,27,28) or step%24>2:return None
    # Do not preempt a later price-gated fertilizer request with a smaller unfertilized team.
    if day==27 and not fertilizer:return None
    count=3 if fertilizer else 2
    positions=[list(farm['farmer'])]+[list(p) for p in farm['hands']]
    for i,c in enumerate([action.get('farmer') or ['PASS'],*(action.get('hands') or [])][:len(positions)]):
        if c and c[0] in MOVES:
            dx,dy=MOVES[c[0]];positions[i]=[max(0,min(9,positions[i][0]+dx)),max(0,min(9,positions[i][1]+dy))]
    access=((4,4),(5,4),(4,5),(5,5));spawns=[]
    native_hires=sum(bool(o) and o[0]=='HIRE' for o in action.get('market',[]))
    for i in range(native_hires+count):
        chosen=min(access,key=lambda p:(sum(tuple(q)==p for q in positions),access.index(p)));positions.append(list(chosen))
        if i>=native_hires:spawns.append(chosen)
    groups=(((5,5),(6,5),(7,5),(8,5)),((9,5),(9,6),(8,6)),((5,6),(6,6),(7,6))) if fertilizer else (tuple((x,5) for x in range(5,10)),tuple((x,6) for x in range(5,10)))
    choices=[];remaining=23-step%24
    for assignment in _r53_permutations(groups):
        costs=[]
        for start,path in zip(spawns,assignment):
            distance=abs(start[0]-path[0][0])+abs(start[1]-path[0][1])
            distance+=sum(abs(a[0]-b[0])+abs(a[1]-b[1]) for a,b in zip(path,path[1:]))
            distance+=min(abs(path[-1][0]-x)+abs(path[-1][1]-y) for x,y in access)
            costs.append(distance+(3 if fertilizer else 2)*len(path)+1+int(fertilizer))
        if max(costs)<=remaining:choices.append((max(costs),sum(costs),assignment))
    if not choices:return None
    _,_,assignment=min(choices)
    return dict(paths=assignment,spawns=spawns,remaining=remaining,workers=count,fertilizer=fertilizer)

_R53_LABOR_PARENT=agent
def agent(observation,configuration=None):
    if isinstance(observation,dict) and observation.get('step')==0:
        for k in _R53_LABOR_REPORT:_R53_LABOR_REPORT[k]=0
    result=_R53_LABOR_PARENT(observation,configuration)
    _R53_LABOR_COMBINED.update(getattr(_R53_LABOR_PARENT,'telemetry',{}));_R53_LABOR_COMBINED.update(_R53_LABOR_REPORT)
    return result
_R53_LABOR_COMBINED={}
agent.telemetry=_R53_LABOR_COMBINED
agent=globals().pop('agent')

# EXP193: deterministic HIRE spawn after native unit actions, then next-turn pickup.
def _r62_input_start(obs,action,index):
    farm=obs['farms'][obs['player']]
    positions=[list(farm['farmer'])]+[list(p) for p in farm['hands']]
    for actor,c in enumerate([action.get('farmer') or ['PASS'],*(action.get('hands') or [])][:len(positions)]):
        if c and c[0] in MOVES:
            dx,dy=MOVES[c[0]]
            positions[actor]=[max(0,min(9,positions[actor][0]+dx)),max(0,min(9,positions[actor][1]+dy))]
    access=((4,4),(5,4),(4,5),(5,5))
    native_hires=sum(bool(o) and o[0]=='HIRE' for o in action.get('market',[]))
    for _ in range(native_hires+index+1):
        chosen=min(access,key=lambda p:(sum(tuple(q)==p for q in positions),access.index(p)))
        positions.append(list(chosen))
    return int(obs['step'])+2,chosen

agent=globals().pop('agent')

def _r68_joint_plans(obs,action,targets,stock,purchases,topup):
    farm=obs['farms'][obs['player']];choices=[]
    for mode,first_crop in enumerate((None,'WHEAT','CARROT')):
        remaining=dict(targets);plans=[];total_q=0;total_cost=0;total_value=0
        all_units={'WHEAT':0,'CARROT':0}
        for i in range(_R51_INPUT_MAX_WORKERS):
            subset={xy:t for xy,t in remaining.items() if t['crop']==first_crop} if i==0 and first_crop else remaining
            path,units=_r51_input_path(obs,subset,action,i);q=len(path)
            if q<3 or len(action.get('market',[]))+2+i>10 or sum(stock.values())+purchases+total_q+q+topup>95:break
            quote=_r37_market_price('FERTILIZER',obs['market']['inventory']['FERTILIZER']-total_q-q-topup)
            cost=(q+(topup if i==0 else 0))*(quote+2)+_v219_fib(int(farm['hires_today'])+i)
            value=sum(n*max(1,_r37_market_price(item,obs['market']['inventory'][item]+all_units[item]+n)-2) for item,n in units.items())
            if value<1.5*cost+50 or farm['money']<total_cost+cost+3000:break
            plans.append({'path':path,'quantity':q,'loaded':False});total_q+=q;total_cost+=cost;total_value+=value
            for item,n in units.items():all_units[item]+=n
            for x,y,_,_ in path:remaining.pop((x,y),None)
        score=(total_value-total_cost,total_value,-total_cost,-len(plans),-mode)
        choices.append((score,plans,total_q,total_cost,all_units))
    _,plans,total_q,total_cost,all_units=max(choices,key=lambda v:v[0])
    return plans,total_q,total_cost,all_units

agent=globals().pop('agent')

_R70_STATES={}
_R70_REPORT={}

def _r70_parent_fert_qty(obs,action,planned,offset):
    stock=dict(projected_shed(action,FarmView(obs)))
    for order in action.get('market',[]):
        if len(order)<3:continue
        op,item,quantity=order[:3];quantity=max(0,int(quantity))
        if op=='SELL':stock[item]=max(0,stock.get(item,0)-quantity)
        elif op in ('BUY_PRODUCT','BUY_ANIMAL'):
            stock[item]=stock.get(item,0)+min(quantity,max(0,100-sum(stock.values())))
    next_action=planned[offset+1] if offset+1<len(planned) else {}
    commands=[next_action.get('farmer') or ['PASS'],*(next_action.get('hands') or [])]
    native_need=sum(max(0,int(c[2]) if len(c)>2 else 1) for c in commands if len(c)>1 and c[:2]==['PICKUP','FERTILIZER'])
    quantity=max(10,10+native_need-max(0,stock.get('FERTILIZER',0)))
    if quantity>max(0,100-sum(stock.values())):
        _R70_REPORT['parent_input_capacity_declines']+=1
        return 10
    if quantity>10:
        _R70_REPORT['parent_input_guard_turns']+=1
        _R70_REPORT['parent_input_guard_extra_units']+=quantity-10
    return quantity

def _r70_before(obs):
    player=int(obs['player']);step=int(obs['step']);state=_R70_STATES.get(player)
    if state is None or step<=state['step']:
        state=_R70_STATES[player]={'step':-1,'pending':[],'roles':set()}
        _R70_REPORT.update(parent_input_guard_turns=0,parent_input_guard_extra_units=0,
            parent_input_requests=0,parent_input_confirmed=0,parent_input_shortfalls=0,
            parent_input_errors=0,parent_input_capacity_declines=0)
    state['step']=step
    for request in state['pending']:
        actor,quantity,old=request
        actual=max(0,int(obs['private']['inventories'][actor].get('FERTILIZER',0))-old)
        _R70_REPORT['parent_input_confirmed']+=min(quantity,actual)
        _R70_REPORT['parent_input_shortfalls']+=max(0,quantity-actual)
    state['pending']=[]
    return state

def _r70_after(obs,action,state):
    player=int(obs['player']);day=int(obs['step'])//24
    for actor,role in _V219_STATES.get(player,{}).get('workers',{}).items():
        if not role.get('needs_fertilizer'):continue
        key=(day,actor);inv=obs['private']['inventories'][actor].get('FERTILIZER',0)
        if key not in state['roles']:
            state['roles'].add(key)
            desired=role.get('fertilizer_quantity',10 if role['kind']=='fertilizer' else 5)
            if role.get('loaded') and not role.get('pickup_requested') and inv<desired:
                _R70_REPORT['parent_input_shortfalls']+=desired-inv
        command=action.get('hands',[])[actor-1] if actor<=len(action.get('hands',[])) else ['PASS']
        if len(command)>1 and command[:2]==['PICKUP','FERTILIZER']:
            quantity=max(0,int(command[2]) if len(command)>2 else 1)
            _R70_REPORT['parent_input_requests']+=quantity
            state['pending'].append((actor,quantity,int(inv)))

_R70_PARENT=agent

def agent(observation,configuration=None):
    state=None
    try:state=_r70_before(observation)
    except Exception:_R70_REPORT['parent_input_errors']=_R70_REPORT.get('parent_input_errors',0)+1
    result=_R70_PARENT(observation,configuration)
    try:
        if state is not None:_r70_after(observation,result,state)
    except Exception:_R70_REPORT['parent_input_errors']=_R70_REPORT.get('parent_input_errors',0)+1
    _R70_REPORT.update(getattr(_R70_PARENT,'telemetry',{}))
    return result

agent.telemetry=_R70_REPORT
agent=globals().pop('agent')

def _r79_tomato_fertilizer_worthwhile(obs,action):
    if obs['market']['prices']['FERTILIZER']<=30:return True
    farm=obs['farms'][obs['player']];day=int(obs['step'])//24;bonus=0
    for y in (5,6):
        for x in range(5,10):
            tile=farm['tiles'][y][x]
            if not isinstance(tile,dict) or tile.get('crop')!='TOMATO':continue
            birth=tile['planted_day'];until=tile.get('fertilized_until_day',-1)
            bonus+=sum(until<d and 8<=d+1-birth<=11 for d in range(day,day+3))
    if not bonus:return False
    inventory=obs['market']['inventory']
    price=max(1,_r37_market_price('TOMATO',inventory['TOMATO']+bonus+10)-2)
    fertilizer=max(1,_r37_market_price('FERTILIZER',inventory['FERTILIZER']-10)+2)
    native_hires=sum(bool(o) and o[0]=='HIRE' for o in action.get('market',[]))
    extra_labor=_v219_fib(int(farm['hires_today'])+native_hires+3)
    return bonus*price>=2*(10*fertilizer+extra_labor)+100

agent=globals().pop('agent')

# EXP216: original adaptation of economic feed and fertilizer-sale concepts.
# Conceptual credit: Steven Lee Hans, "Lord Momo Returns", September12 snapshot.
_R85_FEED = True
_R85_FERT = True
_R85_PARENT = agent
_R85_STATES = {}
_R85_REPORT = {}

def _r85_feed(obs, action):
    step=int(obs['step']);day=step//24
    if not 10<=day<=28 or step%24>21:return action
    player=int(obs['player']);native=_IMPL.chassis.players[player]
    tape=_v219_native_day(native,day)
    expected=max(len(a.get('hands',[])) for a in tape)
    farm=obs['farms'][player];positions=[farm['farmer'],*farm['hands']]
    commands=[action.get('farmer') or ['PASS'],*(action.get('hands') or [])]
    prices=obs['market']['prices'];changed=False
    for actor,command in enumerate(commands[:expected+1]):
        if command!=['FEED'] or actor>=len(positions):continue
        tile=_tile_at(farm['tiles'],positions[actor])
        if not isinstance(tile,dict) or tile.get('animal') not in ('GOOSE','COW','SHEEP'):continue
        if tile.get('fed_today') or int(tile.get('consecutive_unfed',0))!=0:continue
        if int(obs['private']['inventories'][actor].get('WHEAT',0))<=0:continue
        item={'GOOSE':'EGG','COW':'MILK','SHEEP':'WOOL'}[tile['animal']]
        bonus=_r88_feed_bonus_cost(tile,day)
        if bonus*(float(prices[item])+5)*1.25>=float(prices['WHEAT']):continue
        if not _r86_next_feed(obs,positions[actor]):continue
        commands[actor]=['PASS'];changed=True
        _R85_REPORT['feed_skips']+=1
    if not changed:return action
    result=copy.deepcopy(action);result['farmer'],result['hands']=commands[0],commands[1:]
    return result

def _r85_reserve(obs, state):
    step=int(obs['step']);player=int(obs['player']);native=_IMPL.chassis.players[player]
    route=native['route'];key=(route,step)
    cache=state.setdefault('native_reserves',{})
    if route not in cache:
        # Backward recurrence preserves field-before-market order within a turn.
        reserve=[0]*720
        for t in range(718,-1,-1):
            a=_IMPL.chassis.routes[2 if t>=648 else route][t]
            pickup=sum(max(0,int(c[2]) if len(c)>2 else 1) for c in [a.get('farmer') or ['PASS'],*(a.get('hands') or [])] if len(c)>1 and c[:2]==['PICKUP','FERTILIZER'])
            purchase=sum(max(0,int(o[2])) for o in a.get('market',[]) if len(o)>2 and o[:2]==['BUY_PRODUCT','FERTILIZER'])
            reserve[t]=pickup+max(0,reserve[t+1]-purchase)
        cache[route]=reserve
    dedicated=0
    for parent in (_V219_STATES.get(player,{}),_V233_STATES.get(player,{})):
        for actor,role in parent.get('workers',{}).items():
            if not isinstance(role,dict) or not role.get('needs_fertilizer') or role.get('loaded'):continue
            desired=role.get('fertilizer_quantity',10 if role.get('kind')=='fertilizer' else 5)
            carried=obs['private']['inventories'][actor].get('FERTILIZER',0)
            dedicated+=max(0,desired-carried)
        pending=parent.get('pending') or {}
        if pending.get('fertilizer'):dedicated+=10
    inputs=_R51_INPUT_STATES.get(player,{})
    for actor,plan in {**inputs.get('workers',{}),**(inputs.get('pending') or {})}.items():
        if not plan.get('loaded'):dedicated+=max(0,int(plan['quantity']))
    return max(14,cache[route][min(719,step+1)]+dedicated)

def _r85_fertilizer(obs, action, state):
    step=int(obs['step']);day=step//24
    if not 6<=day<=28:return action
    market=action.get('market',[])
    if len(market)>=MAX_ORDERS or any(o and o[0]!='SELL' for o in market):return action
    stock=projected_shed(action,FarmView(obs))
    held=max(0,int(stock.get('FERTILIZER',0)))
    sold=sum(max(0,int(o[2])) for o in market if len(o)>2 and o[:2]==['SELL','FERTILIZER'])
    extra=held-sold-_r85_reserve(obs,state)
    if extra<=0:return action
    result=copy.deepcopy(action);result['market'].append(['SELL','FERTILIZER',extra])
    _R85_REPORT['fert_sale_turns']+=1;_R85_REPORT['fert_sale_units']+=extra
    return result

def agent(observation, configuration=None):
    result=_R85_PARENT(observation,configuration)
    try:
        step=int(observation['step']);player=int(observation['player'])
        state=_R85_STATES.get(player)
        if state is None or step<=state['step']:
            state=_R85_STATES[player]={'step':-1}
            _R85_REPORT.update(feed_skips=0,fert_sale_turns=0,fert_sale_units=0,economic_overlay_errors=0)
        state['step']=step
        if configuration is not None and any(configuration.get(k,v)!=v for k,v in [('boardSize',10),('turnsPerDay',24),('shedCapacity',100),('maxMarketOrdersPerTurn',10)]):return result
        if _R85_FEED:result=_r85_feed(observation,result)
        if _R85_FERT:result=_r85_fertilizer(observation,result,state)
        if step%24==23:result=_r51_close_warehouse(observation,result)
    except Exception:
        _R85_REPORT['economic_overlay_errors']=_R85_REPORT.get('economic_overlay_errors',0)+1
    _R85_REPORT.update(getattr(_R85_PARENT,'telemetry',{}))
    return result

agent.telemetry=_R85_REPORT
agent=globals().pop('agent')

# EXP217: planned next-day service is required before discretionary feed cuts.
_R86_FEED_CACHE = {}

def _r86_next_feed(obs, target):
    step=int(obs['step']);day=step//24
    if day==28:return True  # No second dawn follows before game termination.
    player=int(obs['player']);native=_IMPL.chassis.players[player]
    tomorrow=day+1;route=2 if tomorrow>=27 else native['route'];key=(route,tomorrow)
    if key not in _R86_FEED_CACHE:
        positions=[(4,4)];wheat=[0];access=((4,4),(5,4),(4,5),(5,5));feeds=set()
        for hour in range(24):
            a=_IMPL.chassis.routes[route][tomorrow*24+hour]
            commands=[a.get('farmer') or ['PASS'],*(a.get('hands') or [])]
            for actor,command in enumerate(commands[:len(positions)]):
                if not command:continue
                pos=positions[actor];op=command[0]
                if op in MOVES:
                    dx,dy=MOVES[op];positions[actor]=(max(0,min(9,pos[0]+dx)),max(0,min(9,pos[1]+dy)))
                elif command[:2]==['PICKUP','WHEAT'] and pos in access:
                    wheat[actor]+=max(0,int(command[2]) if len(command)>2 else 1)
                elif op=='FEED' and wheat[actor]>0:
                    wheat[actor]-=1
                    if hour<=21:feeds.add(pos)
                elif op=='DROP' and pos in access:wheat[actor]=0
                elif command[:2]==['PLACE','WHEAT'] and pos in access:
                    wheat[actor]=max(0,wheat[actor]-max(0,int(command[2]) if len(command)>2 else 1))
            for order in a.get('market',[]):
                if order and order[0]=='HIRE':
                    chosen=min(access,key=lambda p:(positions.count(p),access.index(p)))
                    positions.append(chosen);wheat.append(0)
        _R86_FEED_CACHE[key]=frozenset(feeds)
    return tuple(target) in _R86_FEED_CACHE[key]

agent=globals().pop('agent')

# EXP219: charge care credits only when this feeding decision can affect them.
_R88_PHASE = True
_R88_HORIZON = True
_R88_ANIMAL_DAYS = {'GOOSE': (4, 1), 'COW': (8, 2), 'SHEEP': (6, 3)}


def _r88_feed_bonus_cost(tile, day):
    first, interval = _R88_ANIMAL_DAYS[tile['animal']]
    first += int(tile['placed_day'])
    tomorrow = day + 1
    produces = tomorrow >= first and (tomorrow - first) % interval == 0
    pending = max(0, int(tile.get('pending_care_bonus', 0)))
    if _R88_PHASE and not produces:
        pending = 0  # It remains banked on non-production dawns.
    care = 1  # Conservative: charge one possible CARE even if not yet observed.
    if _R88_HORIZON:
        # Today's care is added AFTER tomorrow's production; its first possible
        # payout is a later production dawn, which must occur before game end.
        next_use = first
        if next_use <= tomorrow:
            next_use += ((tomorrow - next_use) // interval + 1) * interval
        if next_use > 29:
            care = 0
    return pending + care


agent = globals().pop('agent')

# EXP226: retain physical grain for two complete days before trimming a buy.
_R95_PARENT = agent
_R95_REPORT = {}
_R95_RESERVES = {}

def _r95_reserve(obs):
    step=int(obs['step']);player=int(obs['player'])
    native=_IMPL.chassis.players[player];route=native['route']
    key=(route,step)
    if key not in _R95_RESERVES:
        demand=6  # Physical buffer beyond every scheduled pickup and sale.
        for t in range(step+1,min(719,step+49)):
            a=_IMPL.chassis.routes[2 if t>=648 else route][t]
            for c in [a.get('farmer') or ['PASS'],*(a.get('hands') or [])]:
                if c[:2]==['PICKUP','WHEAT']:
                    demand+=max(0,int(c[2]) if len(c)>2 else 1)
            for o in a.get('market',[]):
                if len(o)>2 and o[:2]==['SELL','WHEAT']:
                    demand+=max(0,int(o[2]))
        _R95_RESERVES[key]=demand
    demand=_R95_RESERVES[key]
    # Reserve full feed for a possible southeast sheep commitment. Do not
    # rely on its future discretionary buy, eligibility, or existing cargo.
    if obs['town']['unlocked_shops'].count('YARN_STORE')>=2:
        demand+=6*len({t//24 for t in range(step+1,step+49) if t//24>=12})
    return demand

def _r95_replenish(obs,action):
    step=int(obs['step'])
    if not 10<=step//24<=11:return action
    orders=action.get('market') or []
    if not any(len(o)>2 and o[:2]==['BUY_PRODUCT','WHEAT'] and int(o[2])>0 for o in orders):return action
    # Preserve all same-turn grain trading/arbitrage sequences unchanged.
    if any(o[:2]==['SELL','WHEAT'] for o in orders):return action
    farm,private=_PLANNER_NS['_clone_state'](obs['farms'][obs['player']],obs['private'])
    commands=[action.get('farmer') or ['PASS'],*(action.get('hands') or [])]
    for actor,c in enumerate(commands[:len(private['inventories'])]):
        _PLANNER_NS['_apply_unit_action'](farm,private,actor,c,10,step//24,24,100)
    held=max(0,int(private['shed'].get('WHEAT',0)))
    reserve=_r95_reserve(obs);result=None;removed=0
    for i,o in enumerate(orders):
        if len(o)<3 or o[:2]!=['BUY_PRODUCT','WHEAT']:continue
        quantity=max(0,int(o[2]));retained=min(quantity,max(0,reserve-held))
        held+=retained
        if retained<quantity:
            if result is None:result=copy.deepcopy(action)
            result['market'][i][2]=retained  # Zero keeps every later order slot.
            removed+=quantity-retained
    if result is None:return action
    _R95_REPORT['replenishment_trim_turns']+=1
    _R95_REPORT['replenishment_trim_units']+=removed
    return result

def agent(observation,configuration=None):
    result=_R95_PARENT(observation,configuration)
    try:
        if int(observation['step'])==0:
            _R95_REPORT.update(replenishment_trim_turns=0,replenishment_trim_units=0,replenishment_errors=0)
        if configuration is not None and any(configuration.get(k,v)!=v for k,v in [('boardSize',10),('turnsPerDay',24),('shedCapacity',100),('maxMarketOrdersPerTurn',10)]):return result
        result=_r95_replenish(observation,result)
    except Exception:
        _R95_REPORT['replenishment_errors']=_R95_REPORT.get('replenishment_errors',0)+1
    _R95_REPORT.update(getattr(_R95_PARENT,'telemetry',{}))
    return result

agent.telemetry=_R95_REPORT
agent=globals().pop('agent')

# EXP231: protect inputs using funded current orders without unassigned cash padding from observed physical resources.
_R97_PARENT=agent
_R97_REPORT={}
_R97_LAST={}

def _r97_market_stock(shed,orders):
    stock=dict(shed);buys={};sales={}
    for index,order in enumerate(orders):
        if len(order)<3:continue
        op,item,n=order[:3];n=max(0,int(n))
        if op=='SELL':
            q=min(n,max(0,stock.get(item,0)));stock[item]=stock.get(item,0)-q;sales[index]=q
        elif op in ('BUY_PRODUCT','BUY_ANIMAL'):
            q=min(n,max(0,100-sum(stock.values())));stock[item]=stock.get(item,0)+q;buys[index]=q
    return stock,buys,sales

def _r97_delivery(stock,private,night):
    stock=dict(stock);lost={}
    if night:
        for inv in private['inventories']:
            for item,q in inv.items():
                q=max(0,int(q));take=min(q,max(0,100-sum(stock.values())))
                stock[item]=stock.get(item,0)+take
                if q>take:lost[item]=lost.get(item,0)+q-take
    return stock,lost

def _r97_budget(obs,orders):
    farm=obs['farms'][obs['player']];cost=0;hires=int(farm['hires_today'])
    # At most ten 100-unit purchases per opponent turn. The additional 1000
    # own units give an intentionally conservative upper bound on buy quotes.
    prices={p:_r37_market_price(p,obs['market']['inventory'][p]-2000) for p in ('WHEAT','FERTILIZER')}
    for order in orders:
        if not order:continue
        op=order[0]
        if op=='HIRE':cost+=_v219_fib(hires);hires+=1
        elif op=='BUY_LAND':cost+=4000
        elif len(order)>2:
            item=order[1];q=max(0,int(order[2]))
            if op=='BUY_PRODUCT':cost+=q*prices[item]
            elif op=='BUY_ANIMAL':cost+=q*{'GOOSE':300,'COW':400,'SHEEP':500}[item]
            elif op=='BUY_SEED':cost+=q*{'WHEAT':10,'CARROT':20,'TOMATO':50,'STRAWBERRY':100,'MELON':80}[item]
    return cost<=farm['money']  # No current sale proceeds are assumed.

def _r97_supply(obs,action):
    step=int(obs['step']);player=int(obs['player']);day=step//24
    if not 144<=step<695:return action
    native=_IMPL.chassis.players[player]
    future=_IMPL.chassis.routes[2 if step+1>=648 else native['route']][step+1]
    commands=[future.get('farmer') or ['PASS'],*(future.get('hands') or [])]
    following=_IMPL.chassis.routes[2 if step+2>=648 else native['route']][step+2]
    next_orders=future.get('market') or []
    prefund=0
    if len(next_orders)==10 and not any(o[:2] in (['BUY_PRODUCT','WHEAT'],['SELL','WHEAT']) for o in next_orders):
        later=[following.get('farmer') or ['PASS'],*(following.get('hands') or [])]
        demand=lambda cs:sum(max(0,int(c[2]) if len(c)>2 else 1) for c in cs if c[:2]==['PICKUP','WHEAT'])
        if demand(later):prefund=demand(commands)+demand(later)
    if not prefund and not any(c[:2]==['PICKUP','WHEAT'] for c in commands):return action
    orders=action.get('market') or []
    if len(orders)>10 or not _r97_budget(obs,orders):
        _R97_REPORT['supply_budget_declines']+=1;return action
    farm,private=_PLANNER_NS['_clone_state'](obs['farms'][player],obs['private'])
    for actor,c in enumerate([action.get('farmer') or ['PASS'],*(action.get('hands') or [])][:len(private['inventories'])]):
        _PLANNER_NS['_apply_unit_action'](farm,private,actor,c,10,day,24,100)
    positions=[tuple(farm['farmer']),*map(tuple,farm['hands'])];night=step%24==23
    access=((4,4),(5,4),(4,5),(5,5))
    if night:positions=[(4,4)]
    else:
        for order in orders:
            if order and order[0]=='HIRE':positions.append(min(access,key=lambda p:(positions.count(p),access.index(p))))
    need=sum(max(0,int(c[2]) if len(c)>2 else 1) for pos,c in zip(positions,commands) if pos in access and c[:2]==['PICKUP','WHEAT'])
    need=max(need,prefund)
    if not need:return action
    original_stock,original_buys,_=_r97_market_stock(private['shed'],orders)
    original_final,original_loss=_r97_delivery(original_stock,private,night)
    if original_final.get('WHEAT',0)>=need:return action
    result=copy.deepcopy(action);proposed=result['market'];blocked=False
    def project(candidate):
        stock,buys,sales=_r97_market_stock(private['shed'],candidate)
        final,loss=_r97_delivery(stock,private,night)
        safe=all(buys.get(i,0)>=q for i,q in original_buys.items()) and all(q<=original_loss.get(item,0) for item,q in loss.items())
        return final,sales,safe
    # Hold an existing grain sale first. Preserve all order indices and every
    # originally funded buy; no extra overnight overflow may be introduced.
    for index in range(len(proposed)-1,-1,-1):
        if proposed[index][:2]!=['SELL','WHEAT']:continue
        final,sales,safe=project(proposed);shortage=max(0,need-final.get('WHEAT',0))
        if not shortage:break
        sold=sales.get(index,0)
        if not sold:continue
        old=proposed[index][2];proposed[index][2]=max(0,sold-shortage)
        after,_,safe=project(proposed)
        if not safe or after.get('WHEAT',0)<=final.get('WHEAT',0):proposed[index][2]=old
    final,_,safe=project(proposed);shortage=max(0,need-final.get('WHEAT',0))
    if shortage:
        last_sale=max((i for i,o in enumerate(proposed) if o[:2]==['SELL','WHEAT']),default=-1)
        index=next((i for i in range(len(proposed)-1,last_sale,-1) if proposed[i][:2]==['BUY_PRODUCT','WHEAT']),None)
        if index is not None:proposed[index][2]=max(0,int(proposed[index][2]))+shortage
        elif len(proposed)<10:proposed.append(['BUY_PRODUCT','WHEAT',shortage])
        else:_R97_REPORT['supply_slot_declines']+=1;return action
    final,_,safe=project(proposed)
    if not safe or final.get('WHEAT',0)<need:
        _R97_REPORT['supply_capacity_declines']+=1;return action
    if not _r97_budget(obs,proposed):
        _R97_REPORT['supply_budget_declines']+=1;return action
    if prefund:
        _R97_REPORT['supply_prefund_changes']+=1
        _R97_REPORT['supply_prefund_units']+=max(0,final.get('WHEAT',0)-original_final.get('WHEAT',0))
    if step<288:
        _R97_REPORT['supply_early_changes']+=1
        _R97_REPORT['supply_early_units']+=max(0,final.get('WHEAT',0)-original_final.get('WHEAT',0))
    _R97_REPORT['supply_guard_changes']+=1
    _R97_REPORT['supply_grain_protected']+=final.get('WHEAT',0)-original_final.get('WHEAT',0)
    _R97_REPORT['supply_buy_units']+=sum(max(0,int(o[2])) for o in proposed if o[:2]==['BUY_PRODUCT','WHEAT'])-sum(max(0,int(o[2])) for o in orders if o[:2]==['BUY_PRODUCT','WHEAT'])
    return result

def agent(observation,configuration=None):
    result=_R97_PARENT(observation,configuration)
    try:
        player=int(observation['player']);step=int(observation['step'])
        if player not in _R97_LAST or step<=_R97_LAST[player]:
            _R97_REPORT.update(supply_guard_changes=0,supply_grain_protected=0,supply_buy_units=0,supply_early_changes=0,supply_early_units=0,supply_prefund_changes=0,supply_prefund_units=0,supply_slot_declines=0,supply_capacity_declines=0,supply_budget_declines=0,supply_errors=0)
        _R97_LAST[player]=step
        if configuration is None or all(configuration.get(k,v)==v for k,v in [('boardSize',10),('turnsPerDay',24),('shedCapacity',100),('maxMarketOrdersPerTurn',10),('farmHandCostMult',1)]):result=_r97_supply(observation,result)
    except Exception:_R97_REPORT['supply_errors']=_R97_REPORT.get('supply_errors',0)+1
    _R97_REPORT.update(getattr(_R97_PARENT,'telemetry',{}))
    return result

agent.telemetry=_R97_REPORT
agent=globals().pop('agent')

"""Original funded planting and first-dawn labor contract, Ahmed Berat Ozer."""
_R124_PARENT=agent
_R124_STATES={}
_R124_REPORT={}

def _r124_labor_reserve(native):
    n=sum(bool(o) and o[0]=='HIRE' for o in native[24].get('market',[]))
    return sum(_v219_fib(i) for i in range(n)),n

def _r124_seed_budget(obs,action,reserve):
    if not any(o and o[0]=='BUY_SEED' for o in action.get('market',[])):return action
    player=int(obs['player']);budget=dict(obs,farms=[dict(f) for f in obs['farms']]);budget['farms'][player]['money']-=reserve
    if _r97_budget(budget,action.get('market',[])):return action
    result=copy.deepcopy(action)
    for i in range(len(result['market'])-1,-1,-1):
        order=result['market'][i]
        if len(order)<3 or order[0]!='BUY_SEED':continue
        before=max(0,int(order[2]));order[2]=before
        while order[2]>0 and not _r97_budget(budget,result['market']):order[2]-=1
        n=before-order[2]
        if n:
            _R124_REPORT['opening_seed_budget_units']+=n
            _R124_REPORT['opening_seed_budget_cost']+=n*{'WHEAT':10,'CARROT':20,'TOMATO':50,'STRAWBERRY':100,'MELON':80}[order[1]]
        if _r97_budget(budget,result['market']):break
    return result

def _r124_atomic(obs,action,state):
    commands=[action.get('farmer') or ['PASS'],*(action.get('hands') or [])]
    demand={}
    for c in commands:
        if len(c)>1 and c[0]=='PLANT':demand[c[1]]=demand.get(c[1],0)+1
    if not demand:return action
    available=obs['private']['seeds'];blocked={p for p,n in demand.items() if n>available.get(p,0)}
    farm,private=_PLANNER_NS['_clone_state'](obs['farms'][obs['player']],obs['private'])
    changed=False;kept=[]
    for actor,c in enumerate(commands):
        if len(c)>1 and c[0]=='PLANT':
            pos=None if actor>=len(private['inventories']) else farm['farmer'] if actor==0 else farm['hands'][actor-1]
            valid=pos is not None and farm['tiles'][pos[1]][pos[0]] is None and private['seeds'].get(c[1],0)>0
            if not valid:
                commands[actor]=['PASS'];changed=True;_R124_REPORT['opening_atomic_dropped']+=1
            elif c[1] in blocked:
                kept.append(dict(xy=list(pos),crop=c[1],birth=0));_R124_REPORT['opening_atomic_rescued_requests']+=1
        if actor<len(private['inventories']):_PLANNER_NS['_apply_unit_action'](farm,private,actor,commands[actor],10,0,24,100)
    if kept:state['pending_plants']=kept
    if changed:
        action=dict(action,farmer=commands[0],hands=commands[1:])
    return action

def agent(observation,configuration=None):
    result=_R124_PARENT(observation,configuration)
    try:
        player=int(observation['player']);step=int(observation['step']);state=_R124_STATES.get(player)
        if state is None or step<=state['step']:
            state=_R124_STATES[player]={'step':-1,'pending_plants':[]}
            _R124_REPORT.update(opening_seed_budget_units=0,opening_seed_budget_cost=0,opening_atomic_dropped=0,opening_atomic_rescued_requests=0,opening_atomic_rescued_confirmed=0,opening_atomic_plant_errors=0,opening_day1_cash=0,opening_day1_hires_requested=0,opening_day1_hires_confirmed=0,opening_day1_hire_shortfalls=0,opening_contract_errors=0)
        state['step']=step;farm=observation['farms'][player]
        for p in state.pop('pending_plants',[]):
            x,y=p['xy'];t=farm['tiles'][y][x]
            if isinstance(t,dict) and t.get('crop')==p['crop'] and t.get('planted_day')==p['birth']:_R124_REPORT['opening_atomic_rescued_confirmed']+=1
            else:_R124_REPORT['opening_atomic_plant_errors']+=1
        standard=configuration is None or all(configuration.get(k,v)==v for k,v in [('boardSize',10),('turnsPerDay',24),('shedCapacity',100),('maxMarketOrdersPerTurn',10),('farmHandCostMult',1)])
        if standard and 0<=step<24:
            native=_ROUTES[_IMPL.chassis.players[player]['route']];reserve,hires=_r124_labor_reserve(native)
            result=_r124_seed_budget(observation,result,reserve);result=_r124_atomic(observation,result,state)
        if step==24:
            _R124_REPORT['opening_day1_cash']=farm['money'];state['hires']=sum(bool(o) and o[0]=='HIRE' for o in result.get('market',[]));_R124_REPORT['opening_day1_hires_requested']=state['hires']
        if step==25:
            actual=len(farm['hands']);_R124_REPORT['opening_day1_hires_confirmed']=actual;_R124_REPORT['opening_day1_hire_shortfalls']=max(0,state.get('hires',0)-actual)
    except Exception:_R124_REPORT['opening_contract_errors']=_R124_REPORT.get('opening_contract_errors',0)+1
    _R124_REPORT.update(getattr(_R124_PARENT,'telemetry',{}))
    return result
agent.telemetry=_R124_REPORT
agent=globals().pop('agent')

"""Fund urgent grain at the first quote slot; avoid unwatered last-hour plants.
Original safety contracts by Ahmed Berat Ozer, EXP258.
"""
_R127_PARENT=agent
_R127_STATES={}
_R127_REPORT={}

def _r127_fields(obs,action):
    farm,private=_PLANNER_NS['_clone_state'](obs['farms'][obs['player']],obs['private'])
    commands=[action.get('farmer') or ['PASS'],*(action.get('hands') or [])]
    demand={}
    for c in commands:
        if len(c)>1 and c[0]=='PLANT':demand[c[1]]=demand.get(c[1],0)+1
    blocked={p for p,n in demand.items() if n>private['seeds'].get(p,0)}
    for actor,c in enumerate(commands[:len(private['inventories'])]):
        if len(c)>1 and c[0]=='PLANT' and c[1] in blocked:continue
        _PLANNER_NS['_apply_unit_action'](farm,private,actor,c,10,int(obs['step'])//24,24,100)
    return farm,private

def _r127_last_hour(obs,action):
    if int(obs['step'])%24!=23:return action
    commands=[action.get('farmer') or ['PASS'],*(action.get('hands') or [])]
    if not any(c and c[0]=='PLANT' for c in commands):return action
    result=copy.deepcopy(action);changed=False
    # Removing rejected requests can unblock the engine's atomic crop batch.
    # Recompute until every retained request ends the turn with a watered crop.
    for _ in range(len(commands)+1):
        farm,_=_r127_fields(obs,result);original=obs['farms'][obs['player']]
        positions=[original['farmer'],*original['hands']];drop=[]
        for actor,c in enumerate(commands):
            if not c or c[0]!='PLANT':continue
            tile=None
            if actor<len(positions):
                x,y=positions[actor];tile=farm['tiles'][y][x]
            if not (isinstance(tile,dict) and tile.get('kind')=='PLANT' and tile.get('crop')==c[1] and tile.get('planted_day')==int(obs['step'])//24 and tile.get('watered_today')):drop.append(actor)
        if not drop:break
        for actor in drop:commands[actor]=['PASS']
        result['farmer']=commands[0];result['hands']=commands[1:];changed=True
        _R127_REPORT['last_hour_plants_dropped']+=len(drop)
    return result if changed else action

def _r127_prefix_bound(obs,quantity):
    inventory=int(obs['market']['inventory']['WHEAT'])
    # Both players quote one unit before either commits. Before own unit j,
    # at most j-1 own and j-1 opponent wheat purchases have depleted inventory.
    return sum(_r37_market_price('WHEAT',inventory-(2*j-1)) for j in range(1,quantity+1))

def _r127_priority(obs,action,state):
    step=int(obs['step']);player=int(obs['player'])
    if not 144<=step<695:return action
    orders=action.get('market') or []
    if len(orders)>9 or any(o[:2] in (['BUY_PRODUCT','WHEAT'],['SELL','WHEAT']) for o in orders):return action
    native=_IMPL.chassis.players[player]
    future=_IMPL.chassis.routes[2 if step+1>=648 else native['route']][step+1]
    commands=[future.get('farmer') or ['PASS'],*(future.get('hands') or [])]
    if not any(c[:2]==['PICKUP','WHEAT'] for c in commands):return action
    if not _r97_budget(obs,orders):return action
    farm,private=_r127_fields(obs,action);night=step%24==23
    access=((4,4),(5,4),(4,5),(5,5));positions=[tuple(farm['farmer']),*map(tuple,farm['hands'])]
    if night:positions=[(4,4)]
    else:
        for o in orders:
            if o and o[0]=='HIRE':positions.append(min(access,key=lambda p:(positions.count(p),access.index(p))))
    need=sum(max(0,int(c[2]) if len(c)>2 else 1) for pos,c in zip(positions,commands) if pos in access and c[:2]==['PICKUP','WHEAT'])
    stock,buys,_=_r97_market_stock(private['shed'],orders);before,loss=_r97_delivery(stock,private,night)
    shortage=max(0,need-before.get('WHEAT',0))
    if not shortage or shortage>100-sum(private['shed'].values()):return action
    cost=_r127_prefix_bound(obs,shortage)
    budget=dict(obs,farms=[dict(f) for f in obs['farms']]);budget['farms'][player]['money']-=cost
    if not _r97_budget(budget,orders):return action
    proposed=[['BUY_PRODUCT','WHEAT',shortage],*copy.deepcopy(orders)]
    stock,after_buys,_=_r97_market_stock(private['shed'],proposed);after,after_loss=_r97_delivery(stock,private,night)
    if after.get('WHEAT',0)<need or any(after_buys.get(i+1,0)<q for i,q in buys.items()) or any(q>loss.get(p,0) for p,q in after_loss.items()):return action
    state['pending_grain']=(step+1,after.get('WHEAT',0),shortage)
    _R127_REPORT['priority_grain_orders']+=1;_R127_REPORT['priority_grain_units']+=shortage
    return dict(action,market=proposed)

def agent(observation,configuration=None):
    result=_R127_PARENT(observation,configuration)
    try:
        player=int(observation['player']);step=int(observation['step']);state=_R127_STATES.get(player)
        if state is None or step<=state['step']:
            state=_R127_STATES[player]={'step':-1}
            _R127_REPORT.update(last_hour_plants_dropped=0,priority_grain_orders=0,priority_grain_units=0,priority_grain_confirmed=0,priority_grain_shortfalls=0,priority_contract_errors=0)
        pending=state.pop('pending_grain',None)
        if pending and step==pending[0]:
            if observation['private']['shed'].get('WHEAT',0)>=pending[1]:_R127_REPORT['priority_grain_confirmed']+=pending[2]
            else:_R127_REPORT['priority_grain_shortfalls']+=1
        state['step']=step
        standard=configuration is None or all(configuration.get(k,v)==v for k,v in [('boardSize',10),('turnsPerDay',24),('shedCapacity',100),('maxMarketOrdersPerTurn',10),('farmHandCostMult',1)])
        if standard:result=_r127_priority(observation,_r127_last_hour(observation,result),state)
    except Exception:_R127_REPORT['priority_contract_errors']=_R127_REPORT.get('priority_contract_errors',0)+1
    _R127_REPORT.update(getattr(_R127_PARENT,'telemetry',{}))
    return result
agent.telemetry=_R127_REPORT
agent=globals().pop('agent')

"""Observed-input service order and guaranteed first-sale funding, Ahmed Berat Ozer."""
_R128_PARENT=agent
_R128_STATES={}
_R128_REPORT={}

def _r128_commands(action):
    return [action.get('farmer') or ['PASS'],*(action.get('hands') or [])]

def _r128_future(obs,offset=1):
    step=int(obs['step'])+offset;native=_IMPL.chassis.players[int(obs['player'])]
    return _IMPL.chassis.routes[2 if step>=648 else native['route']][step]

def _r128_sale_credit(obs,action):
    orders=action.get('market') or []
    if not orders or len(orders[0])<3 or orders[0][0]!='SELL' or orders[0][1]=='WHEAT':return 0
    item=orders[0][1]
    if item not in obs['market']['inventory']:return 0
    _,private=_r127_fields(obs,action)
    q=min(max(0,int(orders[0][2])),max(0,int(private['shed'].get(item,0))))
    inventory=int(obs['market']['inventory'][item])
    return sum(_r37_market_price(item,inventory+2*j) for j in range(q))

def _r128_next_need(obs,farm,orders,advanced):
    step=int(obs['step']);access=((4,4),(5,4),(4,5),(5,5))
    positions=[tuple(farm['farmer']),*map(tuple,farm['hands'])]
    if step%24==23:positions=[(4,4)]
    else:
        for order in orders:
            if order and order[0]=='HIRE':positions.append(min(access,key=lambda p:(positions.count(p),access.index(p))))
    return sum(max(0,int(c[2]) if len(c)>2 else 1) for actor,(p,c) in enumerate(zip(positions,_r128_commands(_r128_future(obs)))) if actor not in advanced and p in access and c[:2]==['PICKUP','WHEAT'])

def _r128_field_safe(obs,before,after,advanced):
    orders=before.get('market') or []
    if not _r97_budget(obs,orders):return False,None
    oldfarm,oldprivate=_r127_fields(obs,before);farm,private=_r127_fields(obs,after)
    oldcommands=_r128_commands(before);commands=_r128_commands(after)
    for i,c in enumerate(oldcommands[:len(private['inventories'])]):
        if c and c[0]=='PICKUP' and c==commands[i]:
            item=c[1]
            if private['inventories'][i].get(item,0)<oldprivate['inventories'][i].get(item,0):return False,None
    oldstock,oldbuys,oldsales=_r97_market_stock(oldprivate['shed'],orders)
    stock,buys,sales=_r97_market_stock(private['shed'],orders)
    oldfinal,oldloss=_r97_delivery(oldstock,oldprivate,int(obs['step'])%24==23)
    final,loss=_r97_delivery(stock,private,int(obs['step'])%24==23)
    if any(buys.get(i,0)<q for i,q in oldbuys.items()) or any(sales.get(i,0)<q for i,q in oldsales.items()) or any(q>oldloss.get(p,0) for p,q in loss.items()):return False,None
    need=_r128_next_need(obs,farm,orders,advanced)
    pending=_R127_STATES.get(int(obs['player']),{}).get('pending_grain')
    if pending and pending[0]==int(obs['step'])+1:need=max(need,pending[1])
    if final.get('WHEAT',0)<min(need,oldfinal.get('WHEAT',0)):return False,None
    return True,private

def _r128_food_need(obs,farm,private,actor):
    farm,private=copy.deepcopy(farm),copy.deepcopy(private);missing=0;day=int(obs['step'])//24
    for offset in range(1,min(5,24-int(obs['step'])%24)):
        cs=_r128_commands(_r128_future(obs,offset));c=cs[actor] if actor<len(cs) else ['PASS']
        if c[:2]==['PICKUP','WHEAT']:break
        if c==['FEED']:
            x,y=farm['farmer'] if actor==0 else farm['hands'][actor-1];tile=farm['tiles'][y][x]
            if isinstance(tile,dict) and tile.get('animal') and not tile.get('fed_today') and private['inventories'][actor].get('WHEAT',0)<=0:
                private['inventories'][actor]['WHEAT']=1;missing+=1
        _PLANNER_NS['_apply_unit_action'](farm,private,actor,c,10,day,24,100)
    return missing

def _r128_service(obs,action,state):
    step=int(obs['step']);farm=obs['farms'][obs['player']];private=obs['private'];positions=[farm['farmer'],*farm['hands']]
    for item in state.pop('arrivals',[]):
        if item['step']!=step or item['actor']>=len(private['inventories']) or private['inventories'][item['actor']].get('WHEAT',0)<item['expected']:_R128_REPORT['service_arrival_errors']+=1
        else:_R128_REPORT['service_confirmed_units']+=item['quantity']
    old_pending=state.pop('swaps',{});result=action;cs=_r128_commands(result)
    for actor,pending in old_pending.items():
        if step!=pending['step'] or actor>=len(positions) or list(positions[actor])!=pending['xy'] or cs[actor]!=pending['pickup']:
            _R128_REPORT['service_swap_errors']+=1;continue
        x,y=positions[actor];tile=farm['tiles'][y][x]
        if not isinstance(tile,dict) or tile.get('animal')!=pending['animal'] or tile.get('placed_day')!=pending['birth']:
            _R128_REPORT['service_swap_errors']+=1;continue
        if private['inventories'][actor].get('WHEAT',0)<pending['quantity']+1:
            _R128_REPORT['service_swap_errors']+=1;continue
        cs[actor]=['PASS'] if tile.get('fed_today') else ['FEED']
        _R128_REPORT['service_swaps_completed']+=1
    if old_pending:
        proposed=dict(result,farmer=cs[0],hands=cs[1:]);safe,_=_r128_field_safe(obs,result,proposed,{})
        if safe:result=proposed
        else:_R128_REPORT['service_swap_errors']+=1
    if not 144<=step<647:return result
    cs=_r128_commands(result);future=_r128_commands(_r128_future(obs));proposed=copy.deepcopy(result);commands=_r128_commands(proposed)
    swaps={};arrivals=[];prefetches=0
    projected_farm,projected_private=_r127_fields(obs,result)
    for actor,c in enumerate(cs[:len(private['inventories'])]):
        x,y=positions[actor]
        if (x,y) not in ((4,4),(5,4),(4,5),(5,5)):continue
        held=max(0,int(private['inventories'][actor].get('WHEAT',0)));tile=farm['tiles'][y][x]
        nxt=future[actor] if actor<len(future) else ['PASS']
        if c==['FEED'] and held==0 and step%24<=21 and isinstance(tile,dict) and tile.get('animal') and not tile.get('fed_today') and nxt[:2]==['PICKUP','WHEAT']:
            q=max(0,int(nxt[2]) if len(nxt)>2 else 1)
            if not q:continue
            commands[actor]=['PICKUP','WHEAT',q+1]
            swaps[actor]=dict(step=step+1,xy=[x,y],animal=tile['animal'],birth=tile.get('placed_day'),quantity=q,pickup=copy.deepcopy(nxt))
            arrivals.append(dict(step=step+1,actor=actor,expected=q+1,quantity=q+1))
        elif c==['PASS']:
            q=_r128_food_need(obs,projected_farm,projected_private,actor)
            if q:
                commands[actor]=['PICKUP','WHEAT',q];prefetches+=1
                arrivals.append(dict(step=step+1,actor=actor,expected=held+q,quantity=q))
    if not arrivals:return result
    proposed['farmer']=commands[0];proposed['hands']=commands[1:]
    safe,after=_r128_field_safe(obs,result,proposed,swaps)
    if not safe or any(after['inventories'][p['actor']].get('WHEAT',0)<p['expected'] for p in arrivals):
        _R128_REPORT['service_capacity_declines']+=1;return result
    state['swaps']=swaps;state['arrivals']=arrivals
    _R128_REPORT['service_swaps_started']+=len(swaps);_R128_REPORT['service_idle_preloads']+=prefetches
    _R128_REPORT['service_requested_units']+=sum(p['quantity'] for p in arrivals)
    return proposed

def _r128_credit_supply(obs,action,state):
    step=int(obs['step'])
    if not 144<=step<695 or _r97_budget(obs,action.get('market') or []):return action
    credit=_r128_sale_credit(obs,action)
    if not credit:return action
    budget=dict(obs,farms=[dict(f) for f in obs['farms']]);budget['farms'][obs['player']]['money']+=credit
    if not _r97_budget(budget,action.get('market') or []):return action
    result=_r97_supply(budget,action)
    if result==action:return action
    assert result['market'][0]==action['market'][0]
    _,private=_r127_fields(obs,result);stock,_,_=_r97_market_stock(private['shed'],result['market']);final,_=_r97_delivery(stock,private,step%24==23)
    quantity=lambda a:sum(max(0,int(o[2])) for o in a.get('market',[]) if len(o)>2 and o[:2]==['BUY_PRODUCT','WHEAT'])
    extra=max(0,quantity(result)-quantity(action));state['credit_pending']=(step+1,final.get('WHEAT',0),extra)
    _R128_REPORT['sale_credit_orders']+=1;_R128_REPORT['sale_credit_lower_bound']+=credit;_R128_REPORT['sale_credit_grain_units']+=extra
    return result

def agent(observation,configuration=None):
    result=_R128_PARENT(observation,configuration)
    try:
        player=int(observation['player']);step=int(observation['step']);state=_R128_STATES.get(player)
        if state is None or step<=state['step']:
            state=_R128_STATES[player]={'step':-1}
            _R128_REPORT.update(service_requested_units=0,service_confirmed_units=0,service_idle_preloads=0,service_swaps_started=0,service_swaps_completed=0,service_capacity_declines=0,service_arrival_errors=0,service_swap_errors=0,sale_credit_orders=0,sale_credit_lower_bound=0,sale_credit_grain_units=0,sale_credit_confirmed_units=0,sale_credit_errors=0,service_errors=0)
        state['step']=step;pending=state.pop('credit_pending',None)
        if pending:
            if pending[0]!=step or observation['private']['shed'].get('WHEAT',0)<pending[1]:_R128_REPORT['sale_credit_errors']+=1
            else:_R128_REPORT['sale_credit_confirmed_units']+=pending[2]
        standard=configuration is None or all(configuration.get(k,v)==v for k,v in [('boardSize',10),('turnsPerDay',24),('shedCapacity',100),('maxMarketOrdersPerTurn',10),('farmHandCostMult',1)])
        if standard:result=_r128_credit_supply(observation,_r128_service(observation,result,state),state)
    except Exception:_R128_REPORT['service_errors']=_R128_REPORT.get('service_errors',0)+1
    _R128_REPORT.update(getattr(_R128_PARENT,'telemetry',{}));_R128_REPORT.update(_R97_REPORT)
    return result
agent.telemetry=_R128_REPORT
agent=globals().pop('agent')


# EXP-260: replay-supported conservative guards.
#
# 1. A PLANT blocked by a same-day weed may be recovered when the original
#    per-worker tape contains an immediate WATER and a later PASS before dawn.
#    The inserted PLANT delays that worker's commands by one position; the PASS
#    absorbs the delay, so the worker is synchronized again within the same day.
# 2. At the end of a 72-turn block, an already-confirmed mirror race may reorder
#    an all-SELL queue.  Quantities and field actions are immutable.  A bounded
#    pair-swap search accepts only a positive gain under the exact public price
#    curve and the conservative hypothesis that the rival has the same queue and
#    saleable stock.  This layer never changes buys, hires, land or input stock.

_R132_PARENT = agent
_R132_STATES = {}
_R132_REPORT = {}


def _r132_commands(action):
    return [action.get('farmer') or ['PASS'], *(action.get('hands') or [])]


def _r132_set_command(action, actor, command):
    result = copy.deepcopy(action)
    if actor == 0:
        result['farmer'] = list(command)
    else:
        hands = list(result.get('hands') or [])
        if actor - 1 >= len(hands):
            return action
        hands[actor - 1] = list(command)
        result['hands'] = hands
    return result


def _r132_standard(configuration):
    return configuration is None or all(
        configuration.get(key, value) == value
        for key, value in (
            ('boardSize', 10),
            ('turnsPerDay', 24),
            ('shedCapacity', 100),
            ('maxMarketOrdersPerTurn', 10),
            ('farmHandCostMult', 1),
        )
    )


def _r132_command_valid(observation, action, actor, command):
    view = FarmView(observation)
    if actor >= len(view.positions):
        return False
    position = view.positions[actor]
    tile = _tile_at(view.tiles, position)
    if _is_noop(command, tile, view.inv(actor), view.seeds, position, view.board):
        return False
    if command and command[0] == 'PLANT':
        crop = command[1]
        demand = sum(
            1 for current in _r132_commands(action)
            if current[:2] == ['PLANT', crop]
        )
        # Include the replacement if the parent's command was not this plant.
        parent = _r132_commands(action)[actor]
        demand += int(parent[:2] != ['PLANT', crop])
        if demand > int(view.seeds.get(crop, 0)):
            return False
    return True


def _r132_start_repair(observation, action, state):
    step = int(observation['step'])
    if step % 24 > 20 or state.get('repair'):
        return
    player = int(observation['player'])
    native = _IMPL.chassis.players.get(player)
    if not native or native.get('route') not in _IMPL.chassis.routes:
        return
    tape = _IMPL.chassis.routes[native['route']]
    if step + 2 >= len(tape):
        return
    raw = _r132_commands(tape[step])
    actual = _r132_commands(action)
    farm = observation['farms'][player]
    positions = [farm['farmer'], *farm['hands']]
    private = observation['private']
    day_end = min(len(tape), (step // 24 + 1) * 24)
    for actor, intended in enumerate(raw[:len(positions)]):
        if intended[:1] != ['PLANT'] or actor >= len(actual) or actual[actor] != ['DIG']:
            continue
        x, y = positions[actor]
        tile = farm['tiles'][y][x]
        crop = intended[1]
        if not (isinstance(tile, dict) and tile.get('kind') == 'WEED'):
            continue
        same_crop = sum(1 for command in raw if command[:2] == ['PLANT', crop])
        if int(private['seeds'].get(crop, 0)) < same_crop:
            continue
        next_commands = _r132_commands(tape[step + 1])
        if actor >= len(next_commands) or next_commands[actor] != ['WATER']:
            continue
        deadline = None
        for future_step in range(step + 1, day_end):
            future = _r132_commands(tape[future_step])
            command = future[actor] if actor < len(future) else ['PASS']
            if command == ['PASS']:
                deadline = future_step
                break
        if deadline is None:
            continue
        state['repair'] = {
            'actor': actor,
            'queue': [list(intended)],
            'next_step': step + 1,
            'deadline': deadline,
            'route': native['route'],
        }
        _R132_REPORT['weed_repair_started'] += 1
        return


def _r132_continue_repair(observation, action, state):
    repair = state.get('repair')
    if not repair:
        return action
    step = int(observation['step'])
    player = int(observation['player'])
    native = _IMPL.chassis.players.get(player) or {}
    if (step != repair['next_step'] or step > repair['deadline'] or
            native.get('route') != repair['route']):
        state.pop('repair', None)
        _R132_REPORT['weed_repair_aborted'] += 1
        return action
    actor = repair['actor']
    commands = _r132_commands(action)
    if actor >= len(commands) or not repair['queue']:
        state.pop('repair', None)
        _R132_REPORT['weed_repair_aborted'] += 1
        return action
    command = repair['queue'].pop(0)
    if not _r132_command_valid(observation, action, actor, command):
        state.pop('repair', None)
        _R132_REPORT['weed_repair_aborted'] += 1
        return action
    displaced = commands[actor]
    if displaced != ['PASS']:
        repair['queue'].append(list(displaced))
    if step >= repair['deadline'] and repair['queue']:
        state.pop('repair', None)
        _R132_REPORT['weed_repair_aborted'] += 1
        return action
    result = _r132_set_command(action, actor, command)
    repair['next_step'] = step + 1
    _R132_REPORT['weed_repair_shifted_steps'] += 1
    if command[:1] == ['PLANT']:
        _R132_REPORT['weed_repair_plants'] += 1
    if command == ['WATER']:
        _R132_REPORT['weed_repair_waters'] += 1
    if not repair['queue']:
        state.pop('repair', None)
        _R132_REPORT['weed_repair_completed'] += 1
    return result


def _r132_price_params(observation):
    params = {item: dict(values) for item, values in _R37_MARKET_PARAMS.items()}
    for item, patch in observation['market'].get('params', {}).items():
        if item in params:
            params[item].update(patch)
    return params


def _r132_mirror_revenue(observation, own_orders, rival_orders, stock):
    inventory = dict(observation['market']['inventory'])
    params = _r132_price_params(observation)
    stocks = [dict(stock), dict(stock)]
    revenue = 0
    for index in range(max(len(own_orders), len(rival_orders))):
        pair = []
        for side, orders in enumerate((own_orders, rival_orders)):
            if index >= len(orders):
                pair.append(None)
                continue
            order = orders[index]
            if not order or len(order) < 3 or order[0] != 'SELL':
                pair.append(None)
                continue
            item = order[1]
            quantity = min(
                max(0, int(order[2])),
                max(0, int(stocks[side].get(item, 0))),
            )
            pair.append([item, quantity])
        while any(current is not None and current[1] > 0 for current in pair):
            sold = []
            for side, current in enumerate(pair):
                if current is None or current[1] <= 0:
                    continue
                item = current[0]
                quote = _r37_market_price(item, inventory[item], params)
                if side == 0:
                    revenue += quote
                current[1] -= 1
                stocks[side][item] = max(0, stocks[side].get(item, 0) - 1)
                sold.append((item, quote))
            for item, quote in sold:
                if quote > 1:
                    inventory[item] += 1
    return revenue


def _r132_mirror_reorder(observation, action):
    step = int(observation['step'])
    player = int(observation['player'])
    rival = 1 - player
    orders = action.get('market') or []
    probe = _R44_PROBES.get(player) or {}
    if not (336 <= step < 696 and step % 72 == 71 and probe.get('matched')):
        return action
    if observation['farms'][player]['money'] > observation['farms'][rival]['money'] + 500:
        return action
    if _r37_similarity(observation) < 0.90 or len(orders) < 2:
        return action
    if any(
        not order or len(order) < 3 or order[0] != 'SELL' or
        order[1] not in PRODUCTS or type(order[2]) is not int or order[2] < 0
        for order in orders
    ):
        return action
    stock = projected_shed(action, FarmView(observation))
    if not any(min(order[2], max(0, int(stock.get(order[1], 0)))) > 0 for order in orders):
        return action
    rival_orders = copy.deepcopy(orders)
    current = copy.deepcopy(orders)
    baseline = _r132_mirror_revenue(observation, current, rival_orders, stock)
    score = baseline
    # At most ten orders: bounded deterministic pair-swap ascent is inexpensive.
    for _ in range(10):
        best_score = score
        best_orders = None
        for left in range(len(current)):
            for right in range(left + 1, len(current)):
                if current[left] == current[right]:
                    continue
                trial = copy.deepcopy(current)
                trial[left], trial[right] = trial[right], trial[left]
                trial_score = _r132_mirror_revenue(
                    observation, trial, rival_orders, stock
                )
                if trial_score > best_score:
                    best_score, best_orders = trial_score, trial
        if best_orders is None:
            break
        current, score = best_orders, best_score
    gain = score - baseline
    if gain < 25 or current == orders:
        return action
    _R132_REPORT['mirror_reorders'] += 1
    _R132_REPORT['mirror_modeled_gain'] += gain
    result = copy.deepcopy(action)
    result['market'] = current
    return result


def agent(observation, configuration=None):
    result = _R132_PARENT(observation, configuration)
    try:
        player = int(observation['player'])
        step = int(observation.get('step', observation['day'] * 24 + observation['hour']))
        state = _R132_STATES.get(player)
        if state is None or step <= state['step']:
            state = _R132_STATES[player] = {'step': -1}
            _R132_REPORT.update(
                weed_repair_started=0,
                weed_repair_shifted_steps=0,
                weed_repair_plants=0,
                weed_repair_waters=0,
                weed_repair_completed=0,
                weed_repair_aborted=0,
                mirror_reorders=0,
                mirror_modeled_gain=0,
                replay_guard_errors=0,
            )
        state['step'] = step
        if not _r132_standard(configuration):
            return result
        if state.get('repair'):
            result = _r132_continue_repair(observation, result, state)
        else:
            _r132_start_repair(observation, result, state)
        result = _r132_mirror_reorder(observation, result)
    except Exception:
        _R132_REPORT['replay_guard_errors'] = _R132_REPORT.get('replay_guard_errors', 0) + 1
    _R132_REPORT.update(getattr(_R132_PARENT, 'telemetry', {}))
    return result


agent.telemetry = _R132_REPORT
agent = globals().pop('agent')


# EXP-261: bounded egg-demand livestock substitution.
# Two early cow slots become geese only when both observed opening shops demand
# eggs and neither demands milk.  Every downstream mutation is confirmed from
# owned stock: purchase -> pickup -> coop build -> goose placement.  FEED, CARE
# and HARVEST are item-agnostic in the parent tape and therefore stay unchanged.

_R134_PARENT = agent
_R134_STATES = {}
_R134_REPORT = {}
_R134_EGG_SHOPS = frozenset(('BAKERY', 'BRUNCH_SPOT'))
_R134_MILK_SHOPS = frozenset(('PIZZA_SHOP', 'ICE_CREAM_SHOP', 'SMOOTHIE_SHOP'))


def _r134_animals(farm, animal):
    return sum(
        1 for row in farm['tiles'] for tile in row
        if isinstance(tile, dict) and tile.get('animal') == animal
    )


def _r134_commit(observation, action, state):
    step = int(observation['step'])
    if state.get('committed') or not 144 <= step <= 168:
        return action
    shops = list(observation['town'].get('unlocked_shops') or [])[:2]
    if (len(shops) != 2 or not all(shop in _R134_EGG_SHOPS for shop in shops) or
            any(shop in _R134_MILK_SHOPS for shop in shops)):
        return action
    orders = action.get('market') or []
    candidates = [
        index for index, order in enumerate(orders)
        if order and len(order) == 3 and order[:2] == ['BUY_ANIMAL', 'COW']
        and type(order[2]) is int and order[2] == 2
    ]
    if len(candidates) != 1 or any(
        order and order[0] == 'BUY_ANIMAL' and index != candidates[0]
        for index, order in enumerate(orders)
    ):
        return action
    player = int(observation['player'])
    farm = observation['farms'][player]
    private = observation['private']
    # Do not take over an unrelated animal already waiting in the depot.
    if int(private['shed'].get('GOOSE', 0)) or any(
        int(inventory.get('GOOSE', 0)) for inventory in private['inventories']
    ):
        return action
    result = copy.deepcopy(action)
    result['market'][candidates[0]][1] = 'GOOSE'
    state.update(
        committed=True,
        quota=2,
        initial_geese=_r134_animals(farm, 'GOOSE'),
        pending=(step + 1, int(private['shed'].get('GOOSE', 0))),
        confirmed=False,
    )
    _R134_REPORT['egg_substitution_requests'] += 1
    return result


def _r134_confirm_purchase(observation, state):
    pending = state.pop('pending', None)
    if not pending:
        return
    step, before = pending
    gained = max(0, int(observation['private']['shed'].get('GOOSE', 0)) - before)
    if int(observation['step']) == step and gained >= state['quota']:
        state['confirmed'] = True
        _R134_REPORT['egg_substitution_confirmed_units'] += state['quota']
    else:
        state['confirmed'] = False
        _R134_REPORT['egg_substitution_purchase_shortfalls'] += state['quota']


def _r134_physical(observation, action, state):
    if not state.get('confirmed'):
        return action
    step = int(observation['step'])
    player = int(observation['player'])
    farm = observation['farms'][player]
    private = observation['private']
    positions = [farm['farmer'], *farm['hands']]
    commands = _r132_commands(action)
    placed = max(0, _r134_animals(farm, 'GOOSE') - state['initial_geese'])
    if placed >= state['quota']:
        state['confirmed'] = False
        _R134_REPORT['egg_substitution_completed'] += 1
        return action
    carrying = sum(
        int(inventory.get('GOOSE', 0))
        for inventory in private['inventories']
    )
    free = max(0, state['quota'] - placed - carrying)
    depot = max(0, int(private['shed'].get('GOOSE', 0)))
    changed = False
    for actor, command in enumerate(commands[:len(private['inventories'])]):
        inventory = private['inventories'][actor]
        x, y = positions[actor]
        tile = farm['tiles'][y][x]
        if command[:2] == ['PICKUP', 'COW'] and free > 0 and depot > 0:
            quantity = max(0, int(command[2]) if len(command) > 2 else 1)
            take = min(quantity, free, depot)
            if take > 0 and _shed_adjacent((x, y), 10):
                commands[actor] = ['PICKUP', 'GOOSE', take]
                free -= take
                depot -= take
                changed = True
                _R134_REPORT['egg_substitution_pickups'] += take
        elif command == ['BUILD_PASTURE'] and int(inventory.get('GOOSE', 0)) > 0:
            if tile is None:
                commands[actor] = ['BUILD_COOP']
                changed = True
                _R134_REPORT['egg_substitution_coops'] += 1
        elif command[:2] == ['PLACE', 'COW'] and int(inventory.get('GOOSE', 0)) > 0:
            if isinstance(tile, dict) and tile.get('kind') == 'COOP' and tile.get('animal') is None:
                commands[actor] = ['PLACE', 'GOOSE']
                changed = True
                _R134_REPORT['egg_substitution_placements'] += 1
    if step > 216 and placed < state['quota']:
        state['confirmed'] = False
        _R134_REPORT['egg_substitution_aborted'] += 1
    if not changed:
        return action
    result = copy.deepcopy(action)
    result['farmer'], result['hands'] = commands[0], commands[1:]
    return result


def agent(observation, configuration=None):
    result = _R134_PARENT(observation, configuration)
    try:
        player = int(observation['player'])
        step = int(observation.get('step', observation['day'] * 24 + observation['hour']))
        state = _R134_STATES.get(player)
        if state is None or step <= state['step']:
            state = _R134_STATES[player] = {'step': -1}
            _R134_REPORT.update(
                egg_substitution_requests=0,
                egg_substitution_confirmed_units=0,
                egg_substitution_purchase_shortfalls=0,
                egg_substitution_pickups=0,
                egg_substitution_coops=0,
                egg_substitution_placements=0,
                egg_substitution_completed=0,
                egg_substitution_aborted=0,
                egg_substitution_errors=0,
            )
        state['step'] = step
        if not _r132_standard(configuration):
            return result
        _r134_confirm_purchase(observation, state)
        result = _r134_commit(observation, result, state)
        result = _r134_physical(observation, result, state)
    except Exception:
        _R134_REPORT['egg_substitution_errors'] = _R134_REPORT.get('egg_substitution_errors', 0) + 1
    _R134_REPORT.update(getattr(_R134_PARENT, 'telemetry', {}))
    return result


agent.telemetry = _R134_REPORT
agent = globals().pop('agent')




# ---------------------------------------------------------------------------
# farmbot layer F1: front-run copies of this public tape.
# Mirror matches between copies of the tape end in exact ties. The chassis
# already ships a front_run hook (sell MILK/WOOL/STRAWBERRY/MELON one step
# before the opponent's scheduled SELL); it was disabled and had no plan.
# We assume the opponent follows the same route we do (same shop draw ->
# same router choice) and feed our own route as their expected plan.
# ---------------------------------------------------------------------------
_F1_PARENT = agent
del agent


def agent(observation, configuration=None):
    try:
        chassis = _IMPL.chassis
        player = _int(_get(observation, "player", 0))
        state = chassis.players.get(player) if isinstance(chassis.players, dict) else None
        route = state.get("route", 0) if isinstance(state, dict) else 0
        chassis.cfg["front_run"] = True
        chassis.opponent_plan = chassis.routes.get(route, chassis.routes.get(0))
    except Exception:
        pass
    return _F1_PARENT(observation, configuration)
