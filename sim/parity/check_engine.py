"""Play games on the official Python engine and check kaggsim turn by turn.

Usage:
    python3 parity/check_engine.py target/release/kaggsim path/to/kaggriculture.py [--games N] [--agent main.py]

Get the engine without installing kaggle-environments and its dependencies:
    pip download --no-deps kaggle-environments==1.32.7
    unzip kaggle_environments-1.32.7-py3-none-any.whl 'kaggle_environments/envs/kaggriculture/*'

Each game uses fuzz agents that send random, partly malformed actions (bad
counts, unknown items, extra hand actions, too many market orders) so the
no-op paths get exercised too. Games alternate between the default
configuration and a small, crowded one (6x6 board, short days, tiny shed,
frequent weeds, price-curve overrides). With --agent, one extra game pits that
Kaggle submission against the fuzzer.

Unlike Kaggle replays, the engine's live state keeps dict order, so carried
inventories are compared in order here.
"""

import argparse
import copy
import importlib.util
import json
import os
import random
import subprocess
import sys
import tempfile
import types

DEFAULT_CFG = {
    "episodeSteps": 720,
    "boardSize": 10,
    "startingMoney": 3000,
    "maxMarketOrdersPerTurn": 10,
    "turnsPerDay": 24,
    "shedCapacity": 100,
    "weedSpawnChance": 0.005,
    "townShopUnlockInterval": 3,
    "townShopSellInterval": 4,
    "townCenterSellInterval": 24,
    "farmHandCostMult": 1,
    "marketParams": {},
}

SMALL_CFG = dict(
    DEFAULT_CFG,
    episodeSteps=300,
    boardSize=6,
    startingMoney=20000,
    maxMarketOrdersPerTurn=4,
    turnsPerDay=8,
    shedCapacity=12,
    weedSpawnChance=0.08,
    townShopUnlockInterval=1,
    townShopSellInterval=3,
    farmHandCostMult=3,
    marketParams={"WHEAT": {"base": 40, "above_func": "hinge"}, "MILK": {"below_func": "log10", "T": 50}},
)

CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
ITEMS = CROPS + ["EGG", "MILK", "WOOL", "FERTILIZER"] + ANIMALS


class Struct(dict):
    """Dict with attribute access, like the objects Kaggle hands to agents."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


def to_struct(v):
    if isinstance(v, dict):
        return Struct({k: to_struct(x) for k, x in v.items()})
    if isinstance(v, list):
        return [to_struct(x) for x in v]
    return v


def load_engine(path):
    utils = types.ModuleType("kaggle_environments.utils")

    def resolve_episode_seed(env, config_key="seed", fallback=None):
        seed = env.info.get("seed")
        if seed is None:
            seed = getattr(env.configuration, config_key, None)
        env.info["seed"] = seed
        return seed

    utils.resolve_episode_seed = resolve_episode_seed
    sys.modules.setdefault("kaggle_environments", types.ModuleType("kaggle_environments"))
    sys.modules["kaggle_environments.utils"] = utils
    spec = importlib.util.spec_from_file_location("kaggriculture_engine", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_submission(path):
    spec = importlib.util.spec_from_file_location("submission_main", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


class Fuzzer:
    def __init__(self, seed):
        self.r = random.Random(seed)

    def count(self):
        return self.r.choice([1, 1, 2, 3, 5, 40, "2", 2.7, True])

    def contextual(self, tile, inv, at_shed, shed):
        """Sensible move for where the unit stands, so animals and fertilizer get used."""
        r = self.r
        if at_shed:
            if inv:
                return ["DROP"] if r.random() < 0.3 else ["PLACE", r.choice(list(inv)), r.choice([1, 2])]
            stocked = [k for k, v in shed.items() if v > 0 and k in ANIMALS + ["WHEAT", "FERTILIZER"]]
            if stocked:
                return ["PICKUP", r.choice(stocked), r.choice([1, 2, 4])]
        if isinstance(tile, dict):
            if tile["kind"] in ("COOP", "PASTURE") and "animal" not in tile:
                return ["PLACE", r.choice(ANIMALS)]
            if "animal" in tile:
                return [r.choice(["FEED", "FEED", "CARE", "HARVEST", "COLLECT_FERTILIZER"])]
            if tile["kind"] == "PLANT":
                return [r.choice(["WATER", "WATER", "HARVEST", "FERTILIZE"])]
        if tile is None:
            return r.choice([["BUILD_COOP"], ["BUILD_PASTURE"], ["PLANT", r.choice(CROPS)]])
        return None

    def unit(self):
        r = self.r
        k = r.random()
        if k < 0.30:
            return [r.choice(["NORTH", "SOUTH", "EAST", "WEST"])]
        if k < 0.42:
            return ["PLANT", r.choice(CROPS + ["CACTUS"])]
        if k < 0.54:
            return ["WATER"]
        if k < 0.60:
            return ["HARVEST"]
        if k < 0.64:
            return ["DROP"]
        if k < 0.70:
            return ["PICKUP", r.choice(ITEMS + ["SEEDS"]), self.count()]
        if k < 0.72:
            return ["PICKUP", r.choice(ITEMS)]
        if k < 0.77:
            return ["PLACE", r.choice(ITEMS), self.count()]
        if k < 0.82:
            return [r.choice(["BUILD_COOP", "BUILD_PASTURE", "DIG"])]
        if k < 0.92:
            return [r.choice(["FEED", "CARE", "COLLECT_FERTILIZER", "FERTILIZE"])]
        if k < 0.95:
            return r.choice([["PASS"], [], ["DANCE"], "NORTH", ["PLANT"], ["PICKUP"]])
        return ["PASS"]

    def order(self):
        r = self.r
        k = r.random()
        n = r.choice([1, 2, 3, 6, 25, "4", 1.9, 0, -2])
        if k < 0.30:
            return ["BUY_SEED", r.choice(CROPS + ["CACTUS"]), n]
        if k < 0.55:
            return ["SELL", r.choice(ITEMS), r.choice([1, 3, 10, 200])]
        if k < 0.68:
            return ["BUY_PRODUCT", r.choice(["WHEAT", "FERTILIZER", "MILK"]), n]
        if k < 0.78:
            return ["BUY_ANIMAL", r.choice(ANIMALS + ["DRAGON"]), r.choice([1, 2])]
        if k < 0.88:
            return ["HIRE"]
        if k < 0.93:
            return ["BUY_LAND"]
        return r.choice([["SELL", "WHEAT"], ["NOPE", "WHEAT", 1], [], "HIRE", ["SELL", "WHEAT", None]])

    def __call__(self, obs, cfg):
        r = self.r
        farm, priv = obs["farms"][obs["player"]], obs["private"]
        half = len(farm["tiles"]) // 2
        units = [farm["farmer"]] + farm["hands"]

        def act(i):
            if i < len(units) and r.random() < 0.6:
                x, y = units[i]
                inv = priv["inventories"][i] if i < len(priv["inventories"]) else {}
                at_shed = x in (half - 1, half) and y in (half - 1, half)
                a = self.contextual(farm["tiles"][y][x], inv, at_shed, priv["shed"])
                if a:
                    return a
            return self.unit()

        hands = [act(i + 1) for i in range(len(farm["hands"]) + (r.random() < 0.1))]
        # Spend only while there is a cushion, so games last long enough to reach animals.
        n_orders = r.choice([0, 0, 1, 2, 3, 12]) if farm["money"] > 1500 else r.choice([0, 1])
        market = [self.order() for _ in range(n_orders)]
        if farm["money"] <= 1500:
            market = [o for o in market if o and o[0] == "SELL"]
        if r.random() < 0.02:
            return {"farmer": "PASS"}
        return {"farmer": act(0), "hands": hands, "market": market}


def snapshot(state):
    obs0 = state[0].observation
    return {
        "day": obs0.day,
        "hour": obs0.hour,
        "farms": copy.deepcopy(obs0.farms),
        "market": {"inventory": dict(obs0.market["inventory"]), "prices": dict(obs0.market["prices"])},
        "town": copy.deepcopy(obs0.town),
        "privates": [
            {
                "shed": dict(s.observation.private["shed"]),
                "seeds": dict(s.observation.private["seeds"]),
                "inventories": [[[k, v] for k, v in inv.items()] for inv in s.observation.private["inventories"]],
            }
            for s in state
        ],
    }


def play(engine, cfg, seed, agents):
    env = types.SimpleNamespace(configuration=types.SimpleNamespace(**cfg, seed=seed), info={}, done=False)
    state = [
        types.SimpleNamespace(observation=types.SimpleNamespace(), action=None, status="ACTIVE", reward=0)
        for _ in range(2)
    ]
    engine.interpreter(state, env)
    state[0].observation.step = 0
    states, actions = [snapshot(state)], []
    for t in range(cfg["episodeSteps"] - 1):
        obs0 = state[0].observation
        turn = []
        for i, agent in enumerate(agents):
            view = {
                "player": i,
                "step": t,
                "day": obs0.day,
                "hour": obs0.hour,
                "farms": obs0.farms,
                "market": obs0.market,
                "town": obs0.town,
                "private": state[i].observation.private,
                "remainingOverageTime": 60,
            }
            a = agent(to_struct(copy.deepcopy(view)), to_struct(cfg))
            turn.append(json.loads(json.dumps(a)))
        for s, a in zip(state, turn):
            s.action = a
        engine.interpreter(state, env)
        obs0.step = t + 1
        states.append(snapshot(state))
        actions.append(turn)
        if all(s.status == "DONE" for s in state):
            break
    return states, actions, [s.reward for s in state]


def first_diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}/{k}: only in {'sim' if k in a else 'engine'}"
            d = first_diff(a[k], b[k], f"{path}/{k}")
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} (sim) vs {len(b)} (engine)"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f"{path}/{i}")
            if d:
                return d
        return None
    if a != b:
        return f"{path}: {a!r} (sim) vs {b!r} (engine)"
    return None


def compare(sim, cfg, seed, states, actions):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"seed": seed, "configuration": cfg, "actions": actions}, f)
    try:
        out = subprocess.run([sim, "replay", f.name], check=True, capture_output=True, text=True).stdout.splitlines()
    finally:
        os.unlink(f.name)
    if len(out) != len(states):
        return f"{len(out)} states from sim vs {len(states)} from engine"
    for t, (line, want) in enumerate(zip(out, states)):
        got = json.loads(line)
        d = first_diff({k: got[k] for k in want}, want)
        if d:
            return f"turn {t}: {d}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sim")
    ap.add_argument("engine")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--agent", help="a Kaggle submission file to play one extra game with")
    args = ap.parse_args()
    engine = load_engine(args.engine)

    games = []
    for g in range(args.games):
        cfg = DEFAULT_CFG if g % 2 == 0 else SMALL_CFG
        games.append((f"fuzz/{'default' if cfg is DEFAULT_CFG else 'small'}", cfg, 1000 + g, [Fuzzer(2 * g), Fuzzer(2 * g + 1)]))
    if args.agent:
        games.append((f"{os.path.basename(args.agent)} vs fuzz", DEFAULT_CFG, 4242, [load_submission(args.agent), Fuzzer(99)]))

    failed = 0
    for name, cfg, seed, agents in games:
        states, actions, rewards = play(engine, cfg, seed, agents)
        err = compare(args.sim, cfg, seed, states, actions)
        print(f"{'FAIL' if err else 'ok  '} {name} seed={seed} turns={len(actions)} rewards={rewards}" + (f"\n     {err}" if err else ""))
        failed += bool(err)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
