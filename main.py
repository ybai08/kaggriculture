"""Kaggriculture submission (auto-generated single file).

Built by `uv run python run.py package` from the `farmbot` package; do not edit
by hand. Standard library only. Kaggle calls ``agent(observation, configuration)``.
"""
import json
import math
import time

# ===== farmbot/constants.py =====
"""Game tables copied from the official engine (kaggle-environments 1.32.7).

Values here are verified against ``kaggle_environments/envs/kaggriculture``
by ``tests/test_constants_parity.py``.
"""


CROPS = {
    "WHEAT": {
        "seed": 10,
        "first_yield_day": 2,
        "max_yield_day": 4,
        "interval": 0,
        "max_yield": 6,
        "ongoing": False,
    },
    "CARROT": {
        "seed": 20,
        "first_yield_day": 2,
        "max_yield_day": 3,
        "interval": 0,
        "max_yield": 4,
        "ongoing": False,
    },
    "TOMATO": {
        "seed": 50,
        "first_yield_day": 8,
        "max_yield_day": 8,
        "interval": 1,
        "max_yield": 4,
        "ongoing": True,
    },
    "STRAWBERRY": {
        "seed": 100,
        "first_yield_day": 10,
        "max_yield_day": 10,
        "interval": 2,
        "max_yield": 4,
        "ongoing": True,
    },
    "MELON": {
        "seed": 80,
        "first_yield_day": 10,
        "max_yield_day": 12,
        "interval": 0,
        "max_yield": 6,
        "ongoing": False,
    },
}

ANIMALS = {
    "GOOSE": {
        "cost": 300,
        "structure": "COOP",
        "first_yield_day": 4,
        "interval": 1,
        "max_held": 4,
        "product": "EGG",
    },
    "COW": {
        "cost": 400,
        "structure": "PASTURE",
        "first_yield_day": 8,
        "interval": 2,
        "max_held": 6,
        "product": "MILK",
    },
    "SHEEP": {
        "cost": 500,
        "structure": "PASTURE",
        "first_yield_day": 6,
        "interval": 3,
        "max_held": 6,
        "product": "WOOL",
    },
}

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
CROP_PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMAL_PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}

MARKET_I0 = 10000
PRICE_FLOOR = 1
HINGE_GAIN = 8.0

MARKET_PARAMS = {
    "WHEAT": {
        "base": 25,
        "I0": MARKET_I0,
        "T": 400,
        "below_func": "sqrt",
        "below_target": 0.80,
        "above_func": "log",
        "above_target": 0.20,
    },
    "CARROT": {
        "base": 35,
        "I0": MARKET_I0,
        "T": 450,
        "below_func": "hinge",
        "below_target": 1.00,
        "above_func": "sqrt",
        "above_target": 0.70,
    },
    "TOMATO": {
        "base": 60,
        "I0": MARKET_I0,
        "T": 200,
        "below_func": "hinge",
        "below_target": 0.40,
        "above_func": "sqrt",
        "above_target": 0.60,
    },
    "STRAWBERRY": {
        "base": 120,
        "I0": MARKET_I0,
        "T": 100,
        "below_func": "sqrt",
        "below_target": 0.70,
        "above_func": "linear",
        "above_target": 1.60,
    },
    "MELON": {
        "base": 250,
        "I0": MARKET_I0,
        "T": 300,
        "below_func": "log",
        "below_target": 0.20,
        "above_func": "sq",
        "above_target": 3.60,
    },
    "EGG": {
        "base": 50,
        "I0": MARKET_I0,
        "T": 332,
        "below_func": "hinge",
        "below_target": 0.40,
        "above_func": "log",
        "above_target": 0.20,
    },
    "MILK": {
        "base": 160,
        "I0": MARKET_I0,
        "T": 122,
        "below_func": "sqrt",
        "below_target": 0.60,
        "above_func": "linear",
        "above_target": 1.60,
    },
    "WOOL": {
        "base": 200,
        "I0": MARKET_I0,
        "T": 105,
        "below_func": "log",
        "below_target": 0.20,
        "above_func": "sq",
        "above_target": 3.20,
    },
    "FERTILIZER": {
        "base": 100,
        "I0": MARKET_I0,
        "T": 200,
        "below_func": "linear",
        "below_target": 0.40,
        "above_func": "linear",
        "above_target": 0.40,
    },
}

SHOPS = {
    "BAKERY": ["EGG", "WHEAT"],
    "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE": ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE": ["CARROT"],
    "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}
MAX_SHOP_INSTANCES = 8

LAND_ORDER = ["NE", "SW", "SE"]
LAND_PRICES = [1000, 2000, 4000]

FARMER_MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}


def fib_hire_cost(n_already_today, mult=1):
    """Cost of the next hire when ``n_already_today`` hands were hired today."""
    a, b = 1, 1
    for _ in range(n_already_today):
        a, b = b, a + b
    return mult * a


def cumulative_hire_cost(count, mult=1):
    return sum(fib_hire_cost(i, mult) for i in range(count))


def shape(func, x, T=None):
    x = max(0.0, x)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1.0 + x)
    if func == "log10":
        return math.log10(1.0 + x)
    if func == "hinge":
        if not T or T <= 0:
            return x
        u = x / T
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    return x


def market_price(item, inventory, params=None):
    """Exact copy of the engine's price function (rounded, floored at 1)."""
    p = (params or MARKET_PARAMS)[item]
    base, I0, T = p["base"], p["I0"], p["T"]
    if inventory < I0:
        f = p["below_func"]
        amp = p["below_target"] * base / shape(f, T, T)
        price = base + amp * shape(f, I0 - inventory, T)
    else:
        f = p["above_func"]
        amp = p["above_target"] * base / shape(f, T, T)
        price = base - amp * shape(f, inventory - I0, T)
    return max(PRICE_FLOOR, int(round(price)))


def quadrant_of(x, y, board_size=10):
    half = board_size // 2
    return ("N" if y < half else "S") + ("W" if x < half else "E")


def shed_access_tiles(board_size=10):
    half = board_size // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]

# ===== farmbot/market.py =====
"""Market reasoning: exact per-unit revenue, demand forecasts, sale planning."""



def sale_revenue(item, inventory, quantity, params=None):
    """Revenue for selling ``quantity`` units one at a time from ``inventory``.

    Units sold at the $1 floor do not raise market inventory (engine rule).
    """
    total = 0
    inv = inventory
    for _ in range(quantity):
        price = market_price(item, inv, params)
        total += price
        if price > 1:
            inv += 1
    return total


def units_sellable_above(item, inventory, min_price, limit, params=None):
    """How many units (up to ``limit``) can be sold before price drops below ``min_price``."""
    n = 0
    inv = inventory
    while n < limit:
        price = market_price(item, inv, params)
        if price < min_price:
            break
        n += 1
        if price > 1:
            inv += 1
    return n


def daily_town_demand(shops):
    """Units per day consumed by the town center plus the unlocked shop instances."""
    demand = {item: 1 for item in PRODUCTS if item != "FERTILIZER"}
    demand["FERTILIZER"] = 0
    for shop in shops or ():
        products = SHOPS.get(shop)
        if not products:
            continue
        multiplier = 2 if len(products) == 1 else 1
        for item in products:
            demand[item] += 6 * multiplier
    return demand


def expected_future_demand(item, day, shops, final_day=29, unlock_interval=3, max_shops=8):
    """Expected units of ``item`` the town will consume from ``day`` to the end.

    Known shops consume deterministically. Each not-yet-drawn shop instance is
    an equal draw over the eight shop types, so it contributes the average
    per-day consumption of ``item`` across types.
    """
    current = daily_town_demand(shops)
    days_left = max(0, final_day - day + 1)
    total = current.get(item, 0) * days_left
    drawn = len(shops or ())
    per_type = 0.0
    for products in SHOPS.values():
        if item in products:
            per_type += 6 * (2 if len(products) == 1 else 1)
    average_daily = per_type / len(SHOPS)
    for k in range(drawn + 1, max_shops + 1):
        unlock_day = k * unlock_interval
        if unlock_day > final_day:
            break
        if unlock_day <= day:
            continue
        total += average_daily * (final_day - unlock_day + 1)
    return total


def base_price(item):
    return MARKET_PARAMS[item]["base"]

# ===== farmbot/state.py =====
"""Observation normalization: turn the raw Kaggle observation into a typed view."""



def _setting(configuration, name, default):
    if configuration is None:
        return default
    if hasattr(configuration, "get"):
        value = configuration.get(name, default)
        return default if value is None else value
    return getattr(configuration, name, default)


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Unit:
    __slots__ = ("index", "pos", "inventory")

    def __init__(self, index, pos, inventory):
        self.index = index
        self.pos = (int(pos[0]), int(pos[1]))
        self.inventory = inventory

    def carried(self, item):
        return self.inventory.get(item, 0)

    def total_carried(self):
        return sum(max(0, n) for n in self.inventory.values())


_SHED_DISTANCE = {}


class World:
    """Everything the planner and executor need, computed once per turn."""

    def __init__(self, obs, configuration=None):
        self.obs = obs
        self.player = obs.get("player", 0)
        self.farms = obs["farms"]
        self.me = self.farms[self.player]
        self.opp = self.farms[1 - self.player] if len(self.farms) > 1 else None
        self.tiles = self.me["tiles"]
        self.board_size = len(self.tiles)
        self.half = self.board_size // 2
        self.money = float(self.me.get("money", 0))
        private = obs.get("private") or {}
        self.shed = dict(private.get("shed") or {})
        self.seeds = dict(private.get("seeds") or {})
        inventories = private.get("inventories") or [{}]
        positions = [self.me["farmer"]] + list(self.me.get("hands") or [])
        self.units = []
        for index, pos in enumerate(positions):
            inventory = inventories[index] if index < len(inventories) else {}
            self.units.append(Unit(index, pos, dict(inventory)))
        market = obs.get("market") or {}
        self.prices = dict(market.get("prices") or {})
        self.inventory = dict(market.get("inventory") or {})
        self.shops = list((obs.get("town") or {}).get("unlocked_shops") or [])
        self.unlocked = list(self.me.get("unlocked_quadrants") or ["NW"])
        self.hires_today = int(self.me.get("hires_today", 0))

        self.turns_per_day = max(1, int(_setting(configuration, "turnsPerDay", 24)))
        self.episode_steps = int(_setting(configuration, "episodeSteps", 720))
        self.order_limit = max(1, int(_setting(configuration, "maxMarketOrdersPerTurn", 10)))
        self.shed_capacity = max(1, int(_setting(configuration, "shedCapacity", 100)))
        self.hire_mult = int(_setting(configuration, "farmHandCostMult", 1))
        self.step = int(
            obs.get("step", obs.get("day", 0) * self.turns_per_day + obs.get("hour", 0))
        )
        self.day = self.step // self.turns_per_day
        self.hour = self.step % self.turns_per_day
        # The framework records an initial state; the last decision is episodeSteps - 2.
        self.final_step = self.episode_steps - 2
        self.final_day = self.final_step // self.turns_per_day
        self.turns_left = self.final_step - self.step + 1
        self.hours_left_today = min(self.turns_per_day - self.hour, self.turns_left)
        self.days_left = self.final_day - self.day
        self.lite = False
        self.shed_tiles = shed_access_tiles(self.board_size)
        self.spawn = self.shed_tiles[0]
        self.shed_count = sum(max(0, n) for n in self.shed.values())
        # One classification pass per turn; everything else reads these lists.
        self.plants = []
        self.animals = []
        self.structures = []
        self.weeds = []
        self.empties = []
        self._unlocked = []
        for y in range(self.board_size):
            row = self.tiles[y]
            for x in range(self.board_size):
                t = row[x]
                if t == "LOCKED":
                    continue
                pos = (x, y)
                self._unlocked.append(pos)
                if t is None:
                    self.empties.append(pos)
                elif t.get("kind") == "PLANT":
                    self.plants.append((pos, t))
                elif "animal" in t:
                    self.animals.append((pos, t))
                elif t.get("kind") == "WEED":
                    self.weeds.append(pos)
                else:
                    self.structures.append((pos, t))
        self._shed_distance = _SHED_DISTANCE.get(self.board_size)
        if self._shed_distance is None:
            self._shed_distance = {
                (x, y): min(abs(x - tx) + abs(y - ty) for tx, ty in self.shed_tiles)
                for x in range(self.board_size)
                for y in range(self.board_size)
            }
            _SHED_DISTANCE[self.board_size] = self._shed_distance

    # --- tile helpers -----------------------------------------------------
    def tile(self, pos):
        return self.tiles[pos[1]][pos[0]]

    def is_unlocked(self, pos):
        return self.tile(pos) != "LOCKED"

    def unlocked_tiles(self):
        return self._unlocked

    def shed_distance(self, pos):
        return self._shed_distance[pos]

    def nearest_shed_tile(self, pos):
        return min(self.shed_tiles, key=lambda t: (manhattan(pos, t), t))

    @staticmethod
    def is_plant(tile):
        return isinstance(tile, dict) and tile.get("kind") == "PLANT"

    @staticmethod
    def is_weed(tile):
        return isinstance(tile, dict) and tile.get("kind") == "WEED"

    @staticmethod
    def is_structure(tile):
        return isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE")

    @staticmethod
    def has_animal(tile):
        return isinstance(tile, dict) and "animal" in tile

    def count_roles(self):
        """Counts of live plants by crop and animals by species on our farm."""
        counts = {}
        for y in range(self.board_size):
            for x in range(self.board_size):
                t = self.tiles[y][x]
                if self.is_plant(t):
                    counts[t["crop"]] = counts.get(t["crop"], 0) + 1
                elif self.has_animal(t):
                    counts[t["animal"]] = counts.get(t["animal"], 0) + 1
        return counts

    def animal_count(self):
        return sum(
            1
            for y in range(self.board_size)
            for x in range(self.board_size)
            if self.has_animal(self.tiles[y][x])
        )

    def shed_animals(self):
        return {a: self.shed.get(a, 0) for a in ANIMALS if self.shed.get(a, 0) > 0}

# ===== farmbot/genome.py =====
"""Strategy genome: the tunable description of a production strategy.

Every field has a default (the hand-designed baseline), a search range and a
type, so offline search can mutate genomes without touching the executor.
"""


# name: (default, low, high, kind) ; kind in {"int", "float", "bool"}
GENOME_SPEC = {
    # --- opening (day 0) -------------------------------------------------
    "opening_melon": (6, 0, 20, "int"),
    "opening_wheat": (10, 0, 15, "int"),
    "opening_cows": (2, 0, 4, "int"),
    "opening_sheep": (3, 0, 5, "int"),
    "opening_hands": (4, 1, 8, "int"),
    # --- livestock -------------------------------------------------------
    "target_cows": (9, 0, 16, "int"),
    "target_sheep": (5, 0, 16, "int"),
    "target_geese": (0, 0, 10, "int"),
    "animal_last_buy_day": (18, 5, 26, "int"),
    "animal_cash_reserve": (150, 0, 1500, "int"),
    "feed_buffer_days": (2, 1, 4, "int"),
    "care_animals": (True, 0, 1, "bool"),
    "care_min_price_fraction": (0.25, 0.0, 1.0, "float"),
    "fertilizer_base_reserve": (4, 0, 20, "int"),
    # --- land ------------------------------------------------------------
    "land_ne_day": (5, 0, 29, "int"),
    "land_sw_day": (8, 0, 29, "int"),
    "land_se_day": (99, 0, 99, "int"),
    "land_cash_reserve": (200, 0, 2000, "int"),
    # --- crops -----------------------------------------------------------
    "strawberry_target": (30, 0, 60, "int"),
    "strawberry_start_day": (2, 0, 10, "int"),
    "strawberry_last_plant_day": (13, 5, 20, "int"),
    "tomato_target": (8, 0, 30, "int"),
    "tomato_start_day": (9, 0, 20, "int"),
    "tomato_last_plant_day": (18, 5, 22, "int"),
    "wheat_target": (18, 0, 60, "int"),
    "wheat_last_plant_day": (25, 15, 26, "int"),
    "carrot_target": (12, 0, 60, "int"),
    "carrot_start_day": (22, 10, 27, "int"),
    "carrot_last_plant_day": (26, 15, 27, "int"),
    "melon_last_plant_day": (2, 0, 19, "int"),
    "fertilize_strawberry": (True, 0, 1, "bool"),
    "fertilize_tomato": (True, 0, 1, "bool"),
    "skip_optional_water": (False, 0, 1, "bool"),
    # --- labor -----------------------------------------------------------
    "hands_early": (5, 0, 12, "int"),
    "hands_mid": (13, 0, 15, "int"),
    "hands_late": (10, 0, 15, "int"),
    "hands_mid_start_day": (6, 1, 15, "int"),
    "hands_late_start_day": (26, 15, 29, "int"),
    "hire_cash_reserve": (30, 0, 500, "int"),
    "labor_extra": (1, -2, 4, "int"),
    "seed_cash_reserve": (0, 0, 500, "int"),
    "distance_weight": (4, 1, 12, "int"),
    "urgent_build": (True, 0, 1, "bool"),
    "plan_selector": (False, 0, 1, "bool"),
    "hire_last_hour": (10, 0, 23, "int"),
    # --- selling ---------------------------------------------------------
    "sell_floor_strawberry": (0.45, 0.05, 1.0, "float"),
    "sell_floor_milk": (0.45, 0.05, 1.0, "float"),
    "sell_floor_wool": (0.45, 0.05, 1.0, "float"),
    "sell_floor_melon": (0.35, 0.05, 1.0, "float"),
    "sell_floor_wheat": (0.55, 0.05, 1.0, "float"),
    "sell_floor_fertilizer": (0.45, 0.05, 1.0, "float"),
    "sell_floor_tomato": (0.45, 0.05, 1.0, "float"),
    "sell_floor_carrot": (0.45, 0.05, 1.0, "float"),
    "sell_floor_egg": (0.55, 0.05, 1.0, "float"),
    "sell_floor_cash_relief": (0.5, 0.0, 1.0, "float"),
    "shed_pressure_level": (70, 30, 100, "int"),
    "liquidation_day": (29, 26, 29, "int"),
    "opponent_supply_sensitivity": (0.5, 0.0, 2.0, "float"),
    "opponent_dump_sensitivity": (0.5, 0.0, 2.0, "float"),
    "sheep_require_yarn_store": (True, 0, 1, "bool"),
    "sheep_no_store_cap": (3, 0, 8, "int"),
    "cows_no_milk_shop_cap": (5, 0, 16, "int"),
    "shop_check_day": (9, 3, 24, "int"),
    "sheep_per_yarn_store": (8, 0, 12, "int"),
    "cows_per_milk_shop": (3, 0, 8, "int"),
    "geese_per_egg_shop": (0, 0, 6, "int"),
    "demand_share": (0.6, 0.2, 1.0, "float"),
    "price_boost_threshold": (1.4, 1.0, 3.0, "float"),
    "price_boost_factor": (1.5, 1.0, 3.0, "float"),
}

DEFAULT_GENOME = {name: spec[0] for name, spec in GENOME_SPEC.items()}


class Genome(dict):
    """A dict of strategy parameters with attribute access and validation."""

    def __init__(self, values=None):
        super().__init__(DEFAULT_GENOME)
        if values:
            for key, value in values.items():
                if key in GENOME_SPEC:
                    self[key] = _coerce(key, value)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def to_json(self):
        return json.dumps(self, sort_keys=True)

    @classmethod
    def from_json(cls, text):
        return cls(json.loads(text))


def _coerce(key, value):
    kind = GENOME_SPEC[key][3]
    if kind == "bool":
        return bool(value)
    if kind == "int":
        return int(value)
    return float(value)


def clamp_genome(genome):
    for key, (_default, low, high, kind) in GENOME_SPEC.items():
        if kind == "bool":
            genome[key] = bool(genome[key])
        else:
            genome[key] = _coerce(key, min(high, max(low, genome[key])))
    return genome

# ===== farmbot/selector_weights.py =====
"""Exported plan-selector weights (generated by research/selector.py --export; do not edit)."""

SELECTOR_MODEL = {
    "branches": ["livestock", "strawberry", "staples"],
    "keys": [
        "day",
        "inv_CARROT",
        "inv_EGG",
        "inv_FERTILIZER",
        "inv_MELON",
        "inv_MILK",
        "inv_STRAWBERRY",
        "inv_TOMATO",
        "inv_WHEAT",
        "inv_WOOL",
        "money",
        "opp_CARROT",
        "opp_COW",
        "opp_GOOSE",
        "opp_MELON",
        "opp_SHEEP",
        "opp_STRAWBERRY",
        "opp_TOMATO",
        "opp_WHEAT",
        "opp_quadrants",
        "own_CARROT",
        "own_COW",
        "own_GOOSE",
        "own_MELON",
        "own_SHEEP",
        "own_STRAWBERRY",
        "own_TOMATO",
        "own_WHEAT",
        "own_quadrants",
        "price_CARROT",
        "price_EGG",
        "price_FERTILIZER",
        "price_MELON",
        "price_MILK",
        "price_STRAWBERRY",
        "price_TOMATO",
        "price_WHEAT",
        "price_WOOL",
        "quadrants",
        "shop_BAKERY",
        "shop_BRUNCH_SPOT",
        "shop_FARMERS_MARKET",
        "shop_ICE_CREAM_SHOP",
        "shop_PET_CAFE",
        "shop_PIZZA_SHOP",
        "shop_SMOOTHIE_SHOP",
        "shop_YARN_STORE",
    ],
    "means": [
        9.0,
        -28.526786,
        -24.026786,
        34.553571,
        -9.0,
        -22.982143,
        -36.401786,
        -20.571429,
        -51.0625,
        -5.741071,
        719.683036,
        0.0,
        1.589286,
        0.0,
        4.674107,
        3.25,
        9.428571,
        0.0,
        2.65625,
        1.763393,
        0.0,
        4.53125,
        0.0,
        9.0,
        2.0,
        16.209821,
        0.0,
        4.910714,
        2.0,
        37.236607,
        51.834821,
        93.075893,
        270.0,
        196.366071,
        168.053571,
        62.325893,
        31.799107,
        204.558036,
        2.0,
        0.383929,
        0.334821,
        0.366071,
        0.428571,
        0.334821,
        0.294643,
        0.455357,
        0.401786,
    ],
    "stds": [
        1.0,
        26.222421,
        18.743551,
        9.999857,
        1.0,
        18.98871,
        21.375453,
        16.19949,
        21.803007,
        26.482051,
        506.415116,
        1.0,
        1.257503,
        1.0,
        1.252159,
        1.920286,
        6.729087,
        1.0,
        4.768072,
        0.455423,
        1.0,
        0.687196,
        1.0,
        1.0,
        1.0,
        2.439572,
        1.0,
        1.302837,
        1.0,
        1.830373,
        1.041308,
        2.137787,
        1.0,
        20.907106,
        16.448154,
        1.912377,
        1.721658,
        16.467127,
        1.0,
        0.562925,
        0.550521,
        0.566876,
        0.600807,
        0.558571,
        0.4935,
        0.66666,
        0.581866,
    ],
    "weights": [
        [
            0.0,
            -0.488085,
            0.041309,
            -0.579119,
            0.0,
            -0.002862,
            -0.541216,
            0.108182,
            -0.25138,
            -0.471497,
            0.592483,
            0.0,
            -0.119534,
            0.0,
            -0.270106,
            -0.995624,
            -0.312447,
            0.0,
            -0.013358,
            1.84927,
            0.0,
            -0.374372,
            0.0,
            0.0,
            0.0,
            0.271059,
            0.0,
            0.31657,
            0.0,
            -0.01943,
            -0.041309,
            -0.01966,
            0.0,
            0.199059,
            -1.121229,
            -0.203225,
            0.772457,
            0.083081,
            0.0,
            0.439992,
            -0.480545,
            -0.358307,
            -0.312121,
            0.162381,
            -0.211492,
            0.875631,
            -0.279395,
            1.067985,
        ],
        [
            0.0,
            0.507543,
            0.257092,
            -0.050706,
            0.0,
            0.309023,
            -0.628463,
            0.542765,
            -0.063313,
            -0.233398,
            -0.095503,
            0.0,
            0.289794,
            0.0,
            -0.248361,
            0.910742,
            0.02649,
            0.0,
            0.536648,
            -0.978072,
            0.0,
            0.784759,
            0.0,
            0.0,
            0.0,
            0.395141,
            0.0,
            -0.458196,
            0.0,
            1.128415,
            -0.257092,
            1.170897,
            0.0,
            0.068193,
            0.949354,
            0.795112,
            -0.480851,
            0.513451,
            0.0,
            0.281251,
            0.483022,
            0.223542,
            0.043552,
            -0.495944,
            -0.323831,
            0.2281,
            -0.502448,
            0.268647,
        ],
        [
            0.0,
            -0.019458,
            -0.298402,
            0.629826,
            0.0,
            -0.306161,
            1.169679,
            -0.650947,
            0.314693,
            0.704895,
            -0.49698,
            0.0,
            -0.17026,
            0.0,
            0.518466,
            0.084882,
            0.285957,
            0.0,
            -0.52329,
            -0.871199,
            0.0,
            -0.410387,
            0.0,
            0.0,
            0.0,
            -0.6662,
            0.0,
            0.141626,
            0.0,
            -1.108985,
            0.298402,
            -1.151237,
            0.0,
            -0.267251,
            0.171874,
            -0.591887,
            -0.291606,
            -0.596532,
            0.0,
            -0.721243,
            -0.002477,
            0.134765,
            0.268569,
            0.333563,
            0.535323,
            -1.103731,
            0.781844,
            -1.336632,
        ],
    ],
}

# ===== farmbot/selector.py =====
"""Optional learned plan selector (tiny multinomial logistic regression).

At the branch point (``shop_check_day``, hour 0) the planner may pick one of
three post-opening plans from compact strategic features. Weights are trained
offline (``research/selector.py --export``) and embedded as data in
``selector_weights.py``. Any failure falls back to the fixed default branch.
"""



BRANCHES = {
    "livestock": {
        "sheep_per_yarn_store": 10,
        "cows_per_milk_shop": 5,
        "demand_share": 0.9,
        "strawberry_target": 20,
        "wheat_target": 24,
    },
    "strawberry": {
        "sheep_per_yarn_store": 4,
        "cows_per_milk_shop": 2,
        "demand_share": 0.4,
        "strawberry_target": 45,
        "strawberry_last_plant_day": 14,
        "wheat_target": 14,
    },
    "staples": {
        "sheep_per_yarn_store": 4,
        "cows_per_milk_shop": 2,
        "demand_share": 0.4,
        "strawberry_target": 20,
        "wheat_target": 40,
        "carrot_target": 25,
        "carrot_start_day": 14,
        "price_boost_factor": 2.0,
    },
}
DEFAULT_BRANCH = "livestock"
ROLE_KEYS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "COW", "SHEEP", "GOOSE")


def branch_features(obs, player):
    """Compact strategic features at the branch point (shared with the dataset)."""
    farm = obs["farms"][player]
    opp = obs["farms"][1 - player]
    shops = obs["town"]["unlocked_shops"]
    features = {
        "day": obs["day"],
        "money": farm["money"],
        "quadrants": len(farm["unlocked_quadrants"]),
    }
    for shop in SHOPS:
        features[f"shop_{shop}"] = sum(1 for s in shops if s == shop)
    for item in PRODUCTS:
        features[f"price_{item}"] = obs["market"]["prices"][item]
        features[f"inv_{item}"] = obs["market"]["inventory"][item] - 10000
    for label, tiles in (("own", farm["tiles"]), ("opp", opp["tiles"])):
        counts = {}
        for row in tiles:
            for t in row:
                if isinstance(t, dict):
                    key = t.get("crop") or t.get("animal")
                    if key:
                        counts[key] = counts.get(key, 0) + 1
        for key in ROLE_KEYS:
            features[f"{label}_{key}"] = counts.get(key, 0)
        features[f"{label}_quadrants"] = len(
            (farm if label == "own" else opp)["unlocked_quadrants"]
        )
    return features


def choose_branch(features, model=None):
    """Return the branch name with the highest logit; DEFAULT_BRANCH on any problem."""
    model = model or SELECTOR_MODEL
    try:
        keys = model["keys"]
        x = [
            (float(features.get(k, 0.0)) - m) / s
            for k, m, s in zip(keys, model["means"], model["stds"])
        ]
        x.append(1.0)
        best_name, best_logit = DEFAULT_BRANCH, -math.inf
        for name, weights in zip(model["branches"], model["weights"]):
            logit = sum(w * v for w, v in zip(weights, x))
            if logit > best_logit:
                best_name, best_logit = name, logit
        return best_name if best_name in BRANCHES else DEFAULT_BRANCH
    except Exception:  # noqa: BLE001 - never let the optional model break the game
        return DEFAULT_BRANCH

# ===== farmbot/planner.py =====
"""Strategic layer: daily layout, purchases, hiring and sales.

The planner turns a Genome plus the current World into (a) a role map that
says what each unlocked tile should eventually hold and (b) the market orders
for this turn. It never moves units; that is the executor's job.
"""


ANIMAL_ROLES = ("COW", "SHEEP", "GOOSE")
CROP_ROLE_ORDER = ("STRAWBERRY", "TOMATO", "MELON", "CARROT", "WHEAT")
SELL_FLOOR_KEY = {
    "STRAWBERRY": "sell_floor_strawberry",
    "MILK": "sell_floor_milk",
    "WOOL": "sell_floor_wool",
    "MELON": "sell_floor_melon",
    "WHEAT": "sell_floor_wheat",
    "FERTILIZER": "sell_floor_fertilizer",
    "TOMATO": "sell_floor_tomato",
    "CARROT": "sell_floor_carrot",
    "EGG": "sell_floor_egg",
}


def crop_days_to_first_sale(crop):
    """Days from planting until the first harvest can be sold (same day)."""
    return CROPS[crop]["first_yield_day"]


class Memory:
    """Per-game persistent state shared by planner and executor."""

    def __init__(self):
        self.roles = {}
        self.assignments = {}
        self.last_day = -1
        self.prev_inventory = None
        self.prev_step = None
        self.prev_shops = []
        self.opp_sales = {item: 0 for item in PRODUCTS}
        self.opp_recent = {item: 0.0 for item in PRODUCTS}
        self.our_sales_last_turn = {}
        self.our_buys_last_turn = {}
        self.pending_animals = {}
        self.wheat_feed_bought = 0
        self.step_times = []

    def observe_market(self, world):
        """Infer opponent sales from the market inventory change since last turn.

        inventory[t] - inventory[t-1] = our sales + opponent sales - town consumption
        (units sold at the $1 floor do not enter inventory, so this is a lower bound).
        """
        if self.prev_inventory is not None and self.prev_step == world.step - 1 and not world.lite:
            prev_step = world.step - 1
            shop_tick = prev_step % 4 == 0
            center_tick = prev_step % 24 == 0
            for item in PRODUCTS:
                consumed = 0
                if center_tick and item != "FERTILIZER":
                    consumed += 1
                if shop_tick:
                    for shop in self.prev_shops:
                        products = SHOPS.get(shop, ())
                        if item in products:
                            consumed += 2 if len(products) == 1 else 1
                delta = world.inventory.get(item, 0) - self.prev_inventory.get(item, 0)
                ours = self.our_sales_last_turn.get(item, 0) - self.our_buys_last_turn.get(item, 0)
                opp = delta + consumed - ours
                opp = max(0, opp)
                self.opp_sales[item] += opp
                self.opp_recent[item] = 0.9 * self.opp_recent[item] + opp
        self.prev_inventory = dict(world.inventory)
        self.prev_step = world.step
        self.prev_shops = list(world.shops)

    def remember_orders(self, orders, world):
        self.our_sales_last_turn = {}
        self.our_buys_last_turn = {}
        for order in orders:
            if order[0] == "SELL":
                qty = min(order[2], world.shed.get(order[1], 0) + 200)
                self.our_sales_last_turn[order[1]] = self.our_sales_last_turn.get(order[1], 0) + qty
            elif order[0] == "BUY_PRODUCT":
                self.our_buys_last_turn[order[1]] = (
                    self.our_buys_last_turn.get(order[1], 0) + order[2]
                )


class Planner:
    def __init__(self, genome):
        self.g = genome
        self.branch = None

    def maybe_select_branch(self, world, memory):
        """At the branch point, optionally let the learned selector pick the plan.

        Applies the branch's parameter overrides for the rest of the game; the
        backbone executor is unchanged. Runs once per game.
        """
        if self.branch is not None or not self.g.plan_selector:
            return
        if world.day < self.g.shop_check_day:
            return
        features = branch_features(world.obs, world.player)
        name = choose_branch(features)
        self.branch = name
        memory.branch = name
        for key, value in BRANCHES.get(name, {}).items():
            self.g[key] = value

    # ------------------------------------------------------------------
    # targets
    # ------------------------------------------------------------------
    def animal_targets(self, world):
        g = self.g
        if world.day == 0:
            return {"COW": g.opening_cows, "SHEEP": g.opening_sheep, "GOOSE": 0}
        if world.day > g.animal_last_buy_day:
            return {"COW": 0, "SHEEP": 0, "GOOSE": 0}
        # A late animal must still reach its first yield with a few days to spare.
        targets = {}
        for animal, target in (
            ("COW", g.target_cows),
            ("SHEEP", g.target_sheep),
            ("GOOSE", g.target_geese),
        ):
            if world.days_left < ANIMALS[animal]["first_yield_day"] + 2:
                target = 0
            targets[animal] = target
        # Demand-aware targets: the town's shop draw decides who buys wool, milk and eggs.
        # Each shop instance consumes 6 units/day (12 for the single-product yarn store);
        # a sheep makes ~1.3 wool/day, a cow ~1.5 milk/day, a goose ~2 eggs/day.
        if world.day >= g.shop_check_day:
            yarn = sum(1 for s in world.shops if s == "YARN_STORE")
            milk_shops = sum(
                1 for s in world.shops if s in ("PIZZA_SHOP", "ICE_CREAM_SHOP", "SMOOTHIE_SHOP")
            )
            egg_shops = sum(1 for s in world.shops if s in ("BAKERY", "BRUNCH_SPOT"))
            share = g.demand_share
            if g.sheep_require_yarn_store and yarn == 0:
                targets["SHEEP"] = min(targets["SHEEP"], g.sheep_no_store_cap)
            elif yarn:
                targets["SHEEP"] = max(
                    targets["SHEEP"],
                    int(g.sheep_no_store_cap + share * yarn * g.sheep_per_yarn_store),
                )
            if milk_shops == 0:
                targets["COW"] = min(targets["COW"], g.cows_no_milk_shop_cap)
            else:
                targets["COW"] = max(
                    targets["COW"],
                    int(g.cows_no_milk_shop_cap + share * milk_shops * g.cows_per_milk_shop),
                )
            if egg_shops and g.geese_per_egg_shop:
                targets["GOOSE"] = max(
                    targets["GOOSE"], int(share * egg_shops * g.geese_per_egg_shop)
                )
            cap = len(world.unlocked_tiles()) // 2
            for animal in list(targets):
                targets[animal] = min(targets[animal], cap)
                if world.days_left < ANIMALS[animal]["first_yield_day"] + 2:
                    targets[animal] = 0
        return targets

    def crop_targets(self, world):
        """Desired number of tiles for NEW plantings of each crop today."""
        g = self.g
        day = world.day
        targets = {}
        if day <= g.melon_last_plant_day:
            targets["MELON"] = g.opening_melon
        if g.strawberry_start_day <= day <= g.strawberry_last_plant_day:
            targets["STRAWBERRY"] = g.strawberry_target
        if g.tomato_start_day <= day <= g.tomato_last_plant_day:
            targets["TOMATO"] = g.tomato_target
        if day <= g.wheat_last_plant_day:
            expanded = len(world.unlocked) > 1
            targets["WHEAT"] = (
                g.wheat_target if expanded else max(g.opening_wheat, min(g.wheat_target, 10))
            )
        if g.carrot_start_day <= day <= g.carrot_last_plant_day:
            targets["CARROT"] = g.carrot_target
        # Price-responsive staples: scarcity (town demand above supply) raises the
        # wheat/carrot price well above base; grow more of them while it lasts.
        if day >= g.shop_check_day:
            for crop in ("WHEAT", "CARROT"):
                if crop in targets and world.prices.get(
                    crop, 0
                ) >= g.price_boost_threshold * market_price(crop, 10000):
                    targets[crop] = int(targets[crop] * g.price_boost_factor)
            if (
                "CARROT" not in targets
                and day >= g.carrot_start_day - 6
                and day <= g.carrot_last_plant_day
            ):
                if world.prices.get("CARROT", 0) >= g.price_boost_threshold * market_price(
                    "CARROT", 10000
                ):
                    targets["CARROT"] = g.carrot_target
        # Never plant what cannot be sold before the end.
        for crop in list(targets):
            if crop_days_to_first_sale(crop) > world.days_left:
                targets.pop(crop)
        return targets

    def desired_hands(self, world):
        g = self.g
        if world.day >= g.hands_late_start_day:
            return g.hands_late
        if world.day >= g.hands_mid_start_day:
            return g.hands_mid
        if world.day == 0:
            return g.opening_hands
        return g.hands_early

    # ------------------------------------------------------------------
    # layout
    # ------------------------------------------------------------------
    def update_roles(self, world, memory):
        """Assign a desired role to every unlocked tile (recomputed each turn).

        Existing plants and animals keep their tile. Free tiles are handed out
        nearest-to-shed first: animals (visited up to four times a day), then
        strawberries, tomatoes, melons, carrots and wheat.
        """
        roles = {}
        counts = {}
        for pos, tile in world.plants:
            roles[pos] = tile["crop"]
            counts[tile["crop"]] = counts.get(tile["crop"], 0) + 1
        for pos, tile in world.animals:
            roles[pos] = tile["animal"]
            counts[tile["animal"]] = counts.get(tile["animal"], 0) + 1
        for pos, tile in world.structures:
            kind = tile.get("kind")
            previous = memory.roles.get(pos)
            if previous in ANIMAL_ROLES and ANIMALS[previous]["structure"] == kind:
                roles[pos] = previous
            else:
                roles[pos] = "COW" if kind == "PASTURE" else "GOOSE"
            counts[roles[pos]] = counts.get(roles[pos], 0) + 1
        free = list(world.empties) + list(world.weeds)
        free.sort(key=lambda p: (world.shed_distance(p), manhattan(p, world.spawn), p))

        animal_targets = self.animal_targets(world)
        # Animals already bought but still in the shed also need a structure.
        for animal in ANIMAL_ROLES:
            # Every animal we own (placed, stored or carried) needs a structure, even
            # after purchases have stopped; targets only add tiles beyond that.
            owned = (
                counts.get(animal, 0)
                + world.shed.get(animal, 0)
                + sum(u.carried(animal) for u in world.units)
            )
            need = max(animal_targets.get(animal, 0), owned) - counts.get(animal, 0)
            for _ in range(max(0, need)):
                if not free:
                    break
                roles[free.pop(0)] = animal
        crop_targets = self.crop_targets(world)
        for crop in CROP_ROLE_ORDER:
            need = crop_targets.get(crop, 0) - counts.get(crop, 0)
            for _ in range(max(0, need)):
                if not free:
                    break
                pos = free.pop(0)
                roles[pos] = crop
        for pos in free:
            roles[pos] = None
        memory.roles = roles
        return roles

    # ------------------------------------------------------------------
    # market orders
    # ------------------------------------------------------------------
    def market_orders(self, world, memory, planned_drops, plant_needs, roles):
        """Ordered market list: sales, hires, seeds, feed, then land and animals."""
        g = self.g
        orders = []
        limit = world.order_limit
        liquidating = world.day >= g.liquidation_day or world.days_left == 0

        sells, money = self.sell_orders(world, memory, planned_drops, liquidating)
        max_sells = (
            limit
            if liquidating
            else min(len(sells), 4 if world.hour <= g.hire_last_hour else limit)
        )
        orders.extend(sells[:max_sells])
        if liquidating:
            return orders[:limit]

        animals_total = self.total_animals(world, memory)

        # Hands: cheap and high leverage, hired as early in the day as cash allows.
        hands_now = len(world.units) - 1
        want = min(self.desired_hands(world), max(0, self.workload_hands(world, roles)))
        if world.day == 0:
            want = g.opening_hands
        if world.hour <= g.hire_last_hour:
            hires = 0
            n_today = world.hires_today
            while hands_now + hires < want and len(orders) < limit:
                cost = fib_hire_cost(n_today + hires, world.hire_mult)
                if money - cost < g.hire_cash_reserve:
                    break
                orders.append(["HIRE"])
                money -= cost
                hires += 1

        # Feed: buy only what today's feeding needs beyond what we hold.
        if animals_total and len(orders) < limit and world.hours_left_today >= 2:
            unfed = self.unfed_animals(world)
            carried = sum(u.carried("WHEAT") for u in world.units)
            available = world.shed.get("WHEAT", 0) + carried
            need = unfed - available
            if world.hour < 6:
                # In the morning also cover animals arriving today.
                need = max(need, animals_total - available)
            if need > 0:
                price = market_price("WHEAT", world.inventory.get("WHEAT", 10000) - 1)
                qty = min(need, int(money // max(1, price)))
                if qty > 0 and world.shed_count + qty <= world.shed_capacity:
                    orders.append(["BUY_PRODUCT", "WHEAT", qty])
                    money -= qty * price

        # Seeds for tiles that want planting (bought now, plantable next turn).
        for crop, need in sorted(plant_needs.items(), key=lambda kv: -CROPS[kv[0]]["seed"]):
            if len(orders) >= limit:
                break
            missing = need - world.seeds.get(crop, 0)
            if missing <= 0 or world.hours_left_today < 3:
                continue
            affordable = int((money - g.seed_cash_reserve) // CROPS[crop]["seed"])
            qty = min(missing, affordable, 30)
            if qty > 0:
                orders.append(["BUY_SEED", crop, qty])
                money -= qty * CROPS[crop]["seed"]

        # Land
        if len(orders) < limit:
            bought = len(world.unlocked) - 1
            if bought < len(LAND_ORDER):
                day_gate = (g.land_ne_day, g.land_sw_day, g.land_se_day)[bought]
                price = LAND_PRICES[bought]
                if (
                    world.day >= day_gate
                    and money - price >= g.land_cash_reserve
                    and world.days_left >= 4
                ):
                    orders.append(["BUY_LAND"])
                    money -= price

        # Animals: one purchase per turn keeps cash flow sane.
        targets = self.animal_targets(world)
        counts = world.count_roles()
        for animal in ("SHEEP", "COW", "GOOSE"):
            if len(orders) >= limit:
                break
            have = (
                counts.get(animal, 0)
                + world.shed.get(animal, 0)
                + sum(u.carried(animal) for u in world.units)
                + memory.pending_animals.get(animal, 0)
            )
            if have < targets.get(animal, 0) and world.hour <= 20:
                cost = ANIMALS[animal]["cost"]
                if money - cost >= g.animal_cash_reserve and world.shed_count < world.shed_capacity:
                    orders.append(["BUY_ANIMAL", animal, 1])
                    money -= cost
                    memory.pending_animals[animal] = memory.pending_animals.get(animal, 0) + 1
                    break
        return orders[:limit]

    @staticmethod
    def total_animals(world, memory):
        placed = world.animal_count()
        stored = sum(world.shed.get(a, 0) for a in ANIMAL_ROLES)
        carried = sum(u.carried(a) for u in world.units for a in ANIMAL_ROLES)
        pending = sum(memory.pending_animals.values())
        return placed + stored + carried + pending

    def workload_hands(self, world, roles):
        """Hands needed today from the tile workload (about 2.5 turns per plant chore, 6 per animal)."""
        plants = animals = planned = 0
        for pos, role in roles.items():
            tile = world.tile(pos)
            if world.is_plant(tile):
                plants += 1
            elif world.has_animal(tile) or (world.is_structure(tile) and role in ANIMAL_ROLES):
                animals += 1
            elif role in CROPS:
                planned += 1
        turns = plants * 2.5 + animals * 6.0 + planned * 4.0
        return int(turns / 22.0 + 0.999) - 1 + int(self.g.labor_extra)

    def fertilizer_reserve(self, world):
        """Fertilizer to hold back for strawberry/tomato doses due within two days."""
        g = self.g
        need = 0
        for y in range(world.board_size):
            for x in range(world.board_size):
                tile = world.tiles[y][x]
                if not world.is_plant(tile) or tile.get("fertilized_until_day", -1) >= world.day:
                    continue
                age = world.day - tile["planted_day"]
                crop = tile["crop"]
                if (
                    crop == "STRAWBERRY"
                    and g.fertilize_strawberry
                    and (6 <= age <= 10 or 11 <= age <= 14)
                ):
                    need += 1
                elif crop == "TOMATO" and g.fertilize_tomato and (4 <= age <= 8 or 9 <= age <= 10):
                    need += 1
        ongoing = sum(
            1
            for y in range(world.board_size)
            for x in range(world.board_size)
            if world.is_plant(world.tiles[y][x]) and CROPS[world.tiles[y][x]["crop"]]["ongoing"]
        )
        if ongoing and world.days_left >= 3:
            need = max(need, min(g.fertilizer_base_reserve, ongoing))
        return need

    @staticmethod
    def unfed_animals(world):
        return sum(1 for _pos, tile in world.animals if not tile.get("fed_today"))

    def sell_orders(self, world, memory, planned_drops, liquidating):
        """Choose sales from shed stock plus items units drop this turn."""
        g = self.g
        stock = {}
        for item in PRODUCTS:
            qty = world.shed.get(item, 0) + planned_drops.get(item, 0)
            if qty > 0:
                stock[item] = qty
        if not stock:
            return [], world.money
        # Keep feed wheat unless liquidating.
        feed_reserve = 0 if liquidating else self.total_animals(world, memory) * g.feed_buffer_days
        cash_relief = world.money < 60
        shed_pressure = world.shed_count + sum(planned_drops.values()) >= g.shed_pressure_level
        orders = []
        revenue = 0
        opp_pressure = self.opponent_supply_pressure(world)
        fert_reserve = 0 if liquidating else self.fertilizer_reserve(world)
        for item, qty in sorted(stock.items(), key=lambda kv: -kv[1] * world.prices.get(kv[0], 1)):
            if item == "WHEAT":
                qty = qty - feed_reserve
            elif item == "FERTILIZER":
                qty = qty - fert_reserve
            if qty <= 0:
                continue
            inv = world.inventory.get(item, 10000)
            if liquidating:
                sell = qty
            else:
                base = market_price(item, 10000)
                floor = g[SELL_FLOOR_KEY[item]]
                if cash_relief:
                    floor *= g.sell_floor_cash_relief
                if shed_pressure:
                    floor *= 0.5
                floor *= max(0.3, 1.0 - g.opponent_supply_sensitivity * opp_pressure.get(item, 0.0))
                # If the opponent is dumping this product right now, sell before the crash deepens.
                dump = memory.opp_recent.get(item, 0.0)
                if dump > 0:
                    floor *= max(0.3, 1.0 - g.opponent_dump_sensitivity * min(1.0, dump / 20.0))
                min_price = max(2, int(base * floor))
                sell = units_sellable_above(item, inv, min_price, qty)
                if sell == 0 and shed_pressure and world.shed_count >= world.shed_capacity - 2:
                    sell = qty
            if sell > 0:
                orders.append(["SELL", item, sell])
                revenue += sum(market_price(item, inv + k) for k in range(min(sell, 200)))
        return orders, world.money + revenue

    def opponent_supply_pressure(self, world):
        """Rough 0..1 estimate of how much of each product the opponent will dump."""
        pressure = {}
        if not world.opp:
            return pressure
        counts = {}
        for row in world.opp["tiles"]:
            for tile in row:
                if isinstance(tile, dict):
                    key = tile.get("crop") or tile.get("animal")
                    if key:
                        counts[key] = counts.get(key, 0) + 1
        for crop, n in counts.items():
            product = ANIMAL_PRODUCT.get(crop, crop)
            capacity = {
                "MELON": 10,
                "STRAWBERRY": 25,
                "MILK": 6,
                "WOOL": 5,
                "WHEAT": 40,
                "CARROT": 30,
                "TOMATO": 20,
                "EGG": 10,
            }.get(product, 30)
            pressure[product] = min(1.0, pressure.get(product, 0.0) + n / capacity)
        return pressure

# ===== farmbot/executor.py =====
"""Deterministic task executor: turn tile needs into one action per unit.

Each turn the executor derives *jobs* from the farm state and the planner's
role map (water this plant, feed that cow, plant a seed here...). Units keep
the job they were working on while it is still valid, and free units take
the most urgent remaining job nearest to them. Jobs that need an item in the
unit's hands (wheat to feed, fertilizer, an animal to place) are routed
through the shed with PICKUP jobs. Everything is deterministic.
"""


INPUT_ITEMS = ("WHEAT", "FERTILIZER", "COW", "SHEEP", "GOOSE")
PRIORITY_DIE = 1
PRIORITY_HARVEST = 2
PRIORITY_FEED = 2
PRIORITY_BONUS = 3
PRIORITY_PLANT = 3
PRIORITY_CHORE = 4
PRIORITY_OPTIONAL = 5
PRIORITY_DELIVER = 6


class Job:
    __slots__ = ("priority", "pos", "op", "item", "qty", "key", "plant_crop")

    def __init__(self, priority, pos, op, item=None, qty=1, plant_crop=None):
        self.priority = priority
        self.pos = pos
        self.op = op
        self.item = item
        self.qty = qty
        self.plant_crop = plant_crop
        self.key = (pos, op[0], item)

    def __repr__(self):
        return f"Job(p{self.priority} {self.op} @{self.pos} item={self.item})"


def step_toward(pos, target):
    if pos[0] < target[0]:
        return ["EAST"]
    if pos[0] > target[0]:
        return ["WEST"]
    if pos[1] < target[1]:
        return ["SOUTH"]
    if pos[1] > target[1]:
        return ["NORTH"]
    return ["PASS"]


class Executor:
    def __init__(self, genome):
        self.g = genome

    # ------------------------------------------------------------------
    # job generation
    # ------------------------------------------------------------------
    def plant_jobs(self, world, pos, tile):
        g = self.g
        jobs = []
        crop = tile["crop"]
        data = CROPS[crop]
        age = world.day - tile["planted_day"]
        step = world.step
        mls = tile.get("max_lifespan_step", -1)
        decaying = mls >= 0 and step >= mls
        units = tile.get("yield_units", 0)
        watered = tile.get("watered_today", False)
        final_day = world.day == world.final_day
        # --- harvest ---------------------------------------------------
        harvest = None
        if units > 0 and age >= data["first_yield_day"]:
            if not data["ongoing"]:
                fertilized = tile.get("fertilized_until_day", -1) >= world.day
                cap = (
                    data["max_yield"] if fertilized else {"WHEAT": 4, "CARROT": 3, "MELON": 6}[crop]
                )
                if decaying or age > data["max_yield_day"]:
                    harvest = PRIORITY_DIE  # yield shrinks every other turn from now on
                elif units >= cap:
                    harvest = PRIORITY_HARVEST
                elif age >= data["max_yield_day"]:
                    # Last bonus day: water first (+1 unit), then harvest.
                    harvest = PRIORITY_HARVEST if watered else None
                    if not watered:
                        jobs.append(Job(PRIORITY_HARVEST, pos, ["WATER"]))
                        return jobs
                elif world.days_left == 0:
                    harvest = PRIORITY_HARVEST
            else:
                if decaying:
                    harvest = PRIORITY_DIE
                elif units >= 2 or world.days_left <= 1:
                    harvest = PRIORITY_HARVEST
                elif units >= 1 and mls >= 0 and step >= mls - 2 * world.turns_per_day:
                    harvest = PRIORITY_HARVEST
                elif units >= 1 and world.hours_left_today <= 4:
                    harvest = PRIORITY_CHORE
        if harvest is not None:
            if final_day:
                travel_back = world.shed_distance(pos)
                # need: reach tile, harvest, walk back, drop, and still sell.
                if world.turns_left < 1 + travel_back + 1 + 1:
                    return jobs
            jobs.append(Job(harvest, pos, ["HARVEST"]))
            if not data["ongoing"]:
                return jobs
        # --- dig exhausted ongoing crops --------------------------------
        if data["ongoing"] and units == 0 and mls >= 0 and step >= mls:
            jobs.append(Job(PRIORITY_CHORE, pos, ["DIG"]))
            return jobs
        # --- watering --------------------------------------------------
        if not watered:
            must = tile.get("consecutive_unwatered", 0) >= 1
            useful = self.harvest_possible_before_end(world, tile, age)
            if not useful:
                return jobs
            if data["ongoing"]:
                production_today = False
                days_since_first = world.day + 1 - tile["planted_day"] - data["first_yield_day"]
                if days_since_first >= 0 and days_since_first % max(1, data["interval"]) == 0:
                    production_today = True
                fert = tile.get("fertilized_until_day", -1) >= world.day
                if must:
                    jobs.append(Job(PRIORITY_DIE, pos, ["WATER"]))
                elif production_today and fert:
                    jobs.append(Job(PRIORITY_BONUS, pos, ["WATER"]))
                elif not g.skip_optional_water and not world.lite:
                    jobs.append(Job(PRIORITY_OPTIONAL, pos, ["WATER"]))
            else:
                window_start = (data["max_yield_day"] + 1) // 2
                in_window = (
                    window_start <= age <= data["max_yield_day"] and units < data["max_yield"]
                )
                if must:
                    jobs.append(Job(PRIORITY_DIE, pos, ["WATER"]))
                elif in_window:
                    jobs.append(Job(PRIORITY_BONUS, pos, ["WATER"]))
                elif not g.skip_optional_water and not world.lite and age < data["max_yield_day"]:
                    jobs.append(Job(PRIORITY_OPTIONAL, pos, ["WATER"]))
        # --- fertilizing -----------------------------------------------
        if (
            data["ongoing"]
            and tile.get("fertilized_until_day", -1) < world.day
            and world.days_left >= 1
        ):
            want = False
            if crop == "STRAWBERRY" and g.fertilize_strawberry and age in (9, 10, 13, 14):
                want = True
            if crop == "TOMATO" and g.fertilize_tomato and age in (7, 8, 10):
                want = True
            if want and world.day + 1 <= world.final_day:
                jobs.append(Job(PRIORITY_HARVEST, pos, ["FERTILIZE"], item="FERTILIZER"))
        return jobs

    @staticmethod
    def harvest_possible_before_end(world, tile, age):
        data = CROPS[tile["crop"]]
        if not data["ongoing"]:
            return age + max(0, data["first_yield_day"] - age) <= age + world.days_left and (
                data["first_yield_day"] - age <= world.days_left
            )
        # Ongoing: another production must land before the final day.
        days_since_first = world.day + 1 - tile["planted_day"] - data["first_yield_day"]
        if days_since_first < 0:
            wait = -days_since_first
        else:
            interval = max(1, data["interval"])
            wait = (interval - days_since_first % interval) % interval
        return wait <= world.days_left - 1 or (tile.get("yield_units", 0) > 0)

    def animal_jobs(self, world, pos, tile):
        g = self.g
        jobs = []
        animal = tile["animal"]
        if not tile.get("fed_today"):
            priority = PRIORITY_DIE if tile.get("consecutive_unfed", 0) >= 1 else PRIORITY_FEED
            if world.days_left == 0:
                priority = PRIORITY_OPTIONAL
            jobs.append(Job(priority, pos, ["FEED"], item="WHEAT"))
        if tile.get("yield_units", 0) > 0:
            info = ANIMALS[animal]
            if (
                tile["yield_units"] >= 2
                or tile["yield_units"] >= info["max_held"] - 1
                or world.days_left <= 1
            ):
                jobs.append(Job(PRIORITY_HARVEST, pos, ["HARVEST"]))
            else:
                jobs.append(Job(PRIORITY_CHORE, pos, ["HARVEST"]))
        if tile.get("fertilizer_available") and world.days_left >= 0:
            jobs.append(Job(PRIORITY_BONUS, pos, ["COLLECT_FERTILIZER"]))
        if g.care_animals and not tile.get("cared_today") and world.days_left >= 1:
            product = ANIMALS[animal]["product"]
            price = world.prices.get(product, 0)
            if price >= g.care_min_price_fraction * MARKET_PARAMS[product]["base"]:
                jobs.append(Job(PRIORITY_BONUS, pos, ["CARE"]))
        return jobs

    def build_jobs(self, world, roles, plant_needs):
        """Jobs for empty/weed/structure tiles based on the role map."""
        jobs = []
        animals_available = {
            a: world.shed.get(a, 0) + sum(u.carried(a) for u in world.units) for a in ANIMALS
        }
        structures_ahead = {a: 0 for a in ANIMALS}
        for pos, role in roles.items():
            tile = world.tile(pos)
            if world.is_plant(tile) or world.has_animal(tile):
                continue
            if world.is_weed(tile):
                if role is not None and world.days_left >= 2:
                    jobs.append(Job(PRIORITY_CHORE, pos, ["DIG"]))
                continue
            if world.is_structure(tile):
                kind = tile.get("kind")
                for animal in ANIMALS:
                    if (
                        ANIMALS[animal]["structure"] == kind
                        and animals_available.get(animal, 0) > 0
                        and role == animal
                    ):
                        jobs.append(Job(PRIORITY_HARVEST, pos, ["PLACE", animal], item=animal))
                        animals_available[animal] -= 1
                        break
                continue
            if tile is not None:
                continue
            if role in ANIMALS:
                # Build one structure per animal that is in the shed/hands, plus one ahead if affordable.
                empty_structures = sum(
                    1
                    for p2, r2 in roles.items()
                    if r2 == role
                    and world.is_structure(world.tile(p2))
                    and not world.has_animal(world.tile(p2))
                )
                wanted = animals_available.get(role, 0) + (
                    1 if world.money >= ANIMALS[role]["cost"] else 0
                )
                if empty_structures + structures_ahead[role] < wanted:
                    structures_ahead[role] += 1
                    op = "BUILD_COOP" if ANIMALS[role]["structure"] == "COOP" else "BUILD_PASTURE"
                    # An animal waiting in the shed earns nothing: build for it urgently.
                    priority = (
                        PRIORITY_HARVEST
                        if animals_available.get(role, 0) > 0 and self.g.urgent_build
                        else PRIORITY_CHORE
                    )
                    jobs.append(Job(priority, pos, [op]))
            elif role in CROPS:
                # Planting requires time to water on the same day (planting day counts as unwatered).
                if (
                    world.hours_left_today >= 3
                    and world.days_left >= CROPS[role]["first_yield_day"]
                ):
                    plant_needs[role] = plant_needs.get(role, 0) + 1
                    jobs.append(Job(PRIORITY_PLANT, pos, ["PLANT", role], plant_crop=role))
        return jobs

    # ------------------------------------------------------------------
    # assignment
    # ------------------------------------------------------------------
    def act(self, world, memory, roles):
        """Return (unit_actions, planned_drops, plant_needs)."""
        plant_needs = {}
        jobs = []
        for pos, tile in world.plants:
            jobs.extend(self.plant_jobs(world, pos, tile))
        for pos, tile in world.animals:
            jobs.extend(self.animal_jobs(world, pos, tile))
        jobs.extend(self.build_jobs(world, roles, plant_needs))

        # Respect the atomic PLANT rule: never issue more PLANTs than seeds owned.
        seeds_left = dict(world.seeds)
        actions = [["PASS"] for _ in world.units]
        planned_drops = {}
        busy = set()
        assigned_keys = set()

        # 1. Final-day delivery and end-of-day logistics first.
        for unit in world.units:
            carried_products = {
                k: v for k, v in unit.inventory.items() if v > 0 and k not in INPUT_ITEMS
            }
            if not carried_products:
                continue
            dist = world.shed_distance(unit.pos)
            deliver_now = False
            if world.day == world.final_day and world.turns_left <= dist + 2:
                deliver_now = True
            elif unit.total_carried() >= 8 and world.turns_left > 1:
                deliver_now = False
            if deliver_now:
                target = world.nearest_shed_tile(unit.pos)
                if unit.pos in world.shed_tiles:
                    actions[unit.index] = self.drop_action(world, unit, planned_drops)
                else:
                    actions[unit.index] = step_toward(unit.pos, target)
                busy.add(unit.index)

        # 2. Keep valid persistent assignments.
        job_by_key = {}
        for job in jobs:
            if job.key not in job_by_key or job.priority < job_by_key[job.key].priority:
                job_by_key[job.key] = job
        if memory.last_day != world.day:
            memory.assignments = {}
        new_assignments = {}
        for unit in world.units:
            if unit.index in busy:
                continue
            key = memory.assignments.get(unit.index)
            job = job_by_key.get(key) if key else None
            if job is None or key in assigned_keys:
                continue
            if job.item and unit.carried(job.item) <= 0:
                continue
            if not self.reachable(world, unit, job):
                continue
            self.commit(world, unit, job, actions, seeds_left, new_assignments, assigned_keys, busy)

        # 3. Assignment. Pass A: units already carrying an input item (wheat,
        #    fertilizer, an animal) take the jobs that consume it, nearest first.
        #    Pass B: everyone else takes jobs scored by priority and distance;
        #    a job needing an item nobody carries becomes a shed PICKUP run.
        remaining = sorted(
            (j for j in job_by_key.values() if j.key not in assigned_keys),
            key=lambda j: (j.priority, j.pos),
        )
        weight = self.g.distance_weight
        for unit in world.units:
            if unit.index in busy:
                continue
            carried_items = [item for item in INPUT_ITEMS if unit.carried(item) > 0]
            if not carried_items:
                continue
            best = None
            for job in remaining:
                if job.key in assigned_keys or job.item not in carried_items:
                    continue
                if not self.reachable(world, unit, job):
                    continue
                score = job.priority * weight + manhattan(unit.pos, job.pos)
                if best is None or score < best[0]:
                    best = (score, job)
            if best is not None:
                self.commit(
                    world, unit, best[1], actions, seeds_left, new_assignments, assigned_keys, busy
                )

        reserved_pickups = {}
        covered = set()
        for unit in world.units:
            if unit.index in busy:
                continue
            best = None
            for job in remaining:
                if job.key in assigned_keys or job.key in covered:
                    continue
                if job.op[0] == "PLANT" and seeds_left.get(job.plant_crop, 0) <= 0:
                    continue
                if job.item:
                    stock = world.shed.get(job.item, 0) - reserved_pickups.get(job.item, 0)
                    if stock <= 0:
                        continue
                    dist = world.shed_distance(unit.pos) + manhattan(
                        world.nearest_shed_tile(unit.pos), job.pos
                    )
                    if dist + 2 > world.hours_left_today:
                        continue
                    score = job.priority * weight + world.shed_distance(unit.pos)
                else:
                    if not self.reachable(world, unit, job):
                        continue
                    score = job.priority * weight + manhattan(unit.pos, job.pos)
                if best is None or score < best[0]:
                    best = (score, job)
            if best is None:
                continue
            job = best[1]
            if job.item:
                stock = world.shed.get(job.item, 0) - reserved_pickups.get(job.item, 0)
                same_item = [
                    j
                    for j in remaining
                    if j.item == job.item and j.key not in assigned_keys and j.key not in covered
                ]
                qty = min(stock, len(same_item), 8)
                reserved_pickups[job.item] = reserved_pickups.get(job.item, 0) + qty
                for other in same_item[:qty]:
                    covered.add(other.key)
                if unit.pos in world.shed_tiles:
                    actions[unit.index] = ["PICKUP", job.item, qty]
                else:
                    actions[unit.index] = step_toward(unit.pos, world.nearest_shed_tile(unit.pos))
                busy.add(unit.index)
            else:
                self.commit(
                    world, unit, job, actions, seeds_left, new_assignments, assigned_keys, busy
                )

        # 5. Idle units carrying products deliver them to the shed.
        for unit in world.units:
            if unit.index in busy:
                continue
            carried_products = {
                k: v for k, v in unit.inventory.items() if v > 0 and k not in INPUT_ITEMS
            }
            if carried_products and world.turns_left > 1:
                if unit.pos in world.shed_tiles:
                    actions[unit.index] = self.drop_action(world, unit, planned_drops)
                else:
                    actions[unit.index] = step_toward(unit.pos, world.nearest_shed_tile(unit.pos))
                busy.add(unit.index)
            elif (
                unit.index != 0
                and not carried_products
                and unit.pos not in world.shed_tiles
                and False
            ):
                pass

        memory.assignments = new_assignments
        memory.last_day = world.day
        return actions, planned_drops, plant_needs

    def reachable(self, world, unit, job):
        dist = manhattan(unit.pos, job.pos)
        if job.op[0] == "PLANT":
            return dist + 2 <= world.hours_left_today
        return dist + 1 <= world.hours_left_today

    def commit(self, world, unit, job, actions, seeds_left, new_assignments, assigned_keys, busy):
        if unit.pos == job.pos:
            if job.op[0] == "PLANT":
                if seeds_left.get(job.plant_crop, 0) <= 0:
                    return
                seeds_left[job.plant_crop] -= 1
            actions[unit.index] = list(job.op)
        else:
            actions[unit.index] = step_toward(unit.pos, job.pos)
            new_assignments[unit.index] = job.key
        assigned_keys.add(job.key)
        busy.add(unit.index)

    @staticmethod
    def drop_action(world, unit, planned_drops):
        room = max(0, world.shed_capacity - world.shed_count - sum(planned_drops.values()))
        inputs = {k: v for k, v in unit.inventory.items() if k in INPUT_ITEMS and v > 0}
        products = {k: v for k, v in unit.inventory.items() if k not in INPUT_ITEMS and v > 0}
        if not products:
            return ["PASS"]
        if not inputs and sum(products.values()) <= room:
            for item, qty in products.items():
                planned_drops[item] = planned_drops.get(item, 0) + qty
            return ["DROP"]
        item = max(products, key=lambda k: (products[k] * world.prices.get(k, 1), k))
        qty = min(products[item], room)
        if qty <= 0:
            return ["PASS"]
        planned_drops[item] = planned_drops.get(item, 0) + qty
        return ["PLACE", item, qty]

# ===== farmbot/agent.py =====
"""Top-level agent: glue between observation, planner and executor."""



SAFE_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}


class FarmAgent:
    """One instance per game and player; keeps the per-game memory."""

    def __init__(self, genome=None):
        self.genome = genome if isinstance(genome, Genome) else Genome(genome)
        self.planner = Planner(self.genome)
        self.executor = Executor(self.genome)
        self.memory = Memory()
        self.last_step = -1
        self.lite = False
        self.slow_steps = 0

    def act(self, obs, configuration=None):
        if not obs.get("farms"):
            return dict(SAFE_ACTION)
        world = World(obs, configuration)
        # Runtime guard: if Kaggle's overage budget is nearly spent, or our own steps
        # got slow, drop optional work (extra watering, market inference) for the rest
        # of the game. Never risks a timeout for a marginal gain.
        overage = obs.get("remainingOverageTime")
        if (overage is not None and overage < 15) or self.slow_steps >= 3:
            self.lite = True
        world.lite = self.lite
        if world.step <= self.last_step:
            self.memory = Memory()
        self.last_step = world.step
        memory = self.memory
        memory.observe_market(world)
        if world.hour == 0 or world.day != memory.last_day:
            memory.pending_animals = {}
        self.planner.maybe_select_branch(world, memory)
        roles = self.planner.update_roles(world, memory)
        actions, planned_drops, plant_needs = self.executor.act(world, memory, roles)
        market = self.planner.market_orders(world, memory, planned_drops, plant_needs, roles)
        memory.remember_orders(market, world)
        hands = actions[1:] if len(actions) > 1 else []
        # The observation may list hands that were not in inventories; keep lengths aligned.
        n_hands = len(world.me.get("hands") or [])
        hands = (hands + [["PASS"]] * n_hands)[:n_hands]
        return {"farmer": actions[0], "hands": hands, "market": market}


_INSTANCES = {}


def agent_for_player(obs, genome=None):
    player = obs.get("player", 0)
    step = obs.get("step", 0)
    instance = _INSTANCES.get(player)
    if instance is None or step == 0 or step <= instance.last_step:
        instance = FarmAgent(genome)
        _INSTANCES[player] = instance
    return instance


def run_agent(obs, configuration=None, genome=None, time_budget=0.5):
    """Deterministic policy with a safe fallback if anything goes wrong."""
    started = time.perf_counter()
    instance = None
    try:
        instance = agent_for_player(obs, genome)
        action = instance.act(obs, configuration)
    except Exception:  # noqa: BLE001 - never crash in the competition runtime
        return dict(SAFE_ACTION)
    if time.perf_counter() - started > time_budget and instance is not None:
        instance.slow_steps += 1
    return action


# ---------------------------------------------------------------------------
# Kaggle entry point. Must remain the LAST top-level callable in this file.
# ---------------------------------------------------------------------------
CHAMPION_GENOME = {'animal_cash_reserve': 150,
 'animal_last_buy_day': 18,
 'care_animals': True,
 'care_min_price_fraction': 0.25,
 'carrot_last_plant_day': 27,
 'carrot_start_day': 22,
 'carrot_target': 12,
 'cows_no_milk_shop_cap': 5,
 'cows_per_milk_shop': 3,
 'demand_share': 0.6,
 'distance_weight': 4,
 'feed_buffer_days': 2,
 'fertilize_strawberry': True,
 'fertilize_tomato': False,
 'fertilizer_base_reserve': 4,
 'geese_per_egg_shop': 0,
 'hands_early': 4,
 'hands_late': 9,
 'hands_late_start_day': 25,
 'hands_mid': 14,
 'hands_mid_start_day': 7,
 'hire_cash_reserve': 0,
 'hire_last_hour': 15,
 'labor_extra': 2,
 'land_cash_reserve': 339,
 'land_ne_day': 5,
 'land_se_day': 82,
 'land_sw_day': 15,
 'liquidation_day': 29,
 'melon_last_plant_day': 0,
 'opening_cows': 3,
 'opening_hands': 4,
 'opening_melon': 11,
 'opening_sheep': 3,
 'opening_wheat': 10,
 'opponent_dump_sensitivity': 0.030740757931349838,
 'opponent_supply_sensitivity': 0.34645928794825276,
 'price_boost_factor': 1.4456976844488132,
 'price_boost_threshold': 1.4,
 'seed_cash_reserve': 24,
 'sell_floor_carrot': 0.45,
 'sell_floor_cash_relief': 0.5,
 'sell_floor_egg': 0.55,
 'sell_floor_fertilizer': 0.45,
 'sell_floor_melon': 0.35,
 'sell_floor_milk': 0.45,
 'sell_floor_strawberry': 0.41315651153654875,
 'sell_floor_tomato': 0.223284582239443,
 'sell_floor_wheat': 0.4275185342342801,
 'sell_floor_wool': 0.43611031269089295,
 'shed_pressure_level': 53,
 'sheep_no_store_cap': 3,
 'sheep_per_yarn_store': 8,
 'sheep_require_yarn_store': True,
 'shop_check_day': 8,
 'skip_optional_water': True,
 'strawberry_last_plant_day': 16,
 'strawberry_start_day': 2,
 'strawberry_target': 30,
 'target_cows': 7,
 'target_geese': 0,
 'target_sheep': 2,
 'tomato_last_plant_day': 18,
 'tomato_start_day': 13,
 'tomato_target': 5,
 'wheat_last_plant_day': 24,
 'wheat_target': 8}

_GENOME = Genome(CHAMPION_GENOME)


def agent(obs, configuration=None):
    return run_agent(obs, configuration, _GENOME)
