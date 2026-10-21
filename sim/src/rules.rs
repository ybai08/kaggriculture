//! Static game tables, copied from `kaggle_environments/envs/kaggriculture/kaggriculture.py`
//! (kaggle-environments 1.32.7).

/// Every stackable thing a farmer or the shed can hold. The first nine are the
/// market products in the engine's `PRODUCTS` order; the last three are animals.
/// The order is also the engine's shed key order (`PRODUCTS + list(ANIMALS)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Item {
    Wheat,
    Carrot,
    Tomato,
    Strawberry,
    Melon,
    Egg,
    Milk,
    Wool,
    Fertilizer,
    Goose,
    Cow,
    Sheep,
}

pub const N_ITEMS: usize = 12;
pub const N_PRODUCTS: usize = 9;
pub const N_CROPS: usize = 5;

pub const ITEMS: [Item; N_ITEMS] = [
    Item::Wheat,
    Item::Carrot,
    Item::Tomato,
    Item::Strawberry,
    Item::Melon,
    Item::Egg,
    Item::Milk,
    Item::Wool,
    Item::Fertilizer,
    Item::Goose,
    Item::Cow,
    Item::Sheep,
];

impl Item {
    pub fn idx(self) -> usize {
        self as usize
    }

    pub fn name(self) -> &'static str {
        ITEM_NAMES[self as usize]
    }

    pub fn from_name(s: &str) -> Option<Item> {
        ITEM_NAMES.iter().position(|n| *n == s).map(|i| ITEMS[i])
    }

    pub fn is_product(self) -> bool {
        (self as usize) < N_PRODUCTS
    }

    pub fn crop(self) -> Option<Crop> {
        CROPS.get(self as usize).copied()
    }

    pub fn animal(self) -> Option<Animal> {
        match self {
            Item::Goose => Some(Animal::Goose),
            Item::Cow => Some(Animal::Cow),
            Item::Sheep => Some(Animal::Sheep),
            _ => None,
        }
    }
}

pub const ITEM_NAMES: [&str; N_ITEMS] = [
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
    "GOOSE",
    "COW",
    "SHEEP",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Crop {
    Wheat,
    Carrot,
    Tomato,
    Strawberry,
    Melon,
}

pub const CROPS: [Crop; N_CROPS] = [Crop::Wheat, Crop::Carrot, Crop::Tomato, Crop::Strawberry, Crop::Melon];

pub struct CropSpec {
    pub seed: i64,
    pub first_yield_day: i64,
    pub max_yield_day: i64,
    pub interval: i64,
    pub max_yield: i64,
    pub ongoing: bool,
}

const CROP_SPECS: [CropSpec; N_CROPS] = [
    CropSpec { seed: 10, first_yield_day: 2, max_yield_day: 4, interval: 0, max_yield: 6, ongoing: false },
    CropSpec { seed: 20, first_yield_day: 2, max_yield_day: 3, interval: 0, max_yield: 4, ongoing: false },
    CropSpec { seed: 50, first_yield_day: 8, max_yield_day: 8, interval: 1, max_yield: 4, ongoing: true },
    CropSpec { seed: 100, first_yield_day: 10, max_yield_day: 10, interval: 2, max_yield: 4, ongoing: true },
    CropSpec { seed: 80, first_yield_day: 10, max_yield_day: 12, interval: 0, max_yield: 6, ongoing: false },
];

impl Crop {
    pub fn idx(self) -> usize {
        self as usize
    }

    pub fn spec(self) -> &'static CropSpec {
        &CROP_SPECS[self as usize]
    }

    pub fn item(self) -> Item {
        ITEMS[self as usize]
    }

    pub fn name(self) -> &'static str {
        ITEM_NAMES[self as usize]
    }

    pub fn from_name(s: &str) -> Option<Crop> {
        Item::from_name(s).and_then(Item::crop)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Structure {
    Coop,
    Pasture,
}

impl Structure {
    pub fn name(self) -> &'static str {
        match self {
            Structure::Coop => "COOP",
            Structure::Pasture => "PASTURE",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Animal {
    Goose,
    Cow,
    Sheep,
}

pub struct AnimalSpec {
    pub cost: i64,
    pub structure: Structure,
    pub first_yield_day: i64,
    pub interval: i64,
    pub max_held: i64,
    pub product: Item,
}

const ANIMAL_SPECS: [AnimalSpec; 3] = [
    AnimalSpec { cost: 300, structure: Structure::Coop, first_yield_day: 4, interval: 1, max_held: 4, product: Item::Egg },
    AnimalSpec { cost: 400, structure: Structure::Pasture, first_yield_day: 8, interval: 2, max_held: 6, product: Item::Milk },
    AnimalSpec { cost: 500, structure: Structure::Pasture, first_yield_day: 6, interval: 3, max_held: 6, product: Item::Wool },
];

impl Animal {
    pub fn spec(self) -> &'static AnimalSpec {
        &ANIMAL_SPECS[self as usize]
    }

    pub fn item(self) -> Item {
        ITEMS[9 + self as usize]
    }

    pub fn name(self) -> &'static str {
        self.item().name()
    }
}

/// NW is always unlocked; the rest unlock in this order at these prices.
pub const LAND_ORDER: [&str; 3] = ["NE", "SW", "SE"];
pub const LAND_PRICES: [i64; 3] = [1000, 2000, 4000];

/// Shops in `sorted(SHOPS)` order, which is the order `rng.choice` draws from.
pub const SHOPS: [(&str, &[Item]); 8] = [
    ("BAKERY", &[Item::Egg, Item::Wheat]),
    ("BRUNCH_SPOT", &[Item::Egg, Item::Wheat, Item::Strawberry]),
    ("FARMERS_MARKET", &[Item::Wheat, Item::Carrot, Item::Tomato, Item::Strawberry]),
    ("ICE_CREAM_SHOP", &[Item::Strawberry, Item::Milk, Item::Wheat]),
    ("PET_CAFE", &[Item::Carrot]),
    ("PIZZA_SHOP", &[Item::Milk, Item::Tomato, Item::Wheat]),
    ("SMOOTHIE_SHOP", &[Item::Strawberry, Item::Milk]),
    ("YARN_STORE", &[Item::Wool]),
];

pub const MAX_SHOP_INSTANCES: usize = 8;

/// Movement deltas; y grows downward.
pub fn move_delta(op: &str) -> Option<(i64, i64)> {
    match op {
        "NORTH" => Some((0, -1)),
        "SOUTH" => Some((0, 1)),
        "EAST" => Some((1, 0)),
        "WEST" => Some((-1, 0)),
        _ => None,
    }
}

pub fn quadrant_of(x: i64, y: i64, board_size: i64) -> &'static str {
    let half = board_size / 2;
    match (y < half, x < half) {
        (true, true) => "NW",
        (true, false) => "NE",
        (false, true) => "SW",
        (false, false) => "SE",
    }
}

/// The four inner-corner tiles around the shed, in NWSE order.
pub fn shed_access_tiles(board_size: i64) -> [(i64, i64); 4] {
    let h = board_size / 2;
    [(h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h)]
}

pub fn is_shed_adjacent(pos: (i64, i64), board_size: i64) -> bool {
    shed_access_tiles(board_size).contains(&pos)
}

/// First shed-access tile in the NW quadrant (the farmer's daily spawn).
pub fn default_spawn(board_size: i64) -> (i64, i64) {
    shed_access_tiles(board_size)
        .into_iter()
        .find(|&(x, y)| quadrant_of(x, y, board_size) == "NW")
        .unwrap_or((0, 0))
}

/// `_fib(0)=1, _fib(1)=1, _fib(2)=2, ...`
pub fn fib(n: i64) -> i64 {
    let (mut a, mut b) = (1i64, 1i64);
    for _ in 0..n {
        let t = a.saturating_add(b);
        a = b;
        b = t;
    }
    a
}
