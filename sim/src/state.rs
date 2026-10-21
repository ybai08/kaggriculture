//! Game state and its JSON form. The JSON mirrors the engine's observation
//! (`farms`, `market`, `town`, per-player `private`) so snapshots can be
//! compared with the Python engine and with Kaggle replays.

use crate::market::MarketParam;
use crate::rules::*;
use serde_json::{json, Map, Value};

#[derive(Clone, Debug, PartialEq)]
pub struct Plant {
    pub crop: Crop,
    pub planted_day: i64,
    pub watered_today: bool,
    pub consecutive_unwatered: i64,
    pub yield_units: i64,
    pub max_lifespan_step: i64,
    pub fertilized_until_day: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AnimalTile {
    pub animal: Animal,
    pub placed_day: i64,
    pub yield_units: i64,
    pub consecutive_unfed: i64,
    pub fed_today: bool,
    pub cared_today: bool,
    pub fertilizer_available: bool,
    pub pending_care_bonus: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Tile {
    Empty,
    Locked,
    Weed,
    Structure(Structure),
    Plant(Plant),
    Animal(AnimalTile),
}

/// A farmer's carried items. Keeps insertion order, like the engine's dict,
/// because the order decides what overflows when the shed fills up.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Inventory {
    pub items: Vec<(Item, i64)>,
}

impl Inventory {
    pub fn get(&self, item: Item) -> i64 {
        self.items.iter().find(|(i, _)| *i == item).map_or(0, |(_, n)| *n)
    }

    pub fn add(&mut self, item: Item, n: i64) {
        match self.items.iter_mut().find(|(i, _)| *i == item) {
            Some((_, c)) => *c += n,
            None => self.items.push((item, n)),
        }
    }

    /// Removes `n` if present; deletes the entry when it reaches zero.
    pub fn take(&mut self, item: Item, n: i64) -> bool {
        let Some(pos) = self.items.iter().position(|(i, _)| *i == item) else { return false };
        if self.items[pos].1 < n {
            return false;
        }
        self.items[pos].1 -= n;
        if self.items[pos].1 == 0 {
            self.items.remove(pos);
        }
        true
    }
}

#[derive(Clone, Debug)]
pub struct Farm {
    pub money: i64,
    pub board_size: i64,
    /// Row-major: `tiles[y * board_size + x]`.
    pub tiles: Vec<Tile>,
    pub farmer: (i64, i64),
    pub hands: Vec<(i64, i64)>,
    pub unlocked_quadrants: Vec<&'static str>,
    pub hires_today: i64,
}

impl Farm {
    pub fn new(board_size: i64, starting_money: i64) -> Farm {
        let mut tiles = Vec::with_capacity((board_size * board_size) as usize);
        for y in 0..board_size {
            for x in 0..board_size {
                tiles.push(if quadrant_of(x, y, board_size) == "NW" { Tile::Empty } else { Tile::Locked });
            }
        }
        Farm {
            money: starting_money,
            board_size,
            tiles,
            farmer: default_spawn(board_size),
            hands: Vec::new(),
            unlocked_quadrants: vec!["NW"],
            hires_today: 0,
        }
    }

    pub fn tile(&self, x: i64, y: i64) -> &Tile {
        &self.tiles[(y * self.board_size + x) as usize]
    }

    pub fn tile_mut(&mut self, x: i64, y: i64) -> &mut Tile {
        &mut self.tiles[(y * self.board_size + x) as usize]
    }

    /// 0 = main farmer, 1.. = hands.
    pub fn unit_pos(&self, idx: usize) -> Option<(i64, i64)> {
        if idx == 0 {
            Some(self.farmer)
        } else {
            self.hands.get(idx - 1).copied()
        }
    }

    pub fn set_unit_pos(&mut self, idx: usize, pos: (i64, i64)) {
        if idx == 0 {
            self.farmer = pos;
        } else {
            self.hands[idx - 1] = pos;
        }
    }
}

#[derive(Clone, Debug)]
pub struct Private {
    pub shed: [i64; N_ITEMS],
    pub seeds: [i64; N_CROPS],
    /// `[main farmer, *hands]`.
    pub inventories: Vec<Inventory>,
}

impl Private {
    pub fn new() -> Private {
        Private { shed: [0; N_ITEMS], seeds: [0; N_CROPS], inventories: vec![Inventory::default()] }
    }

    pub fn shed_total(&self) -> i64 {
        self.shed.iter().sum()
    }

    pub fn inventory(&mut self, idx: usize) -> &mut Inventory {
        while self.inventories.len() <= idx {
            self.inventories.push(Inventory::default());
        }
        &mut self.inventories[idx]
    }
}

impl Default for Private {
    fn default() -> Self {
        Private::new()
    }
}

#[derive(Clone, Debug)]
pub struct Market {
    pub inventory: [i64; N_PRODUCTS],
    pub prices: [i64; N_PRODUCTS],
    pub params: [MarketParam; N_PRODUCTS],
}

#[derive(Clone, Debug, Default)]
pub struct Town {
    /// Indices into `SHOPS`; repeats allowed.
    pub unlocked_shops: Vec<usize>,
}

pub fn tile_json(t: &Tile) -> Value {
    match t {
        Tile::Empty => Value::Null,
        Tile::Locked => json!("LOCKED"),
        Tile::Weed => json!({"kind": "WEED"}),
        Tile::Structure(s) => json!({"kind": s.name()}),
        Tile::Plant(p) => json!({
            "kind": "PLANT",
            "crop": p.crop.name(),
            "planted_day": p.planted_day,
            "watered_today": p.watered_today,
            "consecutive_unwatered": p.consecutive_unwatered,
            "yield_units": p.yield_units,
            "max_lifespan_step": p.max_lifespan_step,
            "fertilized_until_day": p.fertilized_until_day,
        }),
        Tile::Animal(a) => json!({
            "kind": a.animal.spec().structure.name(),
            "animal": a.animal.name(),
            "placed_day": a.placed_day,
            "yield_units": a.yield_units,
            "consecutive_unfed": a.consecutive_unfed,
            "fed_today": a.fed_today,
            "cared_today": a.cared_today,
            "fertilizer_available": a.fertilizer_available,
            "pending_care_bonus": a.pending_care_bonus,
        }),
    }
}

impl Farm {
    pub fn to_json(&self) -> Value {
        let n = self.board_size as usize;
        let rows: Vec<Value> = self.tiles.chunks(n).map(|r| Value::Array(r.iter().map(tile_json).collect())).collect();
        json!({
            "money": self.money as f64,
            "tiles": rows,
            "farmer": [self.farmer.0, self.farmer.1],
            "hands": self.hands.iter().map(|p| json!([p.0, p.1])).collect::<Vec<_>>(),
            "unlocked_quadrants": self.unlocked_quadrants,
            "hires_today": self.hires_today,
        })
    }
}

impl Private {
    /// Inventories are emitted as `[[item, n], ...]` lists so their order can
    /// be compared too.
    pub fn to_json(&self) -> Value {
        let shed: Map<String, Value> = ITEMS.iter().map(|i| (i.name().to_string(), json!(self.shed[i.idx()]))).collect();
        let seeds: Map<String, Value> = CROPS.iter().map(|c| (c.name().to_string(), json!(self.seeds[c.idx()]))).collect();
        let invs: Vec<Value> = self
            .inventories
            .iter()
            .map(|inv| Value::Array(inv.items.iter().map(|(i, n)| json!([i.name(), n])).collect()))
            .collect();
        json!({"shed": shed, "seeds": seeds, "inventories": invs})
    }
}

impl Market {
    pub fn to_json(&self) -> Value {
        let inv: Map<String, Value> =
            ITEMS[..N_PRODUCTS].iter().map(|i| (i.name().to_string(), json!(self.inventory[i.idx()]))).collect();
        let prices: Map<String, Value> =
            ITEMS[..N_PRODUCTS].iter().map(|i| (i.name().to_string(), json!(self.prices[i.idx()]))).collect();
        json!({"inventory": inv, "prices": prices})
    }
}

impl Town {
    pub fn to_json(&self) -> Value {
        json!({"unlocked_shops": self.unlocked_shops.iter().map(|&s| SHOPS[s].0).collect::<Vec<_>>()})
    }
}
