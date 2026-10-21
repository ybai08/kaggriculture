//! Agent actions, parsed from the same JSON shape Kaggle sends to the engine:
//! `{"farmer": [op, ...], "hands": [[op, ...], ...], "market": [[op, ...], ...]}`.
//!
//! Anything the engine treats as a silent no-op parses to `UnitAction::Noop` or
//! `Order::Invalid`. Inputs that would raise inside the Python engine (for
//! example `["PICKUP", "WHEAT", "abc"]`) also become no-ops here.

use crate::rules::{move_delta, Crop, Item};
use serde_json::{json, Value};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UnitAction {
    Noop,
    Pass,
    Move(i64, i64),
    Drop,
    /// `None` count means the count failed to parse.
    Pickup(Option<Item>, Option<i64>),
    Place(Option<Item>, Option<i64>),
    Plant(Option<Crop>),
    Water,
    Harvest,
    Fertilize,
    Dig,
    BuildCoop,
    BuildPasture,
    Feed,
    CollectFertilizer,
    Care,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TradeKind {
    Sell,
    BuyProduct,
    BuySeed,
    BuyAnimal,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Order {
    Hire,
    BuyLand,
    Trade { kind: TradeKind, item: Option<Item>, n: i64 },
    Invalid,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Action {
    pub farmer: UnitAction,
    pub hands: Vec<UnitAction>,
    pub market: Vec<Order>,
}

impl Default for Action {
    fn default() -> Self {
        Action { farmer: UnitAction::Pass, hands: Vec::new(), market: Vec::new() }
    }
}

/// Python's `int(v)` for the argument types JSON can carry. `None` where
/// Python would raise.
pub fn py_int(v: &Value) -> Option<i64> {
    match v {
        Value::Bool(b) => Some(*b as i64),
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().filter(|f| f.is_finite()).map(|f| f.trunc() as i64)),
        Value::String(s) => {
            let t = s.trim();
            if t.starts_with('_') || t.ends_with('_') || t.contains("__") {
                return None;
            }
            t.replace('_', "").parse::<i64>().ok()
        }
        _ => None,
    }
}

fn item_arg(a: &[Value], i: usize) -> Option<Item> {
    a.get(i).and_then(Value::as_str).and_then(Item::from_name)
}

/// Optional count at index 2: defaults to 1 when absent.
fn count_arg(a: &[Value]) -> Option<i64> {
    match a.get(2) {
        None => Some(1),
        Some(v) => py_int(v),
    }
}

impl UnitAction {
    pub fn parse(v: &Value) -> UnitAction {
        let Some(a) = v.as_array() else { return UnitAction::Noop };
        let Some(op) = a.first().and_then(Value::as_str) else { return UnitAction::Noop };
        if let Some((dx, dy)) = move_delta(op) {
            return UnitAction::Move(dx, dy);
        }
        match op {
            "PASS" => UnitAction::Pass,
            "DROP" => UnitAction::Drop,
            "PICKUP" if a.len() >= 2 => UnitAction::Pickup(item_arg(a, 1), count_arg(a)),
            "PLACE" if a.len() >= 2 => UnitAction::Place(item_arg(a, 1), count_arg(a)),
            "PLANT" if a.len() >= 2 => UnitAction::Plant(a[1].as_str().and_then(Crop::from_name)),
            "WATER" => UnitAction::Water,
            "HARVEST" => UnitAction::Harvest,
            "FERTILIZE" => UnitAction::Fertilize,
            "DIG" => UnitAction::Dig,
            "BUILD_COOP" => UnitAction::BuildCoop,
            "BUILD_PASTURE" => UnitAction::BuildPasture,
            "FEED" => UnitAction::Feed,
            "COLLECT_FERTILIZER" => UnitAction::CollectFertilizer,
            "CARE" => UnitAction::Care,
            _ => UnitAction::Noop,
        }
    }

    pub fn to_json(&self) -> Value {
        let item = |i: &Option<Item>| i.map(|i| i.name()).unwrap_or("?");
        match self {
            UnitAction::Noop => json!([]),
            UnitAction::Pass => json!(["PASS"]),
            UnitAction::Move(0, -1) => json!(["NORTH"]),
            UnitAction::Move(0, 1) => json!(["SOUTH"]),
            UnitAction::Move(1, 0) => json!(["EAST"]),
            UnitAction::Move(_, _) => json!(["WEST"]),
            UnitAction::Drop => json!(["DROP"]),
            UnitAction::Pickup(i, n) => json!(["PICKUP", item(i), n.unwrap_or(0)]),
            UnitAction::Place(i, n) => json!(["PLACE", item(i), n.unwrap_or(0)]),
            UnitAction::Plant(c) => json!(["PLANT", c.map(|c| c.name()).unwrap_or("?")]),
            UnitAction::Water => json!(["WATER"]),
            UnitAction::Harvest => json!(["HARVEST"]),
            UnitAction::Fertilize => json!(["FERTILIZE"]),
            UnitAction::Dig => json!(["DIG"]),
            UnitAction::BuildCoop => json!(["BUILD_COOP"]),
            UnitAction::BuildPasture => json!(["BUILD_PASTURE"]),
            UnitAction::Feed => json!(["FEED"]),
            UnitAction::CollectFertilizer => json!(["COLLECT_FERTILIZER"]),
            UnitAction::Care => json!(["CARE"]),
        }
    }
}

impl Order {
    pub fn parse(v: &Value) -> Order {
        let Some(a) = v.as_array() else { return Order::Invalid };
        let Some(op) = a.first().and_then(Value::as_str) else { return Order::Invalid };
        let kind = match op {
            "HIRE" => return Order::Hire,
            "BUY_LAND" => return Order::BuyLand,
            "SELL" => TradeKind::Sell,
            "BUY_PRODUCT" => TradeKind::BuyProduct,
            "BUY_SEED" => TradeKind::BuySeed,
            "BUY_ANIMAL" => TradeKind::BuyAnimal,
            _ => return Order::Invalid,
        };
        if a.len() < 3 {
            return Order::Invalid;
        }
        match py_int(&a[2]) {
            Some(n) if n > 0 => Order::Trade { kind, item: item_arg(a, 1), n },
            _ => Order::Invalid,
        }
    }

    pub fn to_json(&self) -> Value {
        match self {
            Order::Hire => json!(["HIRE"]),
            Order::BuyLand => json!(["BUY_LAND"]),
            Order::Invalid => json!([]),
            Order::Trade { kind, item, n } => {
                let op = match kind {
                    TradeKind::Sell => "SELL",
                    TradeKind::BuyProduct => "BUY_PRODUCT",
                    TradeKind::BuySeed => "BUY_SEED",
                    TradeKind::BuyAnimal => "BUY_ANIMAL",
                };
                json!([op, item.map(|i| i.name()).unwrap_or("?"), n])
            }
        }
    }
}

impl Action {
    pub fn parse(v: &Value) -> Action {
        let Some(obj) = v.as_object() else { return Action::default() };
        let farmer = match obj.get("farmer") {
            None => UnitAction::Pass,
            Some(f) => UnitAction::parse(f),
        };
        let hands = match obj.get("hands") {
            Some(Value::Array(h)) => h.iter().map(UnitAction::parse).collect(),
            _ => Vec::new(),
        };
        let market = match obj.get("market") {
            Some(Value::Array(m)) => m.iter().map(Order::parse).collect(),
            _ => Vec::new(),
        };
        Action { farmer, hands, market }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "farmer": self.farmer.to_json(),
            "hands": self.hands.iter().map(UnitAction::to_json).collect::<Vec<_>>(),
            "market": self.market.iter().map(Order::to_json).collect::<Vec<_>>(),
        })
    }
}
