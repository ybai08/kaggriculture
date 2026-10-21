//! Price curves. Mirrors `market_price` and `_shape` in the engine, operation
//! for operation, so floating-point results (and banker's rounding) match.

use crate::rules::{Item, N_PRODUCTS};
use serde_json::Value;

pub const MARKET_I0: i64 = 10000;
pub const PRICE_FLOOR: i64 = 1;
pub const HINGE_GAIN: f64 = 8.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    Linear,
    Sq,
    Sqrt,
    Log,
    Log10,
    Hinge,
    /// Unknown names fall through to `return x` in the engine.
    Other,
}

impl Shape {
    pub fn from_name(s: &str) -> Shape {
        match s {
            "linear" => Shape::Linear,
            "sq" => Shape::Sq,
            "sqrt" => Shape::Sqrt,
            "log" => Shape::Log,
            "log10" => Shape::Log10,
            "hinge" => Shape::Hinge,
            _ => Shape::Other,
        }
    }

    pub fn eval(self, x: f64, t: f64) -> f64 {
        let x = x.max(0.0);
        match self {
            Shape::Linear | Shape::Other => x,
            Shape::Sq => x * x,
            Shape::Sqrt => x.sqrt(),
            Shape::Log => (1.0 + x).ln(),
            Shape::Log10 => (1.0 + x).log10(),
            Shape::Hinge => {
                if t <= 0.0 {
                    return x;
                }
                let u = x / t;
                u + HINGE_GAIN * (u - 1.0).max(0.0).powf(2.0)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct MarketParam {
    pub base: f64,
    pub i0: i64,
    pub t: f64,
    pub below_func: Shape,
    pub below_target: f64,
    pub above_func: Shape,
    pub above_target: f64,
}

const fn p(base: f64, t: f64, bf: Shape, bt: f64, af: Shape, at: f64) -> MarketParam {
    MarketParam { base, i0: MARKET_I0, t, below_func: bf, below_target: bt, above_func: af, above_target: at }
}

pub fn default_params() -> [MarketParam; N_PRODUCTS] {
    use Shape::*;
    [
        p(25.0, 400.0, Sqrt, 0.80, Log, 0.20),
        p(35.0, 450.0, Hinge, 1.00, Sqrt, 0.70),
        p(60.0, 200.0, Hinge, 0.40, Sqrt, 0.60),
        p(120.0, 100.0, Sqrt, 0.70, Linear, 1.60),
        p(250.0, 300.0, Log, 0.20, Sq, 3.60),
        p(50.0, 332.0, Hinge, 0.40, Log, 0.20),
        p(160.0, 122.0, Sqrt, 0.60, Linear, 1.60),
        p(200.0, 105.0, Log, 0.20, Sq, 3.20),
        p(100.0, 200.0, Linear, 0.40, Linear, 0.40),
    ]
}

/// Applies a sparse `marketParams` override object onto the defaults.
pub fn resolve_params(overrides: Option<&Value>) -> [MarketParam; N_PRODUCTS] {
    let mut out = default_params();
    let Some(Value::Object(map)) = overrides else { return out };
    for (name, patch) in map {
        let Some(item) = Item::from_name(name).filter(|i| i.is_product()) else { continue };
        let Value::Object(patch) = patch else { continue };
        let mp = &mut out[item.idx()];
        for (k, v) in patch {
            match (k.as_str(), v) {
                ("base", v) => mp.base = v.as_f64().unwrap_or(mp.base),
                ("I0", v) => mp.i0 = v.as_f64().map(|f| f as i64).unwrap_or(mp.i0),
                ("T", v) => mp.t = v.as_f64().unwrap_or(mp.t),
                ("below_func", Value::String(s)) => mp.below_func = Shape::from_name(s),
                ("above_func", Value::String(s)) => mp.above_func = Shape::from_name(s),
                ("below_target", v) => mp.below_target = v.as_f64().unwrap_or(mp.below_target),
                ("above_target", v) => mp.above_target = v.as_f64().unwrap_or(mp.above_target),
                _ => {}
            }
        }
    }
    out
}

/// Price of one unit at the given market inventory, floored at `PRICE_FLOOR`.
pub fn market_price(p: &MarketParam, inventory: i64) -> i64 {
    let price = if inventory < p.i0 {
        let amp = p.below_target * p.base / p.below_func.eval(p.t, p.t);
        p.base + amp * p.below_func.eval((p.i0 - inventory) as f64, p.t)
    } else {
        let amp = p.above_target * p.base / p.above_func.eval(p.t, p.t);
        p.base - amp * p.above_func.eval((inventory - p.i0) as f64, p.t)
    };
    PRICE_FLOOR.max(price.round_ties_even() as i64)
}
