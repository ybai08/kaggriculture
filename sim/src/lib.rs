//! Rust port of the Kaggriculture game engine
//! (`kaggle_environments/envs/kaggriculture`, kaggle-environments 1.32.7).
//!
//! `Game::step` plays one turn with the same rules, order of operations,
//! pricing arithmetic and weed/shop RNG as the Python interpreter, so a
//! recorded episode replays to the same state.

pub mod action;
pub mod agents;
pub mod game;
pub mod market;
pub mod rng;
pub mod rules;
pub mod state;

pub use action::{Action, Order, TradeKind, UnitAction};
pub use game::{Config, Game, NUM_AGENTS};
