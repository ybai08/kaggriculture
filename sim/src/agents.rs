//! Built-in agents for self-play and benchmarking.

use crate::action::{Action, Order, TradeKind, UnitAction};
use crate::game::Game;
use crate::rules::*;
use crate::state::Tile;

pub trait Agent {
    fn act(&mut self, game: &Game, player: usize) -> Action;
}

/// Does nothing.
pub struct PassAgent;

impl Agent for PassAgent {
    fn act(&mut self, _: &Game, _: usize) -> Action {
        Action::default()
    }
}

/// Port of the engine's `starter_agent`: one carrot at a time on the spawn tile.
pub struct StarterAgent;

impl Agent for StarterAgent {
    fn act(&mut self, game: &Game, player: usize) -> Action {
        let farm = &game.farms[player];
        let private = &game.privates[player];
        let (fx, fy) = farm.farmer;
        let carrot = Crop::Carrot;
        let seeds = private.seeds[carrot.idx()];
        let shed = private.shed[Item::Carrot.idx()];

        let mut market = Vec::new();
        if shed > 0 {
            market.push(Order::Trade { kind: TradeKind::Sell, item: Some(Item::Carrot), n: shed });
        }
        if seeds == 0 && farm.money >= carrot.spec().seed {
            market.push(Order::Trade { kind: TradeKind::BuySeed, item: Some(Item::Carrot), n: 1 });
        }

        let farmer = match farm.tile(fx, fy) {
            Tile::Empty if seeds > 0 => UnitAction::Plant(Some(carrot)),
            Tile::Plant(p) if p.crop == carrot => {
                if game.day() - p.planted_day >= carrot.spec().max_yield_day {
                    UnitAction::Harvest
                } else if !p.watered_today {
                    UnitAction::Water
                } else {
                    UnitAction::Pass
                }
            }
            _ => UnitAction::Pass,
        };
        Action { farmer, hands: Vec::new(), market }
    }
}

/// Uniformly random legal-looking actions from a seeded xorshift stream.
/// Useful for fuzzing and throughput numbers, not for playing well.
pub struct RandomAgent {
    state: u64,
}

impl RandomAgent {
    pub fn new(seed: u64) -> Self {
        RandomAgent { state: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1 }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len() as u64) as usize]
    }

    fn unit(&mut self) -> UnitAction {
        match self.below(16) {
            0..=4 => {
                let (dx, dy) = self.pick(&[(0, -1), (0, 1), (1, 0), (-1, 0)]);
                UnitAction::Move(dx, dy)
            }
            5 | 6 => UnitAction::Plant(Some(self.pick(&CROPS))),
            7 | 8 => UnitAction::Water,
            9 => UnitAction::Harvest,
            10 => UnitAction::Drop,
            11 => UnitAction::Pickup(Some(self.pick(&ITEMS)), Some(1 + self.below(3) as i64)),
            12 => UnitAction::Place(Some(self.pick(&[Item::Goose, Item::Cow, Item::Sheep])), Some(1)),
            13 => self.pick(&[UnitAction::BuildCoop, UnitAction::BuildPasture, UnitAction::Dig]),
            14 => self.pick(&[UnitAction::Feed, UnitAction::Care, UnitAction::CollectFertilizer, UnitAction::Fertilize]),
            _ => UnitAction::Pass,
        }
    }
}

impl Agent for RandomAgent {
    fn act(&mut self, game: &Game, player: usize) -> Action {
        let farmer = self.unit();
        let hands = (0..game.farms[player].hands.len()).map(|_| self.unit()).collect();
        let mut market = Vec::new();
        for _ in 0..self.below(3) {
            let n = 1 + self.below(4) as i64;
            market.push(match self.below(10) {
                0..=2 => Order::Trade { kind: TradeKind::BuySeed, item: Some(self.pick(&CROPS).item()), n },
                3..=5 => Order::Trade { kind: TradeKind::Sell, item: Some(self.pick(&ITEMS[..N_PRODUCTS])), n },
                6 => Order::Trade { kind: TradeKind::BuyProduct, item: Some(self.pick(&[Item::Wheat, Item::Fertilizer])), n },
                7 => Order::Trade { kind: TradeKind::BuyAnimal, item: Some(self.pick(&[Item::Goose, Item::Cow, Item::Sheep])), n: 1 },
                8 => Order::Hire,
                _ => Order::BuyLand,
            });
        }
        Action { farmer, hands, market }
    }
}

pub fn by_name(name: &str, seed: u64) -> Option<Box<dyn Agent>> {
    match name {
        "pass" => Some(Box::new(PassAgent)),
        "starter" => Some(Box::new(StarterAgent)),
        "random" => Some(Box::new(RandomAgent::new(seed))),
        _ => None,
    }
}
