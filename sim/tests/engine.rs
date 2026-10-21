//! Expected values below were produced by the Python engine
//! (kaggle-environments 1.32.7) and CPython 3.13's `random` module.

use kaggsim::market::{default_params, market_price};
use kaggsim::rng::PyRandom;
use kaggsim::rules::*;
use kaggsim::state::Tile;
use kaggsim::{Action, Config, Game, Order, TradeKind, UnitAction};
use serde_json::json;

#[test]
fn rng_matches_cpython() {
    // (seed, random(), random(), choice(range(8)), getrandbits(32))
    let cases: [(i128, f64, f64, u32, u32); 5] = [
        (0, 0.8444218515250481, 0.7579544029403025, 6, 173879092),
        (1, 0.13436424411240122, 0.8474337369372327, 1, 1095513148),
        (12345, 0.41661987254534116, 0.010169169457068361, 4, 3672791226),
        ((257702497 * 1000003) ^ 5, 0.6689788817604095, 0.5154447249082096, 5, 3186552708),
        ((1 << 40) + 7, 0.6137037779936511, 0.8149162973309487, 7, 1077368710),
    ];
    for (seed, r1, r2, c, bits) in cases {
        let mut r = PyRandom::new(seed);
        assert_eq!(r.random(), r1, "seed {seed}");
        assert_eq!(r.random(), r2, "seed {seed}");
        assert_eq!(r.randbelow(8), c, "seed {seed}");
        assert_eq!(r.getrandbits(32), bits, "seed {seed}");
    }
}

#[test]
fn prices_match_engine() {
    let inventories = [10000, 9999, 9600, 9000, 8000, 10001, 10400, 11000, 13000, 30000];
    let expected: [[i64; 10]; N_PRODUCTS] = [
        [25, 26, 45, 57, 70, 24, 20, 19, 18, 17],
        [35, 35, 66, 531, 3513, 34, 12, 1, 1, 1],
        [60, 60, 300, 3252, 15852, 57, 9, 1, 1, 1],
        [120, 128, 288, 386, 496, 118, 1, 1, 1, 1],
        [250, 256, 303, 311, 317, 250, 1, 1, 1, 1],
        [50, 50, 81, 758, 4209, 49, 40, 38, 36, 33],
        [160, 169, 334, 435, 549, 158, 1, 1, 1, 1],
        [200, 206, 251, 259, 265, 200, 1, 1, 1, 1],
        [100, 100, 180, 300, 500, 100, 20, 1, 1, 1],
    ];
    let params = default_params();
    for (i, row) in expected.iter().enumerate() {
        let got: Vec<i64> = inventories.iter().map(|&inv| market_price(&params[i], inv)).collect();
        assert_eq!(got, row.to_vec(), "{}", ITEMS[i].name());
    }
}

fn act(farmer: UnitAction, market: Vec<Order>) -> Action {
    Action { farmer, hands: vec![], market }
}

fn buy(kind: TradeKind, item: Item, n: i64) -> Order {
    Order::Trade { kind, item: Some(item), n }
}

fn step(g: &mut Game, a: Action) {
    g.step(&[a, Action::default()]);
}

/// Plays PASS until the start of the next day.
fn finish_day(g: &mut Game) {
    while g.hour() != 0 || g.step == 0 {
        step(g, Action::default());
        if g.hour() == 0 {
            break;
        }
    }
}

#[test]
fn watered_carrot_grows_and_harvests() {
    let mut g = Game::new(Config::default(), 7);
    step(&mut g, act(UnitAction::Pass, vec![buy(TradeKind::BuySeed, Item::Carrot, 1)]));
    assert_eq!(g.farms[0].money, 3000 - 20);
    step(&mut g, act(UnitAction::Plant(Some(Crop::Carrot)), vec![]));
    let (x, y) = g.farms[0].farmer;
    for _ in 0..3 {
        step(&mut g, act(UnitAction::Water, vec![]));
        finish_day(&mut g);
    }
    // Planted day 0, watered on days 0-2; days 2 and 3 are in the bonus window.
    let Tile::Plant(p) = g.farms[0].tile(x, y) else { panic!("carrot gone") };
    assert_eq!(p.yield_units, 2);
    step(&mut g, act(UnitAction::Water, vec![]));
    step(&mut g, act(UnitAction::Harvest, vec![]));
    assert_eq!(*g.farms[0].tile(x, y), Tile::Empty);
    assert_eq!(g.privates[0].inventories[0].get(Item::Carrot), 3);
}

#[test]
fn unwatered_plant_becomes_weed() {
    let mut g = Game::new(Config::default(), 7);
    step(&mut g, act(UnitAction::Pass, vec![buy(TradeKind::BuySeed, Item::Wheat, 1)]));
    step(&mut g, act(UnitAction::Plant(Some(Crop::Wheat)), vec![]));
    finish_day(&mut g);
    let (x, y) = g.farms[0].farmer;
    assert_eq!(*g.farms[0].tile(x, y), Tile::Weed);
}

#[test]
fn plant_requests_beyond_seed_count_all_fail() {
    let mut g = Game::new(Config::default(), 7);
    step(&mut g, act(UnitAction::Pass, vec![buy(TradeKind::BuySeed, Item::Wheat, 1), Order::Hire]));
    let plant = UnitAction::Plant(Some(Crop::Wheat));
    // One seed, two requests (farmer + hand): neither plants.
    step(&mut g, Action { farmer: plant, hands: vec![UnitAction::Move(1, 0), plant], market: vec![] });
    assert_eq!(g.privates[0].seeds[Crop::Wheat.idx()], 1);
}

#[test]
fn hire_costs_follow_fibonacci() {
    let mut g = Game::new(Config::default(), 7);
    step(&mut g, act(UnitAction::Pass, vec![Order::Hire; 6]));
    assert_eq!(g.farms[0].money, 3000 - (1 + 1 + 2 + 3 + 5 + 8));
    assert_eq!(g.farms[0].hands.len(), 6);
    // Hands fill the four shed-access tiles, least occupied first.
    assert_eq!(g.farms[0].hands[..3], shed_access_tiles(10)[1..]);
}

#[test]
fn land_unlocks_in_order() {
    let mut g = Game::new(Config::default(), 7);
    // 1000 + 2000 spends everything; the third (4000) is unaffordable.
    step(&mut g, act(UnitAction::Pass, vec![Order::BuyLand, Order::BuyLand, Order::BuyLand]));
    assert_eq!(g.farms[0].unlocked_quadrants, vec!["NW", "NE", "SW"]);
    assert_eq!(g.farms[0].money, 0);
    assert_eq!(*g.farms[0].tile(9, 0), Tile::Empty);
    assert_eq!(*g.farms[0].tile(0, 9), Tile::Empty);
    assert_eq!(*g.farms[0].tile(9, 9), Tile::Locked);
}

#[test]
fn sell_moves_price_and_market() {
    let mut g = Game::new(Config::default(), 7);
    step(&mut g, act(UnitAction::Pass, vec![buy(TradeKind::BuyProduct, Item::Wheat, 3)]));
    // Bought at the post-buy inventory: 9999, 9998, 9997.
    assert_eq!(g.farms[0].money, 3000 - 26 - 26 - 27);
    // The town center then buys one wheat on turn 0, so inventory is 9996.
    assert_eq!(g.market.inventory[Item::Wheat.idx()], 9996);
    let wheat = &g.market.params[Item::Wheat.idx()];
    let proceeds: i64 = (9996..9999).map(|inv| market_price(wheat, inv)).sum();
    step(&mut g, act(UnitAction::Pass, vec![buy(TradeKind::Sell, Item::Wheat, 3)]));
    assert_eq!(g.farms[0].money, 3000 - 79 + proceeds);
    assert_eq!(g.market.inventory[Item::Wheat.idx()], 9999);
}

#[test]
fn malformed_json_actions_are_no_ops() {
    let a = Action::parse(&json!({"farmer": "NORTH", "hands": 3, "market": [["SELL", "WHEAT"], ["BUY_SEED", "WHEAT", 0], "HIRE"]}));
    assert_eq!(a.farmer, UnitAction::Noop);
    assert!(a.hands.is_empty());
    assert_eq!(a.market, vec![Order::Invalid; 3]);
    let b = Action::parse(&json!({"market": [["BUY_SEED", "MELON", "2"], ["SELL", "EGG", 2.9]]}));
    assert_eq!(b.farmer, UnitAction::Pass);
    assert_eq!(b.market, vec![buy(TradeKind::BuySeed, Item::Melon, 2), buy(TradeKind::Sell, Item::Egg, 2)]);
}

#[test]
fn game_ends_after_719_turns() {
    let mut g = Game::new(Config::default(), 7);
    let mut turns = 0;
    while !g.done {
        step(&mut g, Action::default());
        turns += 1;
    }
    assert_eq!(turns, 719);
    assert_eq!(g.town.unlocked_shops.len(), MAX_SHOP_INSTANCES);
}
