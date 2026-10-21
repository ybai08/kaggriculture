//! The turn loop. Each function mirrors the engine function named in its doc
//! comment, in the same order of operations.

use crate::action::{Action, Order, TradeKind, UnitAction};
use crate::market::{market_price, resolve_params};
use crate::rng::PyRandom;
use crate::rules::*;
use crate::state::*;
use serde_json::{json, Value};

pub const NUM_AGENTS: usize = 2;

#[derive(Clone, Debug)]
pub struct Config {
    pub episode_steps: i64,
    pub board_size: i64,
    pub starting_money: i64,
    pub max_market_orders: usize,
    pub turns_per_day: i64,
    pub shed_capacity: i64,
    pub weed_spawn_chance: f64,
    pub town_shop_unlock_interval: i64,
    pub town_shop_sell_interval: i64,
    pub town_center_sell_interval: i64,
    pub farm_hand_cost_mult: i64,
    pub market_params: Option<Value>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            episode_steps: 720,
            board_size: 10,
            starting_money: 3000,
            max_market_orders: 10,
            turns_per_day: 24,
            shed_capacity: 100,
            weed_spawn_chance: 0.005,
            town_shop_unlock_interval: 3,
            town_shop_sell_interval: 4,
            town_center_sell_interval: 24,
            farm_hand_cost_mult: 1,
            market_params: None,
        }
    }
}

impl Config {
    /// Reads a Kaggle `configuration` object; missing keys keep defaults.
    pub fn from_json(v: &Value) -> Config {
        let mut c = Config::default();
        let int = |k: &str, d: i64| v.get(k).and_then(crate::action::py_int).unwrap_or(d);
        c.episode_steps = int("episodeSteps", c.episode_steps);
        c.board_size = int("boardSize", c.board_size);
        c.starting_money = int("startingMoney", c.starting_money);
        c.max_market_orders = int("maxMarketOrdersPerTurn", 10).max(1) as usize;
        c.turns_per_day = int("turnsPerDay", 24).max(1);
        c.shed_capacity = int("shedCapacity", c.shed_capacity);
        c.weed_spawn_chance = v.get("weedSpawnChance").and_then(Value::as_f64).unwrap_or(c.weed_spawn_chance);
        c.town_shop_unlock_interval = int("townShopUnlockInterval", 3).max(1);
        c.town_shop_sell_interval = int("townShopSellInterval", 4).max(1);
        c.town_center_sell_interval = int("townCenterSellInterval", 24).max(1);
        c.farm_hand_cost_mult = int("farmHandCostMult", c.farm_hand_cost_mult);
        c.market_params = v.get("marketParams").filter(|m| m.as_object().is_some_and(|o| !o.is_empty())).cloned();
        c
    }
}

#[derive(Clone, Debug)]
pub struct Game {
    pub cfg: Config,
    pub seed: i64,
    /// Number of turns already played; the next `step()` plays turn `step`.
    pub step: i64,
    pub farms: Vec<Farm>,
    pub privates: Vec<Private>,
    pub market: Market,
    pub town: Town,
    pub done: bool,
}

impl Game {
    /// `_initialize`
    pub fn new(cfg: Config, seed: i64) -> Game {
        let params = resolve_params(cfg.market_params.as_ref());
        let mut market = Market { inventory: [0; N_PRODUCTS], prices: [0; N_PRODUCTS], params };
        for i in 0..N_PRODUCTS {
            market.inventory[i] = market.params[i].i0;
            market.prices[i] = market.params[i].base as i64;
        }
        Game {
            farms: (0..NUM_AGENTS).map(|_| Farm::new(cfg.board_size, cfg.starting_money)).collect(),
            privates: (0..NUM_AGENTS).map(|_| Private::new()).collect(),
            market,
            town: Town::default(),
            cfg,
            seed,
            step: 0,
            done: false,
        }
    }

    pub fn day(&self) -> i64 {
        self.step / self.cfg.turns_per_day
    }

    pub fn hour(&self) -> i64 {
        self.step % self.cfg.turns_per_day
    }

    pub fn price(&self, item: Item) -> i64 {
        market_price(&self.market.params[item.idx()], self.market.inventory[item.idx()])
    }

    pub fn scores(&self) -> Vec<i64> {
        self.farms.iter().map(|f| f.money).collect()
    }

    /// `interpreter`: plays one turn with one action per player.
    pub fn step(&mut self, actions: &[Action]) {
        if self.done {
            return;
        }
        let step = self.step;
        let tpd = self.cfg.turns_per_day;
        let day = step / tpd;
        let default = Action::default();

        for p in 0..NUM_AGENTS {
            let a = actions.get(p).unwrap_or(&default);
            // Atomic PLANT check: if this turn's PLANT requests for a crop exceed
            // the seeds on hand, every PLANT of that crop becomes PASS.
            let mut demand = [0i64; N_CROPS];
            for ua in std::iter::once(&a.farmer).chain(a.hands.iter()) {
                if let UnitAction::Plant(Some(c)) = ua {
                    demand[c.idx()] += 1;
                }
            }
            let seeds = self.privates[p].seeds;
            let blocked = |ua: &UnitAction| matches!(ua, UnitAction::Plant(Some(c)) if demand[c.idx()] > seeds[c.idx()]);

            if !blocked(&a.farmer) {
                self.apply_unit(p, 0, &a.farmer, day);
            }
            for (h, ha) in a.hands.iter().enumerate() {
                if !blocked(ha) {
                    self.apply_unit(p, h + 1, ha, day);
                }
            }
        }

        self.process_market(actions);
        self.town_consume(step);
        for f in &mut self.farms {
            decay_plants(f, step);
        }
        if (step + 1) % tpd == 0 {
            self.end_of_day(day);
        }
        self.step = step + 1;
        if step >= self.cfg.episode_steps - 2 {
            self.done = true;
        }
    }

    /// `_apply_unit_action`
    fn apply_unit(&mut self, p: usize, idx: usize, action: &UnitAction, day: i64) {
        let bs = self.cfg.board_size;
        let tpd = self.cfg.turns_per_day;
        let cap = self.cfg.shed_capacity;
        let farm = &mut self.farms[p];
        let private = &mut self.privates[p];
        if matches!(action, UnitAction::Noop) {
            return;
        }
        let Some((fx, fy)) = farm.unit_pos(idx) else { return };
        private.inventory(idx);

        match *action {
            UnitAction::Move(dx, dy) => {
                let (nx, ny) = (fx + dx, fy + dy);
                if (0..bs).contains(&nx) && (0..bs).contains(&ny) {
                    farm.set_unit_pos(idx, (nx, ny));
                }
                return;
            }
            UnitAction::Pass => return,
            UnitAction::Drop => {
                if !is_shed_adjacent((fx, fy), bs) {
                    return;
                }
                let items = std::mem::take(&mut private.inventories[idx].items);
                for (item, n) in items {
                    if n <= 0 {
                        continue;
                    }
                    let take = n.min((cap - private.shed_total()).max(0));
                    if take > 0 {
                        private.shed[item.idx()] += take;
                    }
                }
                return;
            }
            UnitAction::Pickup(item, n) => {
                if !is_shed_adjacent((fx, fy), bs) {
                    return;
                }
                let Some(n) = n else { return };
                if n <= 0 {
                    return;
                }
                let Some(item) = item else { return };
                let n = n.min(private.shed[item.idx()]);
                if n <= 0 {
                    return;
                }
                private.shed[item.idx()] -= n;
                private.inventories[idx].add(item, n);
                return;
            }
            UnitAction::Place(item, n) => {
                if let Some(animal) = item.and_then(Item::animal) {
                    if let Tile::Structure(s) = farm.tile(fx, fy) {
                        if *s == animal.spec().structure {
                            if private.inventories[idx].take(animal.item(), 1) {
                                *farm.tile_mut(fx, fy) = Tile::Animal(new_animal(animal, day));
                            }
                            return;
                        }
                    }
                }
                if is_shed_adjacent((fx, fy), bs) {
                    let Some(n) = n else { return };
                    if n <= 0 {
                        return;
                    }
                    let Some(item) = item else { return };
                    let inv = &mut private.inventories[idx];
                    let n = n.min(inv.get(item));
                    if n <= 0 {
                        return;
                    }
                    let n = n.min((cap - private.shed_total()).max(0));
                    if n <= 0 {
                        return;
                    }
                    private.inventories[idx].take(item, n);
                    private.shed[item.idx()] += n;
                }
                return;
            }
            _ => {}
        }

        // Everything below changes the tile under the unit, which must be owned.
        let tile = farm.tile_mut(fx, fy);
        if *tile == Tile::Locked {
            return;
        }
        let inv = &mut private.inventories[idx];

        match *action {
            UnitAction::Plant(Some(crop)) => {
                if *tile != Tile::Empty || private.seeds[crop.idx()] <= 0 {
                    return;
                }
                private.seeds[crop.idx()] -= 1;
                *tile = Tile::Plant(new_plant(crop, day, tpd));
            }
            UnitAction::Water => {
                let Tile::Plant(pl) = tile else { return };
                if pl.watered_today {
                    return;
                }
                pl.watered_today = true;
                let cd = pl.crop.spec();
                if !cd.ongoing {
                    let age = day - pl.planted_day;
                    let window_start = (cd.max_yield_day + 1) / 2;
                    if window_start <= age && age <= cd.max_yield_day {
                        let bonus = if pl.fertilized_until_day >= day { 2 } else { 1 };
                        pl.yield_units = cd.max_yield.min(pl.yield_units + bonus);
                    }
                }
            }
            UnitAction::Harvest => match tile {
                Tile::Plant(pl) => {
                    if pl.yield_units <= 0 {
                        return;
                    }
                    let cd = pl.crop.spec();
                    if day - pl.planted_day < cd.first_yield_day {
                        return;
                    }
                    inv.add(pl.crop.item(), pl.yield_units);
                    pl.yield_units = 0;
                    if !cd.ongoing {
                        *tile = Tile::Empty;
                    }
                }
                Tile::Animal(a) => {
                    if a.yield_units <= 0 {
                        return;
                    }
                    inv.add(a.animal.spec().product, a.yield_units);
                    a.yield_units = 0;
                }
                _ => {}
            },
            UnitAction::Fertilize => {
                let Tile::Plant(pl) = tile else { return };
                if !inv.take(Item::Fertilizer, 1) {
                    return;
                }
                // Active for day, day+1, day+2.
                pl.fertilized_until_day = pl.fertilized_until_day.max(day + 2);
            }
            UnitAction::Dig => {
                if !matches!(tile, Tile::Empty | Tile::Animal(_)) {
                    *tile = Tile::Empty;
                }
            }
            UnitAction::BuildCoop => {
                if *tile == Tile::Empty {
                    *tile = Tile::Structure(Structure::Coop);
                }
            }
            UnitAction::BuildPasture => {
                if *tile == Tile::Empty {
                    *tile = Tile::Structure(Structure::Pasture);
                }
            }
            UnitAction::Feed => {
                let Tile::Animal(a) = tile else { return };
                if a.fed_today || !inv.take(Item::Wheat, 1) {
                    return;
                }
                a.fed_today = true;
            }
            UnitAction::CollectFertilizer => {
                let Tile::Animal(a) = tile else { return };
                if !a.fertilizer_available {
                    return;
                }
                a.fertilizer_available = false;
                inv.add(Item::Fertilizer, 1);
            }
            UnitAction::Care => {
                if let Tile::Animal(a) = tile {
                    a.cared_today = true;
                }
            }
            _ => {}
        }
    }

    /// `_process_market`: order slot by slot; within a slot both players are
    /// quoted against the same inventory, then committed in player order, one
    /// unit at a time.
    fn process_market(&mut self, actions: &[Action]) {
        let max_orders = self.cfg.max_market_orders;
        let queues: Vec<&[Order]> = (0..NUM_AGENTS)
            .map(|p| actions.get(p).map_or(&[][..], |a| &a.market[..a.market.len().min(max_orders)]))
            .collect();
        let max_len = queues.iter().map(|q| q.len()).max().unwrap_or(0);

        for i in 0..max_len {
            // (kind, item, remaining) per player, None once the order is spent or aborted.
            let mut live: [Option<(TradeKind, Option<Item>, i64)>; NUM_AGENTS] = [None; NUM_AGENTS];
            for (p, q) in queues.iter().enumerate() {
                match q.get(i) {
                    Some(Order::Hire) => self.do_hire(p),
                    Some(Order::BuyLand) => self.do_buy_land(p),
                    Some(Order::Trade { kind, item, n }) => live[p] = Some((*kind, *item, *n)),
                    _ => {}
                }
            }

            let mut guard = 0;
            loop {
                guard += 1;
                if guard >= 100_000 {
                    break;
                }
                let mut quoted: [Option<(TradeKind, Item, i64)>; NUM_AGENTS] = [None; NUM_AGENTS];
                for p in 0..NUM_AGENTS {
                    let Some((kind, item, remaining)) = live[p] else { continue };
                    if remaining <= 0 {
                        continue;
                    }
                    let valid = item.filter(|it| match kind {
                        TradeKind::Sell => it.is_product(),
                        TradeKind::BuyProduct => matches!(it, Item::Wheat | Item::Fertilizer),
                        TradeKind::BuySeed => it.crop().is_some(),
                        TradeKind::BuyAnimal => it.animal().is_some(),
                    });
                    let Some(it) = valid else {
                        live[p] = None;
                        continue;
                    };
                    let price = match kind {
                        TradeKind::Sell => self.price(it),
                        TradeKind::BuyProduct => {
                            // Quoted at the post-buy inventory.
                            market_price(&self.market.params[it.idx()], self.market.inventory[it.idx()] - 1)
                        }
                        TradeKind::BuySeed => it.crop().unwrap().spec().seed,
                        TradeKind::BuyAnimal => it.animal().unwrap().spec().cost,
                    };
                    quoted[p] = Some((kind, it, price));
                }
                if quoted.iter().all(Option::is_none) {
                    break;
                }
                let mut committed_any = false;
                for p in 0..NUM_AGENTS {
                    let Some((kind, it, price)) = quoted[p] else { continue };
                    if self.commit_unit(p, kind, it, price) {
                        if let Some(l) = live[p].as_mut() {
                            l.2 -= 1;
                        }
                        committed_any = true;
                    } else {
                        live[p] = None;
                    }
                }
                if !committed_any {
                    break;
                }
            }
            self.refresh_prices();
        }
    }

    /// `_commit_unit`
    fn commit_unit(&mut self, p: usize, kind: TradeKind, item: Item, price: i64) -> bool {
        let cap = self.cfg.shed_capacity;
        let farm = &mut self.farms[p];
        let private = &mut self.privates[p];
        match kind {
            TradeKind::Sell => {
                if private.shed[item.idx()] <= 0 {
                    return false;
                }
                private.shed[item.idx()] -= 1;
                farm.money += price;
                // Sales at $1 do not add supply.
                if price > 1 {
                    self.market.inventory[item.idx()] += 1;
                }
                true
            }
            TradeKind::BuyProduct => {
                if farm.money < price || private.shed_total() >= cap {
                    return false;
                }
                farm.money -= price;
                private.shed[item.idx()] += 1;
                self.market.inventory[item.idx()] -= 1;
                true
            }
            TradeKind::BuySeed => {
                if farm.money < price {
                    return false;
                }
                farm.money -= price;
                private.seeds[item.idx()] += 1;
                true
            }
            TradeKind::BuyAnimal => {
                if farm.money < price || private.shed_total() >= cap {
                    return false;
                }
                farm.money -= price;
                private.shed[item.idx()] += 1;
                true
            }
        }
    }

    /// `_do_hire`: the n-th hire of the day costs mult * fib(n); the hand
    /// spawns on the least-occupied shed-access tile (NWSE tie-break).
    fn do_hire(&mut self, p: usize) {
        let bs = self.cfg.board_size;
        let farm = &mut self.farms[p];
        let cost = self.cfg.farm_hand_cost_mult.saturating_mul(fib(farm.hires_today));
        if farm.money < cost {
            return;
        }
        farm.money -= cost;
        farm.hires_today += 1;
        let access = shed_access_tiles(bs);
        let mut occ = [0usize; 4];
        for pos in std::iter::once(&farm.farmer).chain(farm.hands.iter()) {
            if let Some(k) = access.iter().position(|t| t == pos) {
                occ[k] += 1;
            }
        }
        let best = (0..4).min_by_key(|&k| (occ[k], k)).unwrap();
        farm.hands.push(access[best]);
        self.privates[p].inventories.push(Inventory::default());
    }

    /// `_do_buy_land`
    fn do_buy_land(&mut self, p: usize) {
        let bs = self.cfg.board_size;
        let farm = &mut self.farms[p];
        let n_extra = farm.unlocked_quadrants.len() - 1;
        if n_extra >= LAND_ORDER.len() {
            return;
        }
        let cost = LAND_PRICES[n_extra];
        if farm.money < cost {
            return;
        }
        farm.money -= cost;
        let q = LAND_ORDER[n_extra];
        farm.unlocked_quadrants.push(q);
        for y in 0..bs {
            for x in 0..bs {
                if quadrant_of(x, y, bs) == q && *farm.tile(x, y) == Tile::Locked {
                    *farm.tile_mut(x, y) = Tile::Empty;
                }
            }
        }
    }

    fn refresh_prices(&mut self) {
        for i in 0..N_PRODUCTS {
            self.market.prices[i] = market_price(&self.market.params[i], self.market.inventory[i]);
        }
    }

    /// `_town_consume`
    fn town_consume(&mut self, step: i64) {
        if step % self.cfg.town_shop_sell_interval == 0 {
            for &s in &self.town.unlocked_shops {
                let products = SHOPS[s].1;
                let mult = if products.len() == 1 { 2 } else { 1 };
                for it in products {
                    self.market.inventory[it.idx()] -= mult;
                }
            }
        }
        if step % self.cfg.town_center_sell_interval == 0 {
            for i in 0..Item::Fertilizer.idx() {
                self.market.inventory[i] -= 1;
            }
        }
        self.refresh_prices();
    }

    /// `_end_of_day`
    fn end_of_day(&mut self, day: i64) {
        let tpd = self.cfg.turns_per_day;
        let bs = self.cfg.board_size;
        let cap = self.cfg.shed_capacity;
        let mut rng = PyRandom::new((self.seed as i128 * 1_000_003) ^ day as i128);

        for p in 0..NUM_AGENTS {
            let farm = &mut self.farms[p];
            daily_refresh_plants(farm, day, tpd);
            daily_refresh_animals(farm, day);
            for t in farm.tiles.iter_mut() {
                if *t == Tile::Empty && rng.random() < self.cfg.weed_spawn_chance {
                    *t = Tile::Weed;
                }
            }
            let private = &mut self.privates[p];
            for inv in std::mem::take(&mut private.inventories) {
                for (item, n) in inv.items {
                    if n <= 0 {
                        continue;
                    }
                    let take = n.min((cap - private.shed_total()).max(0));
                    if take > 0 {
                        private.shed[item.idx()] += take;
                    }
                }
            }
            private.inventories = vec![Inventory::default()];
            farm.farmer = default_spawn(bs);
            farm.hands.clear();
            farm.hires_today = 0;
        }

        let next_day = day + 1;
        if next_day > 0
            && next_day % self.cfg.town_shop_unlock_interval == 0
            && self.town.unlocked_shops.len() < MAX_SHOP_INSTANCES
        {
            self.town.unlocked_shops.push(rng.randbelow(SHOPS.len() as u32) as usize);
        }
    }

    /// Full state in the engine's observation shape, plus `step` and `done`.
    pub fn snapshot(&self) -> Value {
        json!({
            "step": self.step,
            "day": self.day(),
            "hour": self.hour(),
            "farms": self.farms.iter().map(Farm::to_json).collect::<Vec<_>>(),
            "market": self.market.to_json(),
            "town": self.town.to_json(),
            "privates": self.privates.iter().map(Private::to_json).collect::<Vec<_>>(),
            "done": self.done,
        })
    }
}

fn new_plant(crop: Crop, day: i64, tpd: i64) -> Plant {
    let cd = crop.spec();
    Plant {
        crop,
        planted_day: day,
        watered_today: false,
        // The planting day counts as unwatered.
        consecutive_unwatered: 1,
        yield_units: if cd.ongoing { 0 } else { 1 },
        max_lifespan_step: if cd.ongoing { -1 } else { (day + cd.max_yield_day + 1) * tpd },
        fertilized_until_day: -1,
    }
}

fn new_animal(animal: Animal, day: i64) -> AnimalTile {
    AnimalTile {
        animal,
        placed_day: day,
        yield_units: 0,
        consecutive_unfed: 0,
        fed_today: false,
        cared_today: false,
        fertilizer_available: false,
        pending_care_bonus: 0,
    }
}

/// `_decay_plants`: past its lifespan a plant loses one unit every 2 turns and
/// turns into a weed at zero.
fn decay_plants(farm: &mut Farm, step: i64) {
    for t in farm.tiles.iter_mut() {
        let Tile::Plant(p) = t else { continue };
        let mls = p.max_lifespan_step;
        if mls < 0 || step < mls || (step - mls) % 2 != 0 {
            continue;
        }
        p.yield_units -= 1;
        if p.yield_units <= 0 {
            *t = Tile::Weed;
        }
    }
}

/// `_daily_refresh_plants`
fn daily_refresh_plants(farm: &mut Farm, current_day: i64, tpd: i64) {
    let next_day = current_day + 1;
    for t in farm.tiles.iter_mut() {
        let Tile::Plant(p) = t else { continue };
        let was_watered = p.watered_today;
        if was_watered {
            p.consecutive_unwatered = 0;
        } else {
            p.consecutive_unwatered += 1;
        }
        p.watered_today = false;
        if p.consecutive_unwatered >= 2 {
            *t = Tile::Weed;
            continue;
        }
        let cd = p.crop.spec();
        if !cd.ongoing {
            continue;
        }
        let days_since_first = next_day - p.planted_day - cd.first_yield_day;
        if days_since_first < 0 || days_since_first % cd.interval != 0 {
            continue;
        }
        let production_count = days_since_first / cd.interval + 1;
        if production_count > cd.max_yield {
            continue;
        }
        // Fertilizer only counts on watered days.
        let fertilized = was_watered && p.fertilized_until_day >= current_day;
        p.yield_units = cd.max_yield.min(p.yield_units + if fertilized { 2 } else { 1 });
        if production_count == cd.max_yield {
            p.max_lifespan_step = (next_day + 1) * tpd;
        }
    }
}

/// `_daily_refresh_animals`
fn daily_refresh_animals(farm: &mut Farm, day: i64) {
    let next_day = day + 1;
    for t in farm.tiles.iter_mut() {
        let Tile::Animal(a) = t else { continue };
        if a.fed_today {
            a.consecutive_unfed = 0;
        } else {
            a.consecutive_unfed += 1;
        }
        if a.consecutive_unfed >= 2 {
            // The animal escapes; the structure stays.
            *t = Tile::Structure(a.animal.spec().structure);
            continue;
        }
        let spec = a.animal.spec();
        let days_since_first = next_day - a.placed_day - spec.first_yield_day;
        if days_since_first >= 0 && days_since_first % spec.interval == 0 {
            // The care bonus is only paid out on a fed production day; either way it resets.
            let bonus = if a.fed_today { a.pending_care_bonus } else { 0 };
            a.yield_units = spec.max_held.min(a.yield_units + 1 + bonus);
            a.pending_care_bonus = 0;
        }
        if a.cared_today && a.fed_today {
            a.pending_care_bonus += 1;
        }
        a.fertilizer_available = true;
        a.fed_today = false;
        a.cared_today = false;
    }
}
