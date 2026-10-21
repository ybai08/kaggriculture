# kaggsim

A Rust port of the Kaggriculture game engine
(`kaggle_environments/envs/kaggriculture`, kaggle-environments 1.32.7), written
after the competition closed. It plays the same game as the Python engine:
same rules, same order of operations within a turn, the same floating-point
price arithmetic with banker's rounding, and a bit-exact copy of CPython's
Mersenne Twister for weeds and shop unlocks. Given the same seed and actions it
reaches the same state, turn for turn.

It runs about 2.5 million turns per second on one core (roughly 3,500 full
720-turn games per second with simple agents), against a few thousand turns
per second for the Python engine.

## Build

```bash
cargo build --release
```

## Use

Replay a Kaggle episode (from `kaggle competitions replay <episode_id>`) or an
action log, printing the state after every turn as one JSON line:

```bash
target/release/kaggsim replay episode-124374569-replay.json --final
```

An action log is `{"seed": 1, "configuration": {...}, "actions": [[p0, p1], ...]}`,
where each action has the Kaggle shape `{"farmer": [...], "hands": [...], "market": [...]}`.

Self-play with the built-in agents (`pass`, `starter`, `random`):

```bash
target/release/kaggsim play --p0 starter --p1 random --games 1000
```

As a library, `Game::new(Config::default(), seed)` and `game.step(&[a0, a1])`
are the whole interface; `game.snapshot()` gives the state in the engine's
observation shape.

## How it is checked

`parity/check_replay.py` replays real episodes from the competition and
compares every turn's farms, market, town and private sheds, seeds and carried
items with what Kaggle recorded, plus the final scores:

```bash
python3 parity/check_replay.py target/release/kaggsim episode-*.json
```

Three public episodes from the final day (124374569, 124390769, 124406994)
match on all 720 states. They cover hands, feeding and care bonuses,
fertilizer, ongoing crops, decay and all eight shop unlocks.

`parity/check_engine.py` runs the official Python engine directly, with fuzz
agents that send random and malformed actions, on the default configuration
and a small crowded one (shed overflow, every land quadrant, escapes, price
overrides). It can also play a submission file against the fuzzer:

```bash
pip download --no-deps kaggle-environments==1.32.7
unzip kaggle_environments-1.32.7-py3-none-any.whl 'kaggle_environments/envs/kaggriculture/*'
python3 parity/check_engine.py target/release/kaggsim \
    kaggle_environments/envs/kaggriculture/kaggriculture.py --games 20 --agent ../main.py
```

Kaggle replays store dicts with sorted keys, so only this second check sees
the order of items a farmer carries (which decides what is lost when the shed
overflows).

`cargo test` covers the RNG, price curves and a few rules against values taken
from the Python engine.

## Known differences

- Inputs that make the Python engine raise, such as a non-numeric count in
  `["PICKUP", "WHEAT", "abc"]`, are no-ops here.
- `log` and `log10` price curves call the platform's libm, as CPython does, so
  results match on the same platform.
- Only the two-player game is supported, as in the competition.
