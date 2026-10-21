use kaggsim::agents::by_name;
use kaggsim::{Action, Config, Game, NUM_AGENTS};
use serde_json::Value;
use std::io::{BufWriter, Write};
use std::process::exit;
use std::time::Instant;

const USAGE: &str = "\
kaggsim replay <file.json> [--final]
    Replays a Kaggle episode replay, or {\"seed\", \"configuration\", \"actions\": [[a0, a1], ...]}.
    Prints one JSON state per line (initial state, then after every turn), or only the last with --final.

kaggsim play [--p0 NAME] [--p1 NAME] [--games N] [--seed S]
    Self-play with built-in agents (pass, starter, random). Prints each game's scores and throughput.";

fn die(msg: &str) -> ! {
    eprintln!("{msg}\n\n{USAGE}");
    exit(2)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("replay") => replay(&args[1..]),
        Some("play") => play(&args[1..]),
        _ => die("missing command"),
    }
}

/// Seed, configuration and per-turn actions from either replay format.
fn load_episode(v: &Value) -> (i64, Config, Vec<Vec<Action>>) {
    let cfg_json = v.get("configuration").cloned().unwrap_or(Value::Null);
    let cfg = Config::from_json(&cfg_json);
    let seed = v
        .pointer("/info/seed")
        .or_else(|| v.get("seed"))
        .or_else(|| cfg_json.get("seed"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let actions = if let Some(steps) = v.get("steps").and_then(Value::as_array) {
        // Kaggle replay: steps[t][i].action is the action that produced state t.
        steps
            .iter()
            .skip(1)
            .map(|st| (0..NUM_AGENTS).map(|i| Action::parse(st.pointer(&format!("/{i}/action")).unwrap_or(&Value::Null))).collect())
            .collect()
    } else if let Some(acts) = v.get("actions").and_then(Value::as_array) {
        acts.iter()
            .map(|pair| (0..NUM_AGENTS).map(|i| Action::parse(pair.get(i).unwrap_or(&Value::Null))).collect())
            .collect()
    } else {
        die("replay file has neither \"steps\" nor \"actions\"")
    };
    (seed, cfg, actions)
}

fn replay(args: &[String]) {
    let path = args.iter().find(|a| !a.starts_with("--")).unwrap_or_else(|| die("replay needs a file"));
    let only_final = args.iter().any(|a| a == "--final");
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| die(&format!("{path}: {e}")));
    let v: Value = serde_json::from_str(&text).unwrap_or_else(|e| die(&format!("{path}: {e}")));
    let (seed, cfg, actions) = load_episode(&v);

    let mut game = Game::new(cfg, seed);
    let out = std::io::stdout();
    let mut out = BufWriter::new(out.lock());
    if !only_final {
        writeln!(out, "{}", game.snapshot()).unwrap();
    }
    for turn in &actions {
        if game.done {
            break;
        }
        game.step(turn);
        if !only_final {
            writeln!(out, "{}", game.snapshot()).unwrap();
        }
    }
    if only_final {
        writeln!(out, "{}", game.snapshot()).unwrap();
    }
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(String::as_str)
}

fn play(args: &[String]) {
    let p0 = flag(args, "--p0").unwrap_or("starter");
    let p1 = flag(args, "--p1").unwrap_or("random");
    let games: u64 = flag(args, "--games").map_or(1, |s| s.parse().unwrap_or_else(|_| die("bad --games")));
    let seed: i64 = flag(args, "--seed").map_or(0, |s| s.parse().unwrap_or_else(|_| die("bad --seed")));

    let start = Instant::now();
    let mut turns = 0u64;
    let mut totals = [0i64; NUM_AGENTS];
    for g in 0..games {
        let s = seed + g as i64;
        let mut agents = [p0, p1].map(|n| by_name(n, s as u64 * 2 + 1).unwrap_or_else(|| die(&format!("unknown agent {n}"))));
        let mut game = Game::new(Config::default(), s);
        while !game.done {
            let acts: Vec<Action> = agents.iter_mut().enumerate().map(|(i, a)| a.act(&game, i)).collect();
            game.step(&acts);
            turns += 1;
        }
        let scores = game.scores();
        totals[0] += scores[0];
        totals[1] += scores[1];
        println!("game {g} seed {s}: {p0} {} vs {p1} {}", scores[0], scores[1]);
    }
    let secs = start.elapsed().as_secs_f64();
    println!(
        "{games} games, mean {:.1} vs {:.1}, {:.0} turns/s, {:.1} games/s",
        totals[0] as f64 / games as f64,
        totals[1] as f64 / games as f64,
        turns as f64 / secs,
        games as f64 / secs
    );
}
