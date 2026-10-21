"""Replay Kaggle episodes through kaggsim and compare every turn.

Usage:
    python3 parity/check_replay.py target/release/kaggsim episode-*.json

Download replays with:
    kaggle competitions episodes <submission_id>
    kaggle competitions replay <episode_id>

For each turn this compares both farms (tiles, money, positions, land, hires),
the market inventory and prices, the town's shops, and each player's private
shed, seeds and carried inventories against the observations recorded in the
replay. Exits non-zero on the first mismatch.

Kaggle writes replays with sorted keys, so the order of items in a carried
inventory is not recorded; inventories are compared as sorted lists here.
`check_engine.py` checks that order against the Python engine.
"""

import json
import subprocess
import sys


def expected_state(step):
    obs0 = step[0]["observation"]
    return {
        "day": obs0["day"],
        "hour": obs0["hour"],
        "farms": obs0["farms"],
        "market": {"inventory": obs0["market"]["inventory"], "prices": obs0["market"]["prices"]},
        "town": obs0["town"],
        "privates": [
            {
                "shed": s["observation"]["private"]["shed"],
                "seeds": s["observation"]["private"]["seeds"],
                "inventories": [sorted(map(list, inv.items())) for inv in s["observation"]["private"]["inventories"]],
            }
            for s in step
        ],
    }


def first_diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}/{k}: only in {'sim' if k in a else 'replay'}"
            d = first_diff(a[k], b[k], f"{path}/{k}")
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} (sim) vs {len(b)} (replay)"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f"{path}/{i}")
            if d:
                return d
        return None
    if a != b or type(a) is bool and type(b) is not bool:
        return f"{path}: {a!r} (sim) vs {b!r} (replay)"
    return None


def check(sim, path):
    with open(path) as f:
        replay = json.load(f)
    steps = replay["steps"]
    out = subprocess.run([sim, "replay", path], check=True, capture_output=True, text=True).stdout.splitlines()
    if len(out) != len(steps):
        return f"{len(out)} states from sim vs {len(steps)} in replay"
    for t, (line, step) in enumerate(zip(out, steps)):
        got = json.loads(line)
        got = {k: got[k] for k in ("day", "hour", "farms", "market", "town", "privates")}
        for priv in got["privates"]:
            priv["inventories"] = [sorted(inv) for inv in priv["inventories"]]
        d = first_diff(got, expected_state(step))
        if d:
            return f"turn {t}: {d}"
    final = json.loads(out[-1])
    scores = [f["money"] for f in final["farms"]]
    if scores != replay["rewards"]:
        return f"final scores {scores} vs rewards {replay['rewards']}"
    return None


def main():
    sim, paths = sys.argv[1], sys.argv[2:]
    failed = 0
    for p in paths:
        err = check(sim, p)
        with open(p) as f:
            rewards = json.load(f)["rewards"]
        print(f"{'FAIL' if err else 'ok  '} {p}  rewards={rewards}" + (f"\n     {err}" if err else ""))
        failed += bool(err)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
