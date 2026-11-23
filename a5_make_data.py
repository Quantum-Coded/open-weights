"""Amendment 5 data (operator, 2026-11-21, built before any model runs on it).

  python a5_make_data.py pilot   # a5_data/cube_pilot12_v2.jsonl: base-only pilot, 2 scrambles at each depth 1-6
  python a5_make_data.py test    # sealed/cube_100.jsonl: 20 scrambles at each depth 2-6

Scrambles come from cube.scramble (no two turns of one face in a row, no three turns on one axis in a row).
Pilot: random.Random(20261121). Test: random.Random(20261122); a test scramble equal to a pilot scramble or to
an earlier test scramble is drawn again.
"""
import hashlib, random, sys
from pathlib import Path
sys.path.insert(0, "/home/dev_gaglani/wsp-r1")
from common import SEALED_DIR, write_jsonl
from cube import fmt, make_row, scramble

A5 = Path("/home/dev_gaglani/wsp-r1/a5_data")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pilot_rows():
    rng = random.Random(20261121)
    return [make_row(i, d, scramble(rng, d)) for d in range(1, 7) for i in range(2)]


def main():
    mode = sys.argv[1]
    pilot = pilot_rows()
    if mode == "pilot":
        A5.mkdir(exist_ok=True)
        out = A5 / "cube_pilot12_v2.jsonl"  # v1 = cube_pilot12 (old prompt, kept)
        write_jsonl(out, pilot)
        print("pilot", len(pilot), out, sha(out))
        return
    assert mode == "test"
    rng = random.Random(20261122)
    seen = {r["scramble"] for r in pilot}
    rows = []
    # depths 2-6: depth 1 has only 18 distinct scrambles (2 already in the pilot), too few for 20 new ones
    for d in range(2, 7):
        i = 0
        while i < 20:
            m = scramble(rng, d)
            if fmt(m) in seen:
                continue
            seen.add(fmt(m))
            rows.append(make_row(i, d, m))
            i += 1
    out = SEALED_DIR / "cube_100.jsonl"
    write_jsonl(out, rows)
    print(f"cube test: {len(rows)} puzzles -> {out}  sha256 {sha(out)}")


if __name__ == "__main__":
    main()
