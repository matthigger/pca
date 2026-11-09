"""Download AFHQ cat faces (64 x 64) for the Cats dataset.

Source: the 64-pixel AFHQ copy zzsi/afhq64_16k on Hugging Face, val split,
where label 1 is cat (rows 0..499). AFHQ is Choi et al. 2020 (StarGAN v2),
CC BY-NC 4.0. Rows come from the datasets-server API, so no parquet reader
is needed.

Run:
    python -I build/fetch_cats.py --out <empty dir> [--n 300]
"""
import argparse
import json
import pathlib
import urllib.request

API = ("https://datasets-server.huggingface.co/rows?dataset=zzsi/afhq64_16k"
       "&config=default&split=val&offset={off}&length={length}")


def fetch(out: pathlib.Path, n: int) -> None:
    """Save the first n cat rows as out/cat_<row>.jpg."""
    out.mkdir(parents=True, exist_ok=True)
    for off in range(0, n, 100):
        url = API.format(off=off, length=min(100, n - off))
        with urllib.request.urlopen(url) as f:
            rows = json.load(f)["rows"]
        for r in rows:
            if r["row"]["label"] != 1:
                continue
            dst = out / f"cat_{r['row_idx']:03d}.jpg"
            urllib.request.urlretrieve(r["row"]["image"]["src"], dst)
        print(f"rows {off}..{off + len(rows) - 1}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    fetch(a.out, a.n)
