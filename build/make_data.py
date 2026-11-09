"""Pack the four image sets into js/data/*.js and the PCA test reference.

Each js/data/<key>.js registers one dataset: all images tiled into a
sprite (a data: URL, so canvas reads it even from file://) plus labels and
the held-out indices. The page computes PCA itself from these pixels; the
reference test/ref.js holds sklearn's answers on the same pixels.

    PCA_DATA.add({key, title, w, h, ch, n, cols, labels, holdout, credit,
                  sprite})

Sets (sources need no network except cats, see fetch_cats.py):
    guesswho: the 24 Guess Who cards (notes/day20_pca_eigen/guess_who),
        name strip cropped off, 60 x 80 RGB
    olivetti: sklearn's Olivetti faces (cached), 400 x 64 x 64 gray
    emoji: Noto Color Emoji smiley faces rendered from the system font,
        64 x 64 RGB on white
    cats: AFHQ cats (Choi et al. 2020), 64 x 64 RGB

Run (from the repo root):
    ~/venv_ds4400/bin/python -I build/make_data.py --cats <dir> \\
        --guesswho <dir>
"""
import argparse
import base64
import io
import json
import math
import pathlib
import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

ROOT = pathlib.Path(__file__).resolve().parent.parent
EMOJI_FONT = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"

# Yellow round faces (Unicode subgroups face-smiling .. face-negative);
# the clown, robot, cat and monkey faces are left out.
EMOJI_RANGES = [(0x1F600, 0x1F637), (0x1F641, 0x1F644), (0x1F910, 0x1F915),
                (0x1F917, 0x1F917), (0x1F920, 0x1F920), (0x1F922, 0x1F925),
                (0x1F927, 0x1F92F), (0x1F970, 0x1F976), (0x1F978, 0x1F97A),
                (0x1F9D0, 0x1F9D0), (0x1FAE0, 0x1FAE5), (0x1FAE8, 0x1FAE8),
                (0x2639, 0x263A)]


def every_kth(n: int, k: int, start: int) -> list:
    """Held-out indices start, start + k, ... below n."""
    return list(range(start, n, k))


def guesswho(src: pathlib.Path) -> dict:
    """Load the Guess Who cards.

    Returns:
        dict: {imgs, labels, holdout}, imgs (np.array) (24, 80, 60, 3) uint8
    """
    imgs, labels = [], []
    for f in sorted(src.glob("*.png")):
        im = Image.open(f).convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im).convert("RGB")
        # The card is 180 x 296 with the name printed below y = 240.
        im = im.crop((0, 0, 180, 240)).resize((60, 80), Image.LANCZOS)
        imgs.append(np.asarray(im))
        labels.append(f.stem.capitalize())
    return dict(imgs=np.stack(imgs), labels=labels,
                holdout=every_kth(len(imgs), 6, 3))


def olivetti() -> dict:
    """Load Olivetti faces: 40 people x 10 photos, one photo each held out.

    Returns:
        dict: {imgs, labels, holdout}, imgs (np.array) (400, 64, 64) uint8
    """
    d = fetch_olivetti_faces(download_if_missing=False)
    imgs = np.round(d.images * 255).astype(np.uint8)
    labels = [f"person {t + 1}, photo {i % 10 + 1}"
              for i, t in enumerate(d.target)]
    return dict(imgs=imgs, labels=labels, holdout=every_kth(400, 10, 9))


def emoji() -> dict:
    """Render the smiley-face emoji, all at one origin so circles align.

    Returns:
        dict: {imgs, labels, holdout}, imgs (np.array) (n, 64, 64, 3) uint8
    """
    font = ImageFont.truetype(EMOJI_FONT, 109)

    def render(ch: str) -> np.array:
        im = Image.new("RGBA", (136, 128), (255, 255, 255, 0))
        ImageDraw.Draw(im).text((0, 0), ch, font=font, embedded_color=True)
        return np.asarray(im)

    missing = render(chr(0x10FFFD))
    imgs, labels = [], []
    for lo, hi in EMOJI_RANGES:
        for cp in range(lo, hi + 1):
            a = render(chr(cp))
            if a[..., 3].max() == 0 or np.array_equal(a, missing):
                continue
            im = Image.fromarray(a)
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im).convert("RGB")
            im = im.crop((4, 0, 132, 128)).resize((64, 64), Image.LANCZOS)
            imgs.append(np.asarray(im))
            labels.append(unicodedata.name(chr(cp), f"U+{cp:X}").lower())
    return dict(imgs=np.stack(imgs), labels=labels,
                holdout=every_kth(len(imgs), 10, 5))


def cats(src: pathlib.Path) -> dict:
    """Load the cat faces saved by fetch_cats.py.

    Returns:
        dict: {imgs, labels, holdout}, imgs (np.array) (n, 64, 64, 3) uint8
    """
    files = sorted(src.glob("cat_*.jpg"))
    imgs = np.stack([np.asarray(Image.open(f).convert("RGB"))
                     for f in files])
    labels = [f"cat {i + 1}" for i in range(len(files))]
    return dict(imgs=imgs, labels=labels,
                holdout=every_kth(len(imgs), 10, 9))


def sprite(imgs, fmt: str) -> tuple:
    """Tile images into one sprite and encode it as a data: URL.

    Args:
        imgs (np.array): (n, h, w) or (n, h, w, 3) uint8
        fmt (str): "png" (lossless) or "jpeg"

    Returns:
        url (str): data: URL of the sprite
        cols (int): images per sprite row
        pix (np.array): (n, h, w[, 3]) uint8 as decoded back from the sprite
    """
    n, h, w = imgs.shape[:3]
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    sheet = np.full((rows * h, cols * w) + imgs.shape[3:], 255, np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = imgs[i]
    buf = io.BytesIO()
    im = Image.fromarray(sheet)
    if fmt == "jpeg":
        im.save(buf, "JPEG", quality=92, subsampling=0)
    else:
        im.save(buf, "PNG", optimize=True)
    raw = buf.getvalue()
    back = np.asarray(Image.open(io.BytesIO(raw)))
    pix = np.stack([back[(i // cols) * h:(i // cols + 1) * h,
                         (i % cols) * w:(i % cols + 1) * w]
                    for i in range(n)])
    url = f"data:image/{fmt};base64," + base64.b64encode(raw).decode()
    return url, cols, pix


def reference(pix, holdout: list) -> dict:
    """sklearn PCA on the sprite pixels (scaled to [0, 1]), for the test.

    Args:
        pix (np.array): (n, h, w[, 3]) uint8

    Returns:
        dict: {all, held}, each {var (first 10 lambda_j, sklearn's
            explained_variance_), frac10 (cumulative fraction at k = 10),
            rank, err5 (mean squared error of image 0 rebuilt with k = 5)}
    """
    x = pix.reshape(len(pix), -1).astype(np.float64) / 255
    out = {}
    for name, train in [("all", np.arange(len(x))),
                        ("held", np.setdiff1d(np.arange(len(x)), holdout))]:
        p = PCA(svd_solver="full").fit(x[train])
        lam = p.explained_variance_
        rank = int((lam > lam[0] * 1e-10).sum())
        p5 = PCA(n_components=5, svd_solver="full").fit(x[train])
        rec = p5.inverse_transform(p5.transform(x[:1]))
        out[name] = dict(var=lam[:10].tolist(),
                         frac10=float(lam[:10].sum() / lam.sum()),
                         rank=rank,
                         err5=float(((rec - x[:1]) ** 2).mean()))
    return out


CREDITS = {
    "guesswho": "Guess Who character cards, © Hasbro; scanned for "
                "teaching.",
    "olivetti": "Olivetti faces, AT&T Laboratories Cambridge (Samaria & "
                "Harter 1994), via scikit-learn.",
    "emoji": "Noto Color Emoji, Google (SIL Open Font License 1.1).",
    "cats": "AFHQ cats (Choi et al. 2020), CC BY-NC 4.0, from the 64-pixel "
            "copy zzsi/afhq64_16k.",
}
TITLES = {"guesswho": "Guess Who", "olivetti": "Olivetti faces",
          "emoji": "Emoji", "cats": "Cats"}


def main(cats_dir: pathlib.Path, gw_dir: pathlib.Path) -> None:
    """Write js/data/<key>.js for every set and test/ref.js."""
    sets = {"guesswho": (guesswho(gw_dir), "png"),
            "olivetti": (olivetti(), "png"),
            "emoji": (emoji(), "png"),
            "cats": (cats(cats_dir), "jpeg")}
    ref = {}
    for key, (d, fmt) in sets.items():
        imgs = d["imgs"]
        url, cols, pix = sprite(imgs, fmt)
        meta = dict(key=key, title=TITLES[key], w=imgs.shape[2],
                    h=imgs.shape[1], ch=1 if imgs.ndim == 3 else 3,
                    n=len(imgs), cols=cols, labels=d["labels"],
                    holdout=d["holdout"], credit=CREDITS[key], sprite=url)
        js = (f"// {TITLES[key]}: generated by build/make_data.py.\n"
              f"PCA_DATA.add({json.dumps(meta)});\n")
        (ROOT / "js" / "data" / f"{key}.js").write_text(js)
        ref[key] = reference(pix, d["holdout"])
        print(f"{key}: n={len(imgs)} {imgs.shape[1:]} "
              f"{len(js) / 1e6:.2f} MB, rank {ref[key]['all']['rank']}")
    (ROOT / "test" / "ref.js").write_text(
        "// sklearn PCA on the sprite pixels: generated by "
        "build/make_data.py.\nconst REF = " + json.dumps(ref, indent=1)
        + ";\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cats", type=pathlib.Path, required=True)
    ap.add_argument("--guesswho", type=pathlib.Path, required=True)
    a = ap.parse_args()
    main(a.cats, a.guesswho)
