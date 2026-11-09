# pca

Interactive teaching demo of principal component analysis for DS 4400
lesson 19, a companion to the other "visually" demos (e.g.
[Projection](https://github.com/matthigger/projection_2d_3d),
[SVMs](https://github.com/matthigger/svm)). Notation follows the lesson:
S = cov(x), unit direction **w**, score c = wᵀ(x − x̄), rebuild
x̂ = x̄ + c**w**, eigenvalues λ₁ ≥ λ₂ ≥ … with eigenvectors **v**ⱼ (the
principal directions **w**ⱼ in the eigenfaces tab). Two tabs:

- **Projection onto a line.** 60 samples whose covariance (with 1/n) is
  exactly the S on screen. Edit S as a matrix (rejected with a reason if
  it is not a covariance) or as λ₁, λ₂ and the angle of **v**₁; the cloud
  reshapes either way. Drag **w** (or its line) and see the rebuilds, the
  errors, the histogram of scores, and the kept / lost variance, which
  always sum to tr S. The variance-by-direction curve peaks at λ₁ on
  **v**₁. Optional overlays: the axes √λⱼ**v**ⱼ with their 2-sd ellipse
  (drag their tips to edit S) and the vector S**w**, parallel to **w**
  only on an eigenvector.
- **Eigenfaces.** Four image sets: Guess Who (24), Olivetti faces (400),
  Noto emoji smileys (101), AFHQ cats (300). Rebuild a picked image from
  its top k components; the eigenvalue bars and their running sum (the
  kept fraction) sit on one k axis. The mean and top 8 eigenfaces, a mixer
  that walks along **w**₁…**w**₄ in units of √λⱼ, a hold-out toggle (PCA
  fit without ~10% of the images, so they never rebuild perfectly), and a
  "Guess who?" game that reveals a hidden image one component at a time.

PCA runs in the browser from the pixels (`js/pca.js`, via the n × n Gram
matrix, Turk & Pentland 1991). Its eigenvalues match sklearn's
`explained_variance_` to ~1e-7 on every set (see Testing).

Plain HTML/CSS/JS with SVG and canvas: no build step for the page and no
dependencies, so it also works offline from `file://`.

## Data

`js/data/<set>.js` holds each set as one sprite (a `data:` URL, so canvas
can read it from `file://`). Rebuild them with

```sh
~/venv_ds4400/bin/python -I build/fetch_cats.py --out <empty dir>
~/venv_ds4400/bin/python -I build/make_data.py --cats <that dir> \
    --guesswho ~/Dropbox/teach/DS4400/notes/day20_pca_eigen/guess_who
```

which also rewrites `test/ref.js` (sklearn's answers on the same pixels).
Olivetti comes from sklearn's cache, emoji from the system Noto Color
Emoji font.

Credits and licenses: Guess Who cards © Hasbro, scanned for teaching;
Olivetti faces, AT&T Laboratories Cambridge (Samaria & Harter 1994);
Noto Color Emoji, Google (SIL OFL 1.1); AFHQ (Choi et al. 2020,
CC BY-NC 4.0) via the 64-pixel copy `zzsi/afhq64_16k`.

## Testing

```sh
~/venv_ds4400/bin/python test/run_headless.py "file://$PWD/test/index.html" \
    --wait '/PASS|FAIL/.test(document.title)' \
    --eval 'document.getElementById("out").textContent'
```

compares the in-browser PCA with `test/ref.js` (variances, kept fraction,
rank, a k = 5 rebuild, orthonormality) for every set, with and without the
held-out images. `run_headless.py` also takes `--shot` screenshots; it
drives snap Chromium, so run it outside the sandbox.

## Run locally

Open `index.html` in a browser, or serve it:

```sh
python3 -m http.server 8000   # then visit http://localhost:8000
```

Link straight to a tab with `index.html#projection` or
`index.html#eigenfaces`.

## Publish on GitHub Pages

```sh
gh repo create matthigger/pca --public --source . --push
gh api -X POST repos/matthigger/pca/pages -f build_type=workflow
```

`.github/workflows/pages.yml` then deploys each push to main. The site
lives at <https://matthigger.github.io/pca/>.
