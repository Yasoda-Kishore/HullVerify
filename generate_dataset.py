"""
Synthetic hull-cleaning verification dataset generator.
Creates before/after image pairs of a painted hull-like surface with
simulated biofouling, for the four-outcome decision demo:
  sufficient, more_cleaning, unclear_image, expert_inspection

HONESTY NOTE: These are computer-generated images, NOT real hull photos.
Say so in any presentation. They demonstrate the pipeline, not real-world accuracy.
"""
import csv
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).parent / "dataset"
OUT.mkdir(exist_ok=True)

SIZE = 512
rng = random.Random(42)
nprng = np.random.default_rng(42)

ZONES_NORMAL = ["waterline", "vertical_side", "flat_bottom", "bow", "stern", "bilge_keel"]
ZONES_SENSITIVE = ["sea_chest", "propeller", "anodes"]


def make_background(seed):
    """Painted hull plate: antifouling red-brown or grey, with noise and a weld line."""
    r = np.random.default_rng(seed)
    base_choices = [(118, 48, 42), (96, 100, 106), (70, 78, 88), (128, 60, 50)]
    base = np.array(base_choices[seed % len(base_choices)], dtype=np.float32)
    img = np.ones((SIZE, SIZE, 3), dtype=np.float32) * base
    # large-scale shading
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32) / SIZE
    shade = 18 * np.sin(2 * math.pi * (xx * r.uniform(0.5, 1.5) + r.uniform(0, 1))) \
          + 12 * np.sin(2 * math.pi * (yy * r.uniform(0.5, 1.5) + r.uniform(0, 1)))
    img += shade[..., None]
    # fine paint noise
    img += r.normal(0, 6, (SIZE, SIZE, 1))
    # weld seam line
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)
    y = r.integers(100, SIZE - 100)
    d.line([(0, int(y)), (SIZE, int(y) + int(r.integers(-30, 30)))],
           fill=tuple(int(c * 0.8) for c in base.astype(int)), width=4)
    return np.asarray(pil).astype(np.float32)


def blob_mask(seed, coverage):
    """Irregular fouling mask (0..1) with roughly `coverage` fraction above 0.5."""
    r = np.random.default_rng(seed)
    acc = np.zeros((SIZE, SIZE), dtype=np.float32)
    n_blobs = int(r.integers(6, 14))
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    for _ in range(n_blobs):
        cx, cy = r.uniform(0, SIZE), r.uniform(0, SIZE)
        sx, sy = r.uniform(25, 90), r.uniform(25, 90)
        amp = r.uniform(0.6, 1.0)
        acc += amp * np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
    # ragged edges
    noise = r.normal(0, 1, (SIZE // 8, SIZE // 8)).astype(np.float32)
    noise = np.asarray(Image.fromarray(noise).resize((SIZE, SIZE), Image.BILINEAR))
    acc += 0.35 * noise
    # threshold so that fraction ~= coverage
    thresh = np.quantile(acc, 1 - coverage)
    mask = np.clip((acc - thresh) / 0.25, 0, 1)
    return mask


def apply_fouling(bg, mask, seed):
    """Greenish-brown slime + darker barnacle speckle under the mask."""
    r = np.random.default_rng(seed)
    slime = np.array([r.uniform(60, 90), r.uniform(85, 115), r.uniform(45, 70)], dtype=np.float32)
    out = bg.copy()
    m = mask[..., None]
    out = out * (1 - 0.85 * m) + slime * (0.85 * m)
    # barnacle speckle
    spk = (r.random((SIZE, SIZE)) < 0.04 * mask).astype(np.float32)
    spk = np.asarray(Image.fromarray((spk * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3))) / 255.0
    out = out * (1 - 0.6 * spk[..., None]) + np.array([200, 198, 190], np.float32) * (0.6 * spk[..., None])
    out += r.normal(0, 3, (SIZE, SIZE, 1))
    return np.clip(out, 0, 255)


def wipe_mask(seed, keep_fraction):
    """Spatial mask of fouling KEPT after cleaning (1=kept). Simulates wipe stripes."""
    r = np.random.default_rng(seed)
    keep = np.ones((SIZE, SIZE), dtype=np.float32)
    # remove horizontal/vertical stripes until roughly keep_fraction remains
    while keep.mean() > keep_fraction:
        if r.random() < 0.5:
            a = int(r.integers(0, SIZE - 80))
            keep[a:a + int(r.integers(50, 120)), :] *= r.uniform(0.0, 0.15)
        else:
            a = int(r.integers(0, SIZE - 80))
            keep[:, a:a + int(r.integers(50, 120))] *= r.uniform(0.0, 0.15)
    keep = np.asarray(Image.fromarray((keep * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(9))) / 255.0
    return keep


def add_rust_scratch(img, seed):
    r = np.random.default_rng(seed)
    pil = Image.fromarray(img.astype(np.uint8))
    d = ImageDraw.Draw(pil)
    for _ in range(int(r.integers(2, 5))):
        x0, y0 = int(r.integers(30, SIZE - 30)), int(r.integers(30, SIZE - 30))
        pts = [(x0, y0)]
        for _ in range(int(r.integers(3, 7))):
            pts.append((pts[-1][0] + int(r.integers(-70, 70)), pts[-1][1] + int(r.integers(-70, 70))))
        col = (int(r.integers(120, 170)), int(r.integers(55, 80)), int(r.integers(25, 45)))
        d.line(pts, fill=col, width=int(r.integers(4, 9)))
    out = np.asarray(pil).astype(np.float32)
    return out


def degrade_unclear(img, seed):
    r = np.random.default_rng(seed)
    pil = Image.fromarray(img.astype(np.uint8))
    mode = int(r.integers(0, 3))
    if mode == 0:   # heavy blur
        pil = pil.filter(ImageFilter.GaussianBlur(r.uniform(6, 10)))
    elif mode == 1:  # dark
        arr = np.asarray(pil).astype(np.float32) * r.uniform(0.12, 0.25)
        pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    else:            # blur + murky green haze
        pil = pil.filter(ImageFilter.GaussianBlur(r.uniform(4, 7)))
        arr = np.asarray(pil).astype(np.float32)
        haze = np.array([40, 90, 80], np.float32)
        arr = arr * 0.45 + haze * 0.55
        pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return np.asarray(pil).astype(np.float32)


def save(img, path):
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path, quality=90)


def main():
    rows = []
    plan = (["sufficient"] * 14) + (["more_cleaning"] * 14) + (["unclear_image"] * 10) + (["expert_inspection"] * 10)
    rng.shuffle(plan)

    for i, label in enumerate(plan, start=1):
        pid = f"pair{i:03d}"
        seed = 1000 + i
        bg = make_background(seed)

        coverage = rng.uniform(0.30, 0.50)
        fmask = blob_mask(seed * 7, coverage)
        before = apply_fouling(bg, fmask, seed * 11)

        if label == "expert_inspection":
            zone = rng.choice(ZONES_SENSITIVE)
        else:
            zone = rng.choice(ZONES_NORMAL)

        if label == "sufficient":
            keep = rng.uniform(0.00, 0.08)
            after = apply_fouling(bg, fmask * wipe_mask(seed * 13, keep), seed * 11)
        elif label == "more_cleaning":
            keep = rng.uniform(0.45, 0.65)
            after = apply_fouling(bg, fmask * wipe_mask(seed * 13, keep), seed * 11)
        elif label == "unclear_image":
            keep = rng.uniform(0.1, 0.6)
            after = apply_fouling(bg, fmask * wipe_mask(seed * 13, keep), seed * 11)
            after = degrade_unclear(after, seed * 17)
        else:  # expert_inspection: almost no improvement + rust scratches
            keep = rng.uniform(0.85, 0.95)
            after = apply_fouling(bg, fmask * wipe_mask(seed * 13, keep), seed * 11)
            after = add_rust_scratch(after, seed * 19)
            before = add_rust_scratch(before, seed * 19)  # damage present before too

        bf, af = f"{pid}_before.jpg", f"{pid}_after.jpg"
        save(before, OUT / bf)
        save(after, OUT / af)
        rows.append({"pair_id": pid, "zone": zone, "before_file": bf, "after_file": af, "label": label})

    # deterministic tune/test split, stratified by label
    counts = {}
    for r in rows:
        k = r["label"]
        r["split"] = "tune" if counts.get(k, 0) % 2 == 0 else "test"
        counts[k] = counts.get(k, 0) + 1

    with open(OUT / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "zone", "before_file", "after_file", "label", "split"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} pairs to {OUT}")
    for k in ["sufficient", "more_cleaning", "unclear_image", "expert_inspection"]:
        print(f"  {k}: {sum(1 for r in rows if r['label'] == k)}")


if __name__ == "__main__":
    main()
