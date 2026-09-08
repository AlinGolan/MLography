"""
Split a (image, labeled-image) dataset into train / test subsets and copy the
files into a new directory tree, leaving the originals untouched.

Layout produced:
    <out_dir>/train/<image_subdir>/...    <out_dir>/train/<label_subdir>/...
    <out_dir>/test/<image_subdir>/...     <out_dir>/test/<label_subdir>/...
    <out_dir>/split_manifest.json

Images and labels are paired by file stem (name without extension); the two
folders may use different extensions.

Two split modes:
  * per file            (default)                -> exact test fraction on files
  * grouped by a regex  (--group_regex '^(\\d+)-') -> every file whose stem
    matches the same capture group goes entirely to train or entirely to test
    (use this for tiles cut from the same source image, to avoid leakage).

Examples
--------
# grains boundary: 128px tiles, keep all tiles of one source image together
python split_dataset.py \
    --image_dir data/squares_128/train/image \
    --label_dir data/squares_128/train/inv_label \
    --out_dir   data/squares_128_split \
    --image_subdir image --label_subdir inv_label \
    --test_frac 0.1 --seed 42 --group_regex '^([0-9]+)-'

# impurities: independent images, plain random split
python split_dataset.py \
    --image_dir data/small/train/image_preprocess_cons \
    --label_dir data/small/train/label_fixed_cons \
    --out_dir   data/small_split \
    --image_subdir image_preprocess_cons --label_subdir label_fixed_cons \
    --test_frac 0.1 --seed 42
"""
import argparse
import json
import os
import random
import re
import shutil
import sys
from collections import OrderedDict


def _list_by_stem(directory):
    """{stem: filename} for every regular file in *directory* (errors on dup stems)."""
    out = {}
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        stem = os.path.splitext(name)[0]
        if stem in out:
            raise ValueError(
                "duplicate stem %r in %s (%s vs %s)" % (stem, directory, out[stem], name))
        out[stem] = name
    return out


def _group_key(stem, pattern):
    if pattern is None:
        return stem
    m = pattern.search(stem)
    if not m:
        raise ValueError("group_regex did not match stem %r" % stem)
    return m.group(1) if m.groups() else m.group(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--label_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--image_subdir", default="image",
                    help="name of the image folder inside train/ and test/")
    ap.add_argument("--label_subdir", default="label",
                    help="name of the label folder inside train/ and test/")
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group_regex", default=None,
                    help="if given, files sharing this capture group stay on the "
                         "same side of the split (no leakage between tiles)")
    ap.add_argument("--allow_unpaired", action="store_true",
                    help="skip images without a matching label instead of erroring")
    ap.add_argument("--overwrite", action="store_true",
                    help="wipe existing <out_dir>/train and <out_dir>/test first")
    ap.add_argument("--dry_run", action="store_true",
                    help="report the split but copy nothing")
    args = ap.parse_args()

    if not 0.0 < args.test_frac < 1.0:
        ap.error("--test_frac must be in (0, 1)")

    images = _list_by_stem(args.image_dir)
    labels = _list_by_stem(args.label_dir)

    only_img = sorted(set(images) - set(labels))
    only_lbl = sorted(set(labels) - set(images))
    if only_img and not args.allow_unpaired:
        raise SystemExit("%d image(s) have no label, e.g. %s (use --allow_unpaired)"
                         % (len(only_img), only_img[:5]))
    if only_lbl:
        print("note: %d label(s) have no image, ignored (e.g. %s)"
              % (len(only_lbl), only_lbl[:5]))

    stems = sorted(set(images) & set(labels))
    if not stems:
        raise SystemExit("no (image, label) pairs found")

    pattern = re.compile(args.group_regex) if args.group_regex else None

    # group -> [stems]
    groups = OrderedDict()
    for stem in stems:
        groups.setdefault(_group_key(stem, pattern), []).append(stem)

    rng = random.Random(args.seed)
    group_names = list(groups)
    rng.shuffle(group_names)

    n_total = len(stems)
    n_test_target = max(1, int(round(args.test_frac * n_total)))

    test_stems, train_stems = [], []
    if pattern is None:
        # per-file: exact count
        shuffled = list(stems)
        rng.shuffle(shuffled)
        test_stems = shuffled[:n_test_target]
        train_stems = shuffled[n_test_target:]
    else:
        # grouped: add whole groups until we reach ~n_test_target files
        acc = 0
        test_groups = set()
        for g in group_names:
            if acc >= n_test_target:
                break
            test_groups.add(g)
            acc += len(groups[g])
        for g, gs in groups.items():
            (test_stems if g in test_groups else train_stems).extend(gs)

    test_stems = sorted(test_stems)
    train_stems = sorted(train_stems)

    print("pairs total      : %d  (%d groups)" % (n_total, len(groups)))
    print("test  target/frac: %d / %.3f" % (n_test_target, args.test_frac))
    print("train pairs      : %d  (%.3f)" % (len(train_stems), len(train_stems) / n_total))
    print("test  pairs      : %d  (%.3f)" % (len(test_stems), len(test_stems) / n_total))
    if pattern is not None:
        tg = sorted({_group_key(s, pattern) for s in test_stems})
        print("test  groups     : %s" % ", ".join(tg))

    manifest = OrderedDict(
        image_dir=os.path.abspath(args.image_dir),
        label_dir=os.path.abspath(args.label_dir),
        out_dir=os.path.abspath(args.out_dir),
        image_subdir=args.image_subdir,
        label_subdir=args.label_subdir,
        test_frac=args.test_frac,
        seed=args.seed,
        group_regex=args.group_regex,
        n_total=n_total,
        n_train=len(train_stems),
        n_test=len(test_stems),
        train_stems=train_stems,
        test_stems=test_stems,
    )

    if args.dry_run:
        print("\n[dry run] nothing copied")
        print(json.dumps({k: manifest[k] for k in list(manifest)[:12]}, indent=2))
        return

    for split in ("train", "test"):
        for sub in (args.image_subdir, args.label_subdir):
            d = os.path.join(args.out_dir, split, sub)
            if args.overwrite and os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
            if os.listdir(d):
                raise SystemExit("%s is not empty (use --overwrite)" % d)

    def copy_split(split, split_stems):
        for stem in split_stems:
            shutil.copy2(os.path.join(args.image_dir, images[stem]),
                         os.path.join(args.out_dir, split, args.image_subdir, images[stem]))
            shutil.copy2(os.path.join(args.label_dir, labels[stem]),
                         os.path.join(args.out_dir, split, args.label_subdir, labels[stem]))

    copy_split("train", train_stems)
    copy_split("test", test_stems)

    with open(os.path.join(args.out_dir, "split_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print("\ncopied into %s" % os.path.abspath(args.out_dir))
    print("manifest: %s" % os.path.join(os.path.abspath(args.out_dir), "split_manifest.json"))


if __name__ == "__main__":
    sys.exit(main())
