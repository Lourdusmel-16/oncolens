"""Run from project root:  python -m tests.debug_sam
Prints a prompt-by-image table: how many regions SAM 3.1 returned for each prompt on EVERY
image in data/test_images. Results are cached on disk, so re-running costs no credits.
'tooth' is a control: if teeth are visible and it returns 0, something is wrong with the call.
Also saves the table to outputs/prompt_table.csv for your evaluation table."""
import csv
import glob
import os

from core.segment import LesionSegmenter, RECORD_RE
from core.utils import load_resized

CONCEPTS = ["tooth", "white patch", "red patch", "ulcer", "oral lesion", "lesion", "dark patch", "mouth sore"]

# Windows globbing is case-insensitive, but a set removes any duplicate paths
paths = sorted(set(
    glob.glob("data/test_images/*.jp*g")
    + glob.glob("data/test_images/*.JP*G")
    + glob.glob("data/test_images/*.png")
))
if not paths:
    raise SystemExit("No images found in data/test_images/. Copy your photos there first.")

seg = LesionSegmenter()
print(f"Found {len(paths)} images.\n")
print(f"{'image':28s}" + "".join(f"{c[:11]:>12s}" for c in CONCEPTS))

rows = []
for p in paths:
    name = os.path.basename(p)
    try:
        _, jpeg = load_resized(p)
    except Exception as e:
        print(f"{name[:27]:28s} SKIPPED ({e})")
        continue
    counts = []
    for c in CONCEPTS:
        try:
            text = seg._call_sam(jpeg, c, use_cache=True)
            counts.append(len(RECORD_RE.findall(text)))
        except Exception as e:
            print(f"   error on {name} / {c}: {e}")
            counts.append(-1)
    print(f"{name[:27]:28s}" + "".join(f"{n:>12d}" for n in counts))
    rows.append([name] + counts)

os.makedirs("outputs", exist_ok=True)
with open("outputs/prompt_table.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["image"] + CONCEPTS)
    w.writerows(rows)

print("\n0 = SAM found nothing for that prompt.  -1 = the call failed (check the error above).")
print("Saved outputs/prompt_table.csv")