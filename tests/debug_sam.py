"""Run from project root:  python -m tests.debug_sam
Shows what SAM 3.1 returns for several prompts. 'tooth' is a control: teeth are in the photo,
so if it finds regions your parsing works and the lesion prompts simply matched nothing."""
from core.segment import LesionSegmenter, RECORD_RE
from core.utils import load_resized

IMG = "data/test_images/sample1.jpeg"
CONCEPTS = ["tooth", "white patch", "red patch", "ulcer", "oral lesion", "lesion", "dark patch", "mouth sore"]

seg = LesionSegmenter()
_, jpeg = load_resized(IMG)
for c in CONCEPTS:
    text = seg._call_sam(jpeg, c, use_cache=False)
    n = len(RECORD_RE.findall(text))
    print(f"{c:12s} chars={len(text):6d}  boxes_in_text={text.count('<|box'):2d}  parsed_records={n:2d}")
    if text and n == 0:
        print("   RAW START:", text[:300])
