import json
with open(r"d:/googledisk8J/git/Di/pretrained_stat.ipynb", "r", encoding="utf-8-sig") as f:
    nb = json.load(f)
# Print cell 15 and 17 in full
for i in [13, 15, 17, 19]:
    cell = nb['cells'][i]
    src = "".join(cell.get('source', []))
    print(f"=== CELL {i} ===")
    print(f"Length: {len(src)}")
    print(src[:2000])
    print("...")
    print()