import json
with open(r"d:/googledisk8J/git/Di/pretrained_stat.ipynb", "r", encoding="utf-8-sig") as f:
    nb = json.load(f)
print("Total cells:", len(nb["cells"]))
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    print(f"Cell {i}: {cell.get('cell_type')} | exec={cell.get('execution_count')} | src_len={len(src)}")