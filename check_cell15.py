import json
with open(r"d:/googledisk8J/git/Di/pretrained_stat.ipynb", "r", encoding="utf-8-sig") as f:
    nb = json.load(f)
# Print full content of cell 15
cell = nb['cells'][15]
src = "".join(cell.get('source', []))
print(src)