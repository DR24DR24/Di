import json
with open(r"d:/googledisk8J/git/Di/pretrained_stat.ipynb", "r", encoding="utf-8-sig") as f:
    nb = json.load(f)
for i in [13, 15, 17, 19]:
    cell = nb['cells'][i]
    src = "".join(cell.get('source', []))
    print("=== CELL {} (exec={}) ===".format(i, cell.get('execution_count')))
    print(src)
    print()