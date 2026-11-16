import json
with open(r"d:/googledisk8J/git/Di/pretrained_stat.ipynb", "r", encoding="utf-8-sig") as f:
    nb = json.load(f)
# Print full content of cell 13 (plot_stage1_visualizations)
cell = nb['cells'][13]
src = "".join(cell.get('source', []))
print("=== CELL 13 (plot_stage1_visualizations) ===")
print(src)