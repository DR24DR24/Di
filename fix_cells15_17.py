import json

with open('pretrained500.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in [15, 17]:
    cell = nb['cells'][idx]
    print(f'Cell {idx} type: {cell["cell_type"]}, lines: {len("".join(cell.get("source", [])).split(chr(10)))}')