import json
with open('d:/googledisk8J/git/Di/pretrained_stat.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        fname = f'd:/googledisk8J/git/Di/cell_{i}.py'
        with open(fname, 'w', encoding='utf-8') as f2:
            f2.write(src)
        print(f"Cell {i} saved to {fname} ({len(src)} chars)")