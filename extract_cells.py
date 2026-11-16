import json
with open('d:/googledisk8J/git/Di/pretrained_stat.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        print(f"=== CELL {i} (exec_count={cell.get('execution_count')}) ===")
        print(src)
        print()