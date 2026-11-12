import json

path = 'd:/googledisk6/user/goit/Diplom/var1_Di/code/Di/pretrained500.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find cells with PC plots
print("=== Cells with PC plotting ===")
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'pc_summary_by_size' in src or 'plot(' in src and 'PC' in src:
        print(f'Cell {i}: type={cell["cell_type"]}, len={len(src)}')
        if 'pc_summary_by_size' in src:
            print('  Has pc_summary_by_size')
        if 'plt.subplots' in src:
            print('  Has plt.subplots')
        if 'format_pc_summary' in src:
            print('  Has format_pc_summary')
        print()

# Find stages
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'Stage 1' in src and ('Pretraining' in src or 'Source Pretraining' in src):
        print(f'Stage 1 cell: {i}')
    if 'Stage 2' in src and ('Fine-Tuning' in src or 'Fine-Tuning' in src or 'Fine' in src):
        print(f'Stage 2 cell: {i}')
