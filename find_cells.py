import json

path = 'd:/googledisk6/user/goit/Diplom/var1_Di/code/Di/pretrained500.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find all code cells with PC plotting
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'pc_summary_by_size' in src or 'ipc_summary_by_size' in src or 'format_pc_summary' in src or 'pc_trajectories' in src or 'PC ' in src or 'PC(' in src:
        if 'plt' in src or 'print(' in src or 'pc_str' in src:
            print(f'===== CELL {i} (code) =====')
            print(src[:3500])
            print()
