import json

path = 'd:/googledisk6/user/goit/Diplom/var1_Di/code/Di/pretrained500.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

checks = {
    9: ['ipc_summary_by_size', 'layer_ips', 'ipc_val', 'ipc_stats', 'return pc_summary_by_size, ipc_summary_by_size'],
    10: ['plt.subplots(3, 1', 'IPC Value'],
    12: ['plt.subplots(2, 1', 'IPC Value', 'stage2_pc_ipc_trajectories'],
    15: ['format_pc_summary(pc_stats, ipc_stats'],
}

for cell_idx, keys in checks.items():
    print('===== CELL', cell_idx, '=====')
    src = ''.join(nb['cells'][cell_idx].get('source', []))
    for key in keys:
        found = key in src
        status = 'FOUND' if found else 'MISSING'
        print('  ' + key + ': ' + status)
    print()
