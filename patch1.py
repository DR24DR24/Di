import json

path = 'd:/googledisk6/user/goit/Diplom/var1_Di/code/Di/pretrained500.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell_idx_9 = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'def evaluate_layer_pc_profiles' in src:
        cell_idx_9 = i
        break

if cell_idx_9 is not None:
    cell = nb['cells'][cell_idx_9]
    src = ''.join(cell['source'])
    
    # 1. Add layer_ips initialization
    old1 = "    pc_summary_by_size = {}"
    new1 = "    ipc_summary_by_size = {}\n    layer_ips = {l: [] for l in layer_names}"
    if old1 in src:
        src = src.replace(old1, new1)
    
    # 2. Add IPC computation after PC computation
    old2 = """                pc_val = projection_correlation(acts[l], labels_for_pc, mode='pcor')
                layer_rep_values[l].append(float(pc_val))"""
    new2 = """                pc_val = projection_correlation(acts[l], labels_for_pc, mode='pcor')
                ipc_val = projection_correlation(acts[l], labels_for_pc, mode='ipcor')
                layer_rep_values[l].append(float(pc_val))
                layer_ips[l].append(float(ipc_val))"""
    if old2 in src:
        src = src.replace(old2, new2)
    
    # 3. Add IPC stats computation and return both summaries
    old3 = """        pc_summary_by_size[n_s] = stats
    return pc_summary_by_size"""
    new3 = """        pc_summary_by_size[n_s] = stats
        ipc_stats = {}
        for l in layer_names:
            ipc_arr = np.array(layer_ips[l])
            ipc_stats[l] = {
                'mean': float(np.mean(ipc_arr)),
                'std': float(np.std(ipc_arr)),
                'values': [float(v) for v in ipc_arr]
            }
        ipc_summary_by_size[n_s] = ipc_stats
    return pc_summary_by_size, ipc_summary_by_size"""
    if old3 in src:
        src = src.replace(old3, new3)
    
    cell['source'] = src.split('\n')
    print('Cell 9 (evaluate_layer_pc_profiles): updated')
    
    # 4. Update format_pc_summary to include IPC
    for j, c in enumerate(nb['cells']):
        if j != cell_idx_9:
            src2 = ''.join(c.get('source', []))
            if 'def format_pc_summary' in src2 and 'pc_stats: Dict[int, Dict[str, Dict[str, float]]' in src2:
                old_fmt = """def format_pc_summary(pc_stats: Dict[int, Dict[str, Dict[str, float]]], target_size: int) -> str:
    \"\"\"Format PC stats as a readable string for logging.\"\"\"
    stats = pc_stats.get(target_size, {})
    parts = []
    for l in CIFAR6CNN.LAYER_NAMES:
        if l in stats:
            parts.append(f"{l}={stats[l]['mean']:.3f}\\u00B1{stats[l]['std']:.3f}")
    return " | ".join(parts)"""
                
                new_fmt = """def format_pc_summary(pc_stats: Dict[int, Dict[str, Dict[str, float]]], ipc_stats: Dict[int, Dict[str, Dict[str, float]]], target_size: int) -> str:
    \"\"\"Format PC and IPC stats as a readable string for logging.\"\"\"
    stats = pc_stats.get(target_size, {})
    ipc_stats_val = ipc_stats.get(target_size, {})
    parts = []
    for l in CIFAR6CNN.LAYER_NAMES:
        if l in stats:
            parts.append(f"{l}=PC {stats[l]['mean']:.3f}\\u00B1{stats[l]['std']:.3f} | IPC {ipc_stats_val[l]['mean']:.3f}\\u00B1{ipc_stats_val[l]['std']:.3f}")
    return " | ".join(parts)"""
                
                if old_fmt in src2:
                    c['source'] = src2.replace(old_fmt, new_fmt).split('\n')
                    print('Cell format_pc_summary: updated')
                    break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Saved cells 9-10 changes')
