import json

path = 'd:/googledisk6/user/goit/Diplom/var1_Di/code/Di/pretrained500.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# CELL 17: all 7 cases visualizations - add IPC comparison subplot
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'Figure 1: Comparative Curves across All 7 Cases' in src:
        old = """# Subplot 1B: Target Test Accuracy vs Epoch
for k in ks:
    hist_k = stage2_results[k]['history']
    eps = [h['epoch'] for h in hist_k]
    t_acc = [h['test_acc'] for h in hist_k]
    axes[1].plot(eps, t_acc, marker=model_markers[k], color=model_colors[k],
                 linewidth=1.8, markersize=4, label=f"M_{k} (k={k})")

axes[1].set_xlabel('Fine-Tuning Epoch', fontweight='bold')
axes[1].set_ylabel('Target Test Accuracy (%)', fontweight='bold')
axes[1].set_title('Target Test Accuracy vs Epoch (M_0..M_6)', fontweight='bold')
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend(fontsize=9, loc='upper right')"""

        new = """# Subplot 1B: Target Test Accuracy vs Epoch
for k in ks:
    hist_k = stage2_results[k]['history']
    eps = [h['epoch'] for h in hist_k]
    t_acc = [h['test_acc'] for h in hist_k]
    axes[1].plot(eps, t_acc, marker=model_markers[k], color=model_colors[k],
                 linewidth=1.8, markersize=4, label=f"M_{k} (k={k})")

axes[1].set_xlabel('Fine-Tuning Epoch', fontweight='bold')
axes[1].set_ylabel('Target Test Accuracy (%)', fontweight='bold')
axes[1].set_title('Target Test Accuracy vs Epoch (M_0..M_6)', fontweight='bold')
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend(fontsize=9, loc='upper right')

# Subplot 1C: Layer-wise PC vs IPC comparison (final profile)
for l in layer_names:
    pc_final = []
    ipc_final = []
    for k in ks:
        if stage2_results[k]['history']:
            last_profile = stage2_results[k]['history'][-1]['pc_profiles'].get(primary_sample_size, {})
            pc_final.append(last_profile.get(l, {}).get('mean', None))
            ipc_final.append(last_profile.get(l, {}).get('ipc_mean', None))
    axes[2].plot(range(1, len(pc_final)+1), pc_final, marker=model_markers[k], color=model_colors[k],
                 linewidth=2, markersize=6, label=f"M_{k} - PC (conv1)" if l=='conv1' else "", linewidth=1.5)
    axes[2].plot(range(1, len(ipc_final)+1), ipc_final, marker=model_markers[k], color=model_colors[k],
                 linewidth=1.5, markersize=5, linestyle=':', label=f"M_{k} - IPC (conv1)" if l=='conv1' else "", alpha=0.4)

axes[2].set_xlabel('Number of Transferred Layers (k)', fontweight='bold')
axes[2].set_ylabel('Projection Correlation Value', fontweight='bold')
axes[2].set_title('Final PC (solid) vs IPC (dashed) per Model', fontweight='bold')
axes[2].grid(True, linestyle=':', alpha=0.6)
axes[2].legend(fontsize=8, loc='upper right')"""
        
        if old in src:
            cell['source'] = src.replace(old, new).split('\n')
            print('Cell 17: updated with IPC comparison subplot')
            break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Patch 2c done (cell 17)')
