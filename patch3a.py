import json

path = 'd:/googledisk6/user/goit/Diplom/var1_Di/code/Di/pretrained500.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# CELL 13 (Stage 1 visualization): add IPC subplot
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if i == 13 and 'plt.subplots(' in src and 'pc_summary_by_size' in src:
        # Find the PC plot code and add IPC subplot
        old = """axes[0].set_title('PC Trajectories: Model M_{k} (k={k})', fontweight='bold', fontsize=11)
    ax.set_xlabel('Fine-Tuning Epoch', fontsize=9)
    ax.set_ylabel(f'PC (n={primary_sample_size})', fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.5)

# 8th subplot: Final PC generality profiles across all 7 models"""

        new = """axes[0].set_title('PC Trajectories: Model M_{k} (k={k})', fontweight='bold', fontsize=11)
    ax.set_xlabel('Fine-Tuning Epoch', fontsize=9)
    ax.set_ylabel(f'PC (n={primary_sample_size})', fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.5)

# 8th subplot: Final PC generality profiles across all 7 models (PC mode)"""

        # Find the subplot with PC trajectories and add IPC subplot after it
        old2 = """    ax.set_ylabel(f'PC (n={primary_sample_size})', fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.5)

# 8th subplot: Final PC generality profiles"""

        new2 = """    ax.set_ylabel(f'PC (n={primary_sample_size})', fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.5)

# 8th subplot: Final PC + IPC comparison across all 7 models (PC mode)
ax_pc_comp = axes_flat[7]
for k in ks:
    hist_k = stage2_results[k]['history']
    pc_final = []
    ipc_final = []
    for h in hist_k:
        if h['pc_profiles'] and primary_sample_size in h['pc_profiles']:
            last_profile = h['pc_profiles'][primary_sample_size]
            pc_final.append(last_profile.get('conv1', {}).get('mean', None))
            ipc_final.append(last_profile.get('conv1', {}).get('ipc_mean', None))
    if pc_final:
        axes_flat[7].plot(range(1, len(pc_final)+1), pc_final, marker=model_markers[k], color=model_colors[k],
                         linewidth=2, markersize=6, label=f"M_{k} - PC (conv1)", linewidth=1.5)
        axes_flat[7].plot(range(1, len(ipc_final)+1), ipc_final, marker=model_markers[k], color=model_colors[k],
                         linewidth=1.5, markersize=5, linestyle=':', label=f"M_{k} - IPC (conv1)", alpha=0.4)

axes_flat[7].set_xticks(range(1, 7))
axes_flat[7].set_xticklabels(layer_names, rotation=25)
axes_flat[7].set_xlabel('Layer', fontweight='bold', fontsize=9)
axes_flat[7].set_ylabel('Value', fontweight='bold', fontsize=9)
axes_flat[7].set_title('Final PC (solid) vs IPC (dashed) per Model', fontweight='bold', fontsize=11)
axes_flat[7].grid(True, linestyle=':', alpha=0.5)
axes_flat[7].legend(fontsize=8, loc='upper right')

axes_flat[0].legend(loc='upper right', fontsize=8)"""

        if old2 in src:
            cell['source'] = src.replace(old2, new2).split('\n')
            print('Cell 13: IPC subplot added')
            break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Patch 3a (cell 13) done')
