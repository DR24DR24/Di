import json

path = 'd:/googledisk6/user/goit/Diplom/var1_Di/code/Di/pretrained500.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# CELL 10: Stage 1 visualization
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'Stage 1: Source Pretraining' in src:
        old = """fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Layer-wise PC trajectories"""
        new = """fig, axes = plt.subplots(3, 1, figsize=(14, 14))

# ---------------------
# Subplot 1: Layer-wise PC trajectories (PC mode)
# ---------------------
for l in layer_names:
    pc_val_list = []
    ipc_val_list = []
    for n_s in sorted(pc_summary_by_size.keys()):
        pc_val_list.append(pc_summary_by_size[n_s].get(l, {}).get('mean', None))
        ipc_val_list.append(ipc_summary_by_size[n_s].get(l, {}).get('mean', None))
    axes[0].plot(range(len(pc_val_list)), pc_val_list, marker='o', label=f'{l} (PC)', linewidth=2)
    axes[1].plot(range(len(ipc_val_list)), ipc_val_list, marker='s', label=f'{l} (IPC)', linewidth=2, linestyle='--')

axes[0].set_xlabel('Sample Size n_s', fontweight='bold')
axes[0].set_ylabel('PC Value', fontweight='bold')
axes[0].set_title('Layer-wise Projection Correlation (PC) vs Sample Size', fontweight='bold')
axes[0].legend(loc='best', fontsize=8)
axes[0].grid(True, linestyle=':', alpha=0.6)

axes[1].set_xlabel('Sample Size n_s', fontweight='bold')
axes[1].set_ylabel('IPC Value', fontweight='bold')
axes[1].set_title('Layer-wise Improved Projection Correlation (IPC) vs Sample Size', fontweight='bold')
axes[1].legend(loc='best', fontsize=8)
axes[1].grid(True, linestyle=':', alpha=0.6)

# Subplot 3: PC vs IPC comparison for conv1
if 'conv1' in layer_names:
    pc_conv1 = [pc_summary_by_size[n_s].get('conv1', {}).get('mean', None) for n_s in sorted(pc_summary_by_size.keys())]
    ipc_conv1 = [ipc_summary_by_size[n_s].get('conv1', {}).get('mean', None) for n_s in sorted(ipc_summary_by_size.keys())]
    axes[2].plot(range(len(pc_conv1)), pc_conv1, marker='o', label='PC (conv1)', linewidth=2)
    axes[2].plot(range(len(ipc_conv1)), ipc_conv1, marker='s', label='IPC (conv1)', linewidth=2, linestyle='--')
    axes[2].set_xlabel('Sample Size n_s', fontweight='bold')
    axes[2].set_ylabel('Value', fontweight='bold')
    axes[2].set_title('Conv1: PC (solid) vs IPC (dashed) Comparison', fontweight='bold')
    axes[2].legend(loc='best', fontsize=9)
    axes[2].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig(f"results/pretrain_stage1_pc_ipc_trajectories_{timestamp}.png", dpi=150)
plt.show()"""
        
        if old in src:
            cell['source'] = src.replace(old, new).split('\n')
            print('Cell 10: updated with IPC subplots')
            break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Patch 2a done (cell 10)')
