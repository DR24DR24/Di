import json

path = 'd:/googledisk6/user/goit/Diplom/var1_Di/code/Di/pretrained500.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# CELL 12: Stage 2 visualization
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'Stage 2' in src and 'PC' in src:
        old = """fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Layer-wise PC trajectories"""
        new = """fig, axes = plt.subplots(2, 1, figsize=(14, 9))

# Layer-wise PC trajectories (PC mode)
for l in layer_names:
    if l in pc_summary_by_size.get(target_size, {}):
        pc_vals = [pc_summary_by_size[target_size][l]['mean']]
        ipc_vals = [ipc_summary_by_size[target_size][l]['mean']]
        axes[0].plot(range(len(pc_vals)), pc_vals, marker='o', label=f'{l} (PC)', linewidth=2)
        axes[1].plot(range(len(ipc_vals)), ipc_vals, marker='s', label=f'{l} (IPC)', linewidth=2, linestyle='--')

axes[0].set_xlabel('Epoch', fontweight='bold')
axes[0].set_ylabel('PC Value', fontweight='bold')
axes[0].set_title('Layer-wise Projection Correlation (PC) During Fine-Tuning', fontweight='bold')
axes[0].legend(loc='best', fontsize=8)
axes[0].grid(True, linestyle=':', alpha=0.6)

axes[1].set_xlabel('Epoch', fontweight='bold')
axes[1].set_ylabel('IPC Value', fontweight='bold')
axes[1].set_title('Layer-wise Improved Projection Correlation (IPC) During Fine-Tuning', fontweight='bold')
axes[1].legend(loc='best', fontsize=8)
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig(f"results/stage2_pc_ipc_trajectories_{timestamp}.png", dpi=150)
plt.show()"""
        
        if old in src:
            cell['source'] = src.replace(old, new).split('\n')
            print('Cell 12: updated with IPC subplots')
            break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Patch 2b done (cell 12)')
