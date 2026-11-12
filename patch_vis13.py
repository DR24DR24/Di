import json

path = 'd:/googledisk6/user/goit/Diplom/var1_Di/code/Di/pretrained500.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# CELL 13 (Stage 1 Visualizations): add IPC subplot
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if i == 13 and 'Stage 1' in src and 'PC Trajectories' in src:
        # Find the PC trajectories section and add IPC subplot after it
        old = """# Layer-wise PC trajectories
for l in layer_names:\n    means = [pc_summary_by_size[n_s][l]['mean'] for n_s in sorted(pc_summary_by_size.keys())]\n    ax.plot(range(len(means)), means, marker='o', label=f'{l} (PC)', linewidth=2)\n    ax.set_xlabel('Sample Size n_s', fontweight='bold')\nax.set_ylabel('PC Value', fontweight='bold')\nax.set_title('Layer-wise Projection Correlation (PC) vs Sample Size', fontweight='bold')\nax.legend(loc='best', fontsize=8)\nax.grid(True, linestyle=':', alpha=0.6)\n\n# 8th subplot: Final PC generality profiles across all 7 models"""

        new = """# Layer-wise PC trajectories
for l in layer_names:
    means = [pc_summary_by_size[n_s][l]['mean'] for n_s in sorted(pc_summary_by_size.keys())]
    ipc_means = [ipc_summary_by_size[n_s][l]['mean'] for n_s in sorted(ipc_summary_by_size.keys())]
    ax.plot(range(len(means)), means, marker='o', label=f'{l} (PC)', linewidth=2)
    ax2.plot(range(len(ipc_means)), ipc_means, marker='s', label=f'{l} (IPC)', linewidth=2, linestyle='--')
    ax.set_xlabel('Sample Size n_s', fontweight='bold')
    ax.set_ylabel('PC Value', fontweight='bold')
    ax2.set_xlabel('Sample Size n_s', fontweight='bold')
    ax2.set_ylabel('IPC Value', fontweight='bold')
    ax.set_title('Layer-wise Projection Correlation (PC) vs Sample Size', fontweight='bold')
    ax2.set_title('Layer-wise Improved Projection Correlation (IPC) vs Sample Size', fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax2.legend(loc='best', fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax2.grid(True, linestyle=':', alpha=0.6)\n\n# 8th subplot: Final PC generality profiles across all 7 models"""

        if old in src:
            cell['source'] = src.replace(old, new).split('\n')
            print('Cell 13: IPC trajectories added')
            break
        else:
            print('Old PC trajectories section not found')
            break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Patch done (cell 13)')