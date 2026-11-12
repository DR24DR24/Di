import json

path = 'd:/googledisk6/user/goit/Diplom/var1_Di/code/Di/pretrained500.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# CELL 17 (Stage 2 Visualizations): add IPC subplot
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if i == 17 and 'Figure 1: Comparative Curves' in src:
        # The figure currently has 3 subplots (val loss, test accuracy, performance vs k)
        # We need to add a 4th subplot for PC vs IPC comparison
        old = """# Subplot 1C: Performance vs Transferred Depth (k)
best_losses = [stage2_results[k]['best_val_loss'] for k in ks]
best_accs = [stage2_results[k]['best_test_acc'] for k in ks]
best_k_loss = min(ks, key=lambda k: best_losses[k])

ax_loss = axes[2]
ax_acc = ax_loss.twinx()

line1 = ax_loss.plot(ks, best_losses, 's-', color='#d62728', linewidth=2.5, markersize=7, label='Best Val Loss')
line2 = ax_acc.plot(ks, best_accs, 'o--', color='#2ca02c', linewidth=2.5, markersize=7, label='Best Test Acc (%)')
ax_loss.axvline(best_k_loss, color='#333333', linestyle=':', linewidth=2, label=f'Optimal k*={best_k_loss}')

ax_loss.set_xlabel('Number of Transferred Layers (k)', fontweight='bold')
ax_loss.set_ylabel('Best Target Val Loss', color='#d62728', fontweight='bold')
ax_acc.set_ylabel('Best Target Test Accuracy (%)', color='#2ca02c', fontweight='bold')
ax_loss.set_title('Performance vs Transferred Depth (k)', fontweight='bold')
ax_loss.set_xticks(ks)
ax_loss.set_xticklabels([f"M_{k}\\n(k={k})" for k in ks])
ax_loss.grid(True, linestyle=':', alpha=0.6)

lines = line1 + line2 + [plt.Line2D([0], [0], color='#333333', linestyle=':', label=f'Optimal k*={best_k_loss}')]
labels = [l.get_label() for l in lines]
ax_loss.legend(lines, labels, loc='center right', fontsize=9)"""

        new = """# Subplot 1C: Performance vs Transferred Depth (k)
best_losses = [stage2_results[k]['best_val_loss'] for k in ks]
best_accs = [stage2_results[k]['best_test_acc'] for k in ks]
best_k_loss = min(ks, key=lambda k: best_losses[k])

ax_loss = axes[2]
ax_acc = ax_loss.twinx()

line1 = ax_loss.plot(ks, best_losses, 's-', color='#d62728', linewidth=2.5, markersize=7, label='Best Val Loss')
line2 = ax_acc.plot(ks, best_accs, 'o--', color='#2ca02c', linewidth=2.5, markersize=7, label='Best Test Acc (%)')
ax_loss.axvline(best_k_loss, color='#333333', linestyle=':', linewidth=2, label=f'Optimal k*={best_k_loss}')

ax_loss.set_xlabel('Number of Transferred Layers (k)', fontweight='bold')
ax_loss.set_ylabel('Best Target Val Loss', color='#d62728', fontweight='bold')
ax_acc.set_ylabel('Best Target Test Accuracy (%)', color='#2ca02c', fontweight='bold')
ax_loss.set_title('Performance vs Transferred Depth (k)', fontweight='bold')
ax_loss.set_xticks(ks)
ax_loss.set_xticklabels([f"M_{k}\\n(k={k})" for k in ks])
ax_loss.grid(True, linestyle=':', alpha=0.6)

lines = line1 + line2 + [plt.Line2D([0], [0], color='#333333', linestyle=':', label=f'Optimal k*={best_k_loss}')]
labels = [l.get_label() for l in lines]
ax_loss.legend(lines, labels, loc='center right', fontsize=9)

# Subplot 1D: Layer-wise PC vs IPC comparison (final profile, conv1 only)
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
        axes[3].plot(range(1, len(pc_final)+1), pc_final, marker=model_markers[k], color=model_colors[k],
                     linewidth=2, markersize=6, label=f"M_{k} - PC (conv1)", linewidth=1.5)
        axes[3].plot(range(1, len(ipc_final)+1), ipc_final, marker=model_markers[k], color=model_colors[k],
                     linewidth=1.5, markersize=5, linestyle=':', label=f"M_{k} - IPC (conv1)", alpha=0.4)

axes[3].set_xlabel('Number of Transferred Layers (k)', fontweight='bold')
axes[3].set_ylabel('Projection Correlation Value', fontweight='bold')
axes[3].set_title('Final PC (solid) vs IPC (dashed) per Model', fontweight='bold')
axes[3].grid(True, linestyle=':', alpha=0.6)
axes[3].legend(fontsize=8, loc='upper right')"""

        if old in src:
            cell['source'] = src.replace(old, new).split('\n')
            print('Cell 17: IPC final profile comparison added')
            break
        else:
            print('Old performance section not found')
            break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Patch done (cell 17)')