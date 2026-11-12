import json

path = 'd:/googledisk6/user/goit/Diplom/var1_Di/code/Di/pretrained500.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# CELL 17: Stage 2 comparative curves - add IPC subplot
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if i == 17 and 'plt.subplots(1, 3' in src:
        # Change to 2x2 grid (4 subplots)
        old1 = "fig, axes = plt.subplots(1, 3, figsize=(18, 5))"
        new1 = "fig, axes = plt.subplots(2, 2, figsize=(18, 10))\naxes = axes.flatten()"
        if old1 in src:
            src = src.replace(old1, new1)
            
            # Replace the PC/IPC comparison section
            old_pc = """ax_pc_comp = axes_flat[7]
for k in ks:
    hist_k = stage2_results[k]['history']
    last_profile = hist_k[-1]['pc_profiles'][primary_sample_size]
    final_means = [last_profile[l]['mean'] for l in layer_names]
    ax_pc_comp.plot(range(1, 7), final_means, marker=model_markers[k], color=model_colors[k],
                    linewidth=1.8, markersize=4, label=f"M_{k} (k={k})")

ax_pc_comp.set_xticks(range(1, 7))
ax_pc_comp.set_xticklabels(layer_names, rotation=25)
ax_pc_comp.set_xlabel('Layer', fontweight='bold', fontsize=9)
ax_pc_comp.set_ylabel('Final PC (n={primary_sample_size})', fontweight='bold', fontsize=9)
ax_pc_comp.set_title('Final PC Generality Profiles', fontweight='bold', fontsize=11)
ax_pc_comp.grid(True, linestyle=':', alpha=0.5)
ax_pc_comp.legend(fontsize=8, loc='upper right')"""

            new_pc = """# Subplot 2D: Layer-wise PC vs IPC comparison (final profile, conv1 only)
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

axes[3].set_xticks(range(1, 7))
axes[3].set_xticklabels(layer_names, rotation=25)
axes[3].set_xlabel('Layer', fontweight='bold', fontsize=9)
axes[3].set_ylabel('Projection Correlation Value', fontweight='bold', fontsize=9)
axes[3].set_title('Final PC (solid) vs IPC (dashed) per Model', fontweight='bold', fontsize=11)
axes[3].grid(True, linestyle=':', alpha=0.5)
axes[3].legend(fontsize=8, loc='upper right')"""

            if old_pc in src:
                src = src.replace(old_pc, new_pc)
                break
            else:
                print('Old PC comparison section not found')
                break
        else:
            print('Old figure not found')
            break
        break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Patch 3c (cell 17) done')
