# Shared styling configuration
model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
model_markers = ['o', 's', '^', 'v', 'D', 'p', '*']
ks = list(range(0, 7))
layer_names = CIFAR6CNN.LAYER_NAMES
try:
    cmap = plt.colormaps['tab10']
except (AttributeError, KeyError):
    cmap = plt.cm.get_cmap('tab10')

# =========================================================================
# Figure 1: Comparative Curves across All 7 Cases (M_0 ... M_6)
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Subplot 1A: Target Validation Loss vs Epoch
for k in ks:
    hist_k = stage2_results[k]['history']
    eps = [h['epoch'] for h in hist_k]
    v_loss = [h['val_loss'] for h in hist_k]
    es = stage2_results[k]['early_stop_epoch']
    axes[0].plot(eps, v_loss, marker=model_markers[k], color=model_colors[k],
                 linewidth=1.8, markersize=4, label=f"M_{k} (k={k})")
    if es and es <= len(v_loss):
        axes[0].scatter([es], [v_loss[es-1]], color='black', s=50, zorder=5, marker='x')

axes[0].set_xlabel('Fine-Tuning Epoch', fontweight='bold')
axes[0].set_ylabel('Target Validation Loss', fontweight='bold')
axes[0].set_title('Target Val Loss vs Epoch (M_0..M_6)', fontweight='bold')
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend(fontsize=9, loc='upper right')

# Subplot 1B: Target Test Accuracy vs Epoch
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
axes[1].legend(fontsize=9, loc='lower right')

# Subplot 1C: Performance vs Transferred Depth (k)
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
ax_loss.set_xticklabels([f"M_{k}\n(k={k})" for k in ks])
ax_loss.grid(True, linestyle=':', alpha=0.6)

lines = line1 + line2 + [plt.Line2D([0], [0], color='#333333', linestyle=':', label=f'Optimal k*={best_k_loss}')]
labels = [l.get_label() for l in lines]
ax_loss.legend(lines, labels, loc='center right', fontsize=9)

plt.tight_layout()
ft_comp_path = make_result_path("stage2_curves_comparison", "png", stage2_ts)
plt.savefig(ft_comp_path, dpi=150)
plt.show()
print(f"Stage 2 comparative curves saved to: {ft_comp_path}")

# =========================================================================
# Figure 2: Individual Fine-Tuning Curves for Each of the 7 Cases
# =========================================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
axes_flat = axes.flatten()

for k in ks:
    ax = axes_flat[k]
    hist_k = stage2_results[k]['history']
    eps = [h['epoch'] for h in hist_k]
    tr_loss = [h['train_loss'] for h in hist_k]
    v_loss = [h['val_loss'] for h in hist_k]
    t_acc = [h['test_acc'] for h in hist_k]
    es = stage2_results[k]['early_stop_epoch']

    ax2 = ax.twinx()
    l1 = ax.plot(eps, tr_loss, 'o-', color='#1f77b4', linewidth=1.8, markersize=3, label='Train Loss')
    l2 = ax.plot(eps, v_loss, 's-', color='#d62728', linewidth=1.8, markersize=3, label='Val Loss')
    l3 = ax2.plot(eps, t_acc, '^--', color='#2ca02c', linewidth=1.5, markersize=3, label='Test Acc (%)')

    if es:
        ax.axvline(es, color='black', linestyle=':', linewidth=1.8, label=f'Early Stop (ep {es})')

    ax.set_title(f"Model M_{k} (k={k} transferred)", fontweight='bold', fontsize=11)
    ax.set_xlabel('Epoch', fontsize=9)
    ax.set_ylabel('Loss', fontsize=9)
    ax2.set_ylabel('Accuracy (%)', color='#2ca02c', fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.5)

    if k == 0:
        lines_all = l1 + l2 + l3
        labels_all = [l.get_label() for l in lines_all]
        ax.legend(lines_all, labels_all, loc='upper right', fontsize=8)

# 8th subplot: Summary Bar Chart
ax_sum = axes_flat[7]
bars = ax_sum.bar(ks, best_accs, color=model_colors, edgecolor='black', alpha=0.85)
ax_sum.set_xticks(ks)
ax_sum.set_xticklabels([f"M_{k}" for k in ks])
ax_sum.set_xlabel('Model', fontweight='bold')
ax_sum.set_ylabel('Best Test Accuracy (%)', fontweight='bold')
ax_sum.set_title('Best Test Accuracy per Model', fontweight='bold', fontsize=11)
ax_sum.grid(True, axis='y', linestyle=':', alpha=0.6)
for bar, acc in zip(bars, best_accs):
    ax_sum.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{acc:.1f}%",
                ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.suptitle(f"Stage 2: Individual Fine-Tuning Curves for All 7 Cases ({EXPERIMENT_CASE.upper()} Case)", fontsize=14, fontweight='bold')
plt.tight_layout()
ft_indiv_path = make_result_path("stage2_curves_all_7_cases", "png", stage2_ts)
plt.savefig(ft_indiv_path, dpi=150)
plt.show()
print(f"Stage 2 individual curves saved to: {ft_indiv_path}")

# =========================================================================
# Figure 3: Layer-wise PC Trajectories for All 7 Cases (M_0 ... M_6)
# =========================================================================
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes_flat = axes.flatten()

for k in ks:
    ax = axes_flat[k]
    hist_k = stage2_results[k]['history']
    eps = [h['epoch'] for h in hist_k]
    es = stage2_results[k]['early_stop_epoch']

    for l_idx, layer in enumerate(layer_names):
        means = [h['pc_profiles'][primary_sample_size][layer]['mean'] for h in hist_k]
        stds = [h['pc_profiles'][primary_sample_size][layer]['std'] for h in hist_k]
        ax.errorbar(
            eps, means, yerr=stds,
            fmt='o-', color=cmap(l_idx),
            linewidth=1.8, markersize=3, capsize=2,
            label=f"{layer}"
        )

    if es:
        ax.axvline(es, color='black', linestyle=':', linewidth=1.8, label=f'Early Stop (ep {es})')

    ax.set_title(f"PC Trajectories: Model M_{k} (k={k})", fontweight='bold', fontsize=11)
    ax.set_xlabel('Fine-Tuning Epoch', fontsize=9)
    ax.set_ylabel(f'PC (n={primary_sample_size})', fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.5)

# 8th subplot: Final PC generality profiles across all 7 models
ax_pc_comp = axes_flat[7]
for k in ks:
    hist_k = stage2_results[k]['history']
    last_profile = hist_k[-1]['pc_profiles'][primary_sample_size]
    final_means = [last_profile[l]['mean'] for l in layer_names]
    ax_pc_comp.plot(range(1, 7), final_means, marker=model_markers[k], color=model_colors[k],
                    linewidth=1.8, markersize=4, label=f"M_{k} (k={k})")

ax_pc_comp.set_xticks(range(1, 7))
ax_pc_comp.set_xticklabels(layer_names, rotation=25)
ax_pc_comp.set_xlabel('Layer', fontweight='bold', fontsize=9)
ax_pc_comp.set_ylabel(f'Final PC (n={primary_sample_size})', fontweight='bold', fontsize=9)
ax_pc_comp.set_title('Final PC Generality Profiles', fontweight='bold', fontsize=11)
ax_pc_comp.grid(True, linestyle=':', alpha=0.5)
ax_pc_comp.legend(fontsize=8, loc='upper right')

axes_flat[0].legend(loc='upper right', fontsize=8)

plt.suptitle(f"Stage 2: Layer-wise PC Trajectories for All 7 Cases ({EXPERIMENT_CASE.upper()} Case, n={primary_sample_size})",
             fontsize=14, fontweight='bold')
plt.tight_layout()
ft_pc_path = make_result_path("stage2_pc_trajectories_all_7_cases", "png", stage2_ts)
plt.savefig(ft_pc_path, dpi=150)
plt.show()
print(f"Stage 2 PC trajectories for all 7 cases saved to: {ft_pc_path}")
