epochs_arr = [h['epoch'] for h in stage1_history]
tr_loss_arr = [h['train_loss'] for h in stage1_history]
s_val_loss_arr = [h['source_val_loss'] for h in stage1_history]
t_val_loss_arr = [h['target_val_loss'] for h in stage1_history]
s_acc_arr = [h['source_test_acc'] for h in stage1_history]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Losses
axes[0].plot(epochs_arr, tr_loss_arr, 'o-', color='#1f77b4', label='Source Train Loss', linewidth=2, markersize=4)
axes[0].plot(epochs_arr, s_val_loss_arr, 's-', color='#d62728', label='Source Val Loss', linewidth=2, markersize=4)
axes[0].plot(epochs_arr, t_val_loss_arr, '^--', color='#9467bd', label='Target Val Loss', linewidth=2, markersize=4)
if early_stop_epoch:
    axes[0].axvline(early_stop_epoch, color='black', linestyle=':', linewidth=2, label=f'Early Stop (ep {early_stop_epoch})')
    axes[0].axvspan(early_stop_epoch, epochs_arr[-1], alpha=0.12, color='orange', label=f'+{POST_EARLY_STOP_EPOCHS} Control Epochs')
axes[0].set_xlabel('Epoch', fontweight='bold')
axes[0].set_ylabel('Loss', fontweight='bold')
axes[0].set_title('Stage 1: Training & Validation Losses', fontweight='bold')
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend()

# Plot 2: Source Accuracy
axes[1].plot(epochs_arr, s_acc_arr, 'o-', color='#2ca02c', label='Source Test Accuracy', linewidth=2, markersize=4)
if early_stop_epoch:
    axes[1].axvline(early_stop_epoch, color='black', linestyle=':', linewidth=2, label=f'Early Stop (ep {early_stop_epoch})')
    axes[1].axvspan(early_stop_epoch, epochs_arr[-1], alpha=0.12, color='orange')
axes[1].set_xlabel('Epoch', fontweight='bold')
axes[1].set_ylabel('Accuracy (%)', fontweight='bold')
axes[1].set_title('Stage 1: Source Classification Accuracy', fontweight='bold')
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend()

plt.tight_layout()
curves_path = make_result_path("stage1_curves", "png", stage1_ts)
plt.savefig(curves_path, dpi=150)
plt.show()
print(f"Stage 1 training curves saved to: {curves_path}")

# Plot 3: Projection Correlation Evolution per Layer
plt.figure(figsize=(12, 6))
layer_names = CIFAR6CNN.LAYER_NAMES
cmap = plt.cm.get_cmap('tab10', len(layer_names))

for l_idx, layer in enumerate(layer_names):
    means = [h['pc_profiles'][primary_sample_size][layer]['mean'] for h in stage1_history]
    stds = [h['pc_profiles'][primary_sample_size][layer]['std'] for h in stage1_history]
    plt.errorbar(
        epochs_arr, means, yerr=stds,
        fmt='o-', color=cmap(l_idx),
        linewidth=2, markersize=4, capsize=3,
        label=f'{layer}'
    )

if early_stop_epoch:
    plt.axvline(early_stop_epoch, color='black', linestyle=':', linewidth=2, label=f'Early Stop (ep {early_stop_epoch})')
    plt.axvspan(early_stop_epoch, epochs_arr[-1], alpha=0.1, color='orange')

plt.xlabel('Pretraining Epoch', fontweight='bold', fontsize=11)
plt.ylabel(f'Projection Correlation (n={primary_sample_size})', fontweight='bold', fontsize=11)
plt.title(f'Stage 1: Layer-wise PC Evolution over Pretraining Epochs ({EXPERIMENT_CASE.upper()} Case)', fontweight='bold', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()

pc_evolution_path = make_result_path("stage1_pc_evolution", "png", stage1_ts)
plt.savefig(pc_evolution_path, dpi=150)
plt.show()
print(f"Stage 1 PC evolution plot saved to: {pc_evolution_path}")
