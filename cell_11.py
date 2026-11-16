print("Stage 1.4")
print("=" * 75)
print("[Stage 1] Pretraining 6-layer CNN on Source Dataset")
print(f"Early stopping patience: {EARLY_STOP_PATIENCE} | Post-early-stop control: {POST_EARLY_STOP_EPOCHS} epochs")
print("=" * 75)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
source_model = CIFAR6CNN(num_classes=splits['num_source_classes']).to(device)
optimizer = torch.optim.RMSprop(source_model.parameters(), lr=LEARNING_RATE_SOURCE, alpha=0.9)
criterion = nn.CrossEntropyLoss()
stage1_history = []
epoch_states = []
best_s_val_loss = float('inf')
best_source_state = None
best_pc_epoch = None
best_ipc_epoch = None
best_pc_value = float('-inf')
best_ipc_value = float('-inf')
early_stop_triggered = False
early_stop_epoch = None
extra_epochs_left = POST_EARLY_STOP_EPOCHS
patience_counter = 0
debug_saved = False
primary_sample_size = SAMPLE_SIZES[0]
num_target_classes = splits['num_target_classes']
# === Model selection criterion ===
# Options: 'source_val_loss' | 'pc' | 'ipc' | 'frankenstein'
BEST_MODEL_CRITERION = 'pc'

for ep in range(1, PRETRAIN_EPOCHS + POST_EARLY_STOP_EPOCHS + 1):
    time_0 = time.perf_counter()
    # Training pass
    source_model.train()
    tr_loss = 0.0
    for x, y in loaders['source_train']:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = source_model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * x.size(0)
    avg_train_loss = tr_loss / len(loaders['source_train'].dataset)
    time_1 = time.perf_counter()
    if 1 == ep:
        print(f"train time{time_1 - time_0}")
    # Source validation & test accuracy
    s_val_loss, s_test_acc = default_eval_loss(source_model, loaders['source_test'], criterion, device)
    time_2 = time.perf_counter()
    if 1 == ep:
        print(f"eval_loss {time_2 - time_1}")
    # Target validation loss on target test set
    t_val_loss = 0  # , _ = default_eval_loss(source_model, target_test_loader, criterion, device)
    # Layer-wise Projection Correlation on fixed subsets
    if (early_stop_triggered or ep == 1) and not debug_saved:
        pc_stats, ipc_stats = evaluate_layer_pc_profiles(
            source_model, fixed_target_subsets, device=device,
            epoch=ep, Debug_Info=1,
            one_hot=True, num_classes=num_target_classes
        )
        debug_saved = True
    else:
        pc_stats, ipc_stats = evaluate_layer_pc_profiles(
            source_model, fixed_target_subsets, device=device,
            one_hot=True, num_classes=num_target_classes
        )
    pc_str = format_pc_summary(pc_stats, ipc_stats, target_size=primary_sample_size)
    time_3 = time.perf_counter()
    if 1 == ep:
        print(f"PC evaluation {time_3 - time_2}")
    # Logging line
    prefix_tag = "[Control]" if early_stop_triggered else "[Pretrain]"
    print(f"{prefix_tag} Epoch {ep:02d}/{PRETRAIN_EPOCHS:02d} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Source Test Acc: {s_test_acc * 100:.1f}% | "
          f"Source Val Loss: {s_val_loss:.4f} | "
          f"Target Val Loss: {t_val_loss:.4f} | "
          f"PC: {pc_str}")
    stage1_history.append({
        'epoch': ep,
        'train_loss': float(avg_train_loss),
        'source_val_loss': float(s_val_loss),
        'source_test_acc': float(s_test_acc * 100.0),
        'target_val_loss': float(t_val_loss),
        'pc_profiles': pc_stats,
        'is_control_phase': early_stop_triggered
    })
    # Track best epoch by PC and IPC (mean across all layers)
    pc_means = [pc_stats[primary_sample_size][l]['mean'] for l in layer_names]
    ipc_means = [ipc_stats[primary_sample_size][l]['mean'] for l in layer_names]
    mean_pc = max(pc_means)  # max across all layers instead of mean
    mean_ipc = max(ipc_means)  # max across all layers instead of mean
    if mean_pc > best_pc_value:
        best_pc_value = mean_pc
        best_pc_epoch = ep - 1  # 0-based index into epoch_states
    if mean_ipc > best_ipc_value:
        best_ipc_value = mean_ipc
        best_ipc_epoch = ep - 1
    # Save epoch state for model selection
    epoch_states.append({k: v.cpu().clone() for k, v in source_model.state_dict().items()})
    # Early stopping check on Source Val Loss
    if not early_stop_triggered:
        if s_val_loss < best_s_val_loss:
            best_s_val_loss = s_val_loss
            best_source_state = {k: v.cpu().clone() for k, v in source_model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                early_stop_triggered = True
                early_stop_epoch = ep
                print(f"\n>>> [Early Stop Triggered] at epoch {ep} (Patience={EARLY_STOP_PATIENCE}).")
                print(f">>> Running {POST_EARLY_STOP_EPOCHS} additional control epochs...\n")
    else:
        extra_epochs_left -= 1
        if extra_epochs_left <= 0:
            print(f"\n>>> Completed {POST_EARLY_STOP_EPOCHS} post-early-stop control epochs. Stopping Stage 1.\n")
            break

# === Model selection based on BEST_MODEL_CRITERION ===
if BEST_MODEL_CRITERION == 'source_val_loss':
    # best_source_state already set during training (min source val loss)
    print(f"Best model selected by Source Val Loss (epoch {early_stop_epoch if early_stop_epoch else 'N/A'}, best_s_val_loss={best_s_val_loss:.4f})")
elif BEST_MODEL_CRITERION == 'pc':
    best_source_state = {k: v.cpu().clone() for k, v in epoch_states[best_pc_epoch].items()}
    print(f"Best model selected by PC (epoch {best_pc_epoch + 1}, max PC={best_pc_value:.4f})")
elif BEST_MODEL_CRITERION == 'ipc':
    best_source_state = {k: v.cpu().clone() for k, v in epoch_states[best_ipc_epoch].items()}
    print(f"Best model selected by IPC (epoch {best_ipc_epoch + 1}, max IPC={best_ipc_value:.4f})")
elif BEST_MODEL_CRITERION == 'frankenstein':
    layer_names = CIFAR6CNN.LAYER_NAMES
    pc_per_epoch = []
    for h in stage1_history:
        pc_per_epoch.append(h['pc_profiles'][primary_sample_size])

    # Select epoch for each layer
    selected_epochs = {}
    # conv1: epoch with max PC(conv1)
    selected_epochs['conv1'] = max(range(len(epoch_states)), key=lambda e: pc_per_epoch[e]['conv1']['mean'])
    # conv2..conv6: epoch with max |PC(prev) - PC(curr)|
    for j in range(1, len(layer_names)):
        prev_layer = layer_names[j-1]
        curr_layer = layer_names[j]
        selected_epochs[curr_layer] = max(range(len(epoch_states)), key=lambda e: abs(pc_per_epoch[e][prev_layer]['mean'] - pc_per_epoch[e][curr_layer]['mean']))

    # Build frankenstein state dict
    frankenstein_state_dict = {}
    for j, layer_name in enumerate(layer_names):
        src_epoch = selected_epochs[layer_name]
        # Model registers layers both as conv1..conv6 and layers.0..layers.5
        prefixes = (f'layers.{j}.', f'{layer_name}.')
        for key, value in epoch_states[src_epoch].items():
            if key.startswith(prefixes):
                frankenstein_state_dict[key] = value.clone()
    # Classifier from the last epoch
    for key, value in epoch_states[-1].items():
        if key.startswith('classifier'):
            frankenstein_state_dict[key] = value.clone()

    # Replace best model with Frankenstein
    best_source_state = frankenstein_state_dict
    print(f"Frankenstein model assembled: layer epochs = {selected_epochs}")
else:
    raise ValueError(f"Unknown BEST_MODEL_CRITERION: {BEST_MODEL_CRITERION}")

# Save Stage 1 artifacts with timestamp
stage1_ts = get_timestamp()
model_save_path = make_result_path("pretrained_model", "pt", stage1_ts)
torch.save({
    'best_state_dict': best_source_state,
    'final_state_dict': source_model.state_dict(),
    'early_stop_epoch': early_stop_epoch,
    'best_s_val_loss': best_s_val_loss}, model_save_path)
print(f"Pretrained model checkpoint saved to: {model_save_path}")
history_save_path = make_result_path("stage1_history", "json", stage1_ts)
with open(history_save_path, 'w', encoding='utf-8') as f:
    json.dump({
        'stage': 'stage1_pretraining',
        'experiment_case': EXPERIMENT_CASE,
        'early_stop_epoch': early_stop_epoch,
        'post_early_stop_epochs': POST_EARLY_STOP_EPOCHS,
        'best_s_val_loss': best_s_val_loss,
        'history': stage1_history}, f, indent=2)
print(f"Stage 1 metrics history saved to: {history_save_path}")

