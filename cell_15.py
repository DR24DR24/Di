print("=" * 75)
print(f"[Stage 2] Target Fine-Tuning on Fixed SAMPLE_SIZES for All 7 Cases (M_0 ... M_6)")
print(f"Fixed sample size: n={primary_sample_size} | Early stopping patience: {TARGET_PATIENCE} | Max epochs: {TARGET_EPOCHS}")
VERBOSE = False
print("=" * 75)
# Fixed target training loader (using the first replication of primary_sample_size)
ft_sample_size = FT_SAMPLE_SIZE if FT_SAMPLE_SIZE is not None else primary_sample_size

target_train_subset_loader, target_val_subset_loader = create_ft_dataloaders_from_subsets(
    fixed_target_subsets=fixed_target_subsets,
    target_train_x=target_train_x,
    target_train_y=target_train_y,
    sample_size=primary_sample_size,
    ft_sample_size=ft_sample_size,
    batch_size=64,
    seed=SEED,
    generator=generator,
)
num_target_classes = splits['num_target_classes']
target_criterion = nn.CrossEntropyLoss()
stage2_results = {}
stage2_ts = get_timestamp()

for k in range(0, 7):
    model_name = f"M_{k}"
    if VERBOSE:
        print("\n" + "=" * 75)
        print(f"[Stage 2] Model {model_name} (transferred k={k} layers from pre-trained source model)")
        print(f"Fine-tuning on fixed target subset (n={primary_sample_size}), monitored metric: Target Validation Loss")
        print("=" * 75)
    # Load best checkpoint from Stage 1 into source_model before transferring layers
    if best_source_state is not None:
        source_model.load_state_dict(best_source_state)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    target_model = create_transferred_model(
        pretrained_model=source_model,
        k=k,
        num_target_classes=num_target_classes,
        freeze_transferred=FEATURE_EXTRACTION
    ).to(device)
    set_trainable_layers(target_model, k_freeze=k if FEATURE_EXTRACTION else 0)
    target_optimizer = torch.optim.RMSprop(
        target_model.parameters(),
        lr=LEARNING_RATE_TARGET,
        alpha=0.9
    )
    stage2_history_k = []
    best_t_val_loss = float('inf')
    best_t_val_acc = 0.0
    best_target_state = None
    target_patience_counter = 0
    target_early_stop_epoch = None
    for ep in range(1, TARGET_EPOCHS + 1):
        target_model.train()
        tr_loss = 0.0
        for x, y in target_train_subset_loader:
            x, y = x.to(device), y.to(device)
            target_optimizer.zero_grad()
            out = target_model(x)
            loss = target_criterion(out, y)
            loss.backward()
            target_optimizer.step()
            tr_loss += loss.item() * x.size(0)
        avg_train_loss = tr_loss / len(target_train_subset_loader.dataset)
        # Evaluate loss and accuracy on full target validation/test set
        val_loss, val_acc = default_eval_loss(target_model, target_val_subset_loader, target_criterion, device)
        test_loss, test_acc = default_eval_loss(target_model, target_test_loader, target_criterion, device)
        # Compute PC on fixed subsets during fine-tuning
        try:
            pc_stats, ipc_stats = evaluate_layer_pc_profiles(
                target_model, fixed_target_subsets, device=device,
                one_hot=True, num_classes=num_target_classes
            )
        except TypeError:
            pc_stats, ipc_stats = evaluate_layer_pc_profiles(target_model, fixed_target_subsets, device=device)
        pc_str = format_pc_summary(pc_stats, ipc_stats, target_size=primary_sample_size)
        # Logged each epoch:
        if VERBOSE:
            print(f"[{model_name}] Epoch {ep:02d}/{TARGET_EPOCHS:02d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.1f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.1f}% | "
              f"PC: {pc_str}")
        stage2_history_k.append({
            'epoch': ep,
            'train_loss': float(avg_train_loss),
            'val_loss': float(val_loss),
            'val_acc': float(val_acc * 100.0),
            'test_loss': float(test_loss),
            'test_acc': float(test_acc * 100.0),
            'pc_profiles': pc_stats
        })
        # Early stopping check on Target Validation Loss
        if val_loss < best_t_val_loss:
            best_t_val_loss = val_loss
            best_t_val_acc = val_acc * 100.0
            best_target_state = {key: val.cpu().clone() for key, val in target_model.state_dict().items()}
            target_patience_counter = 0
        else:
            target_patience_counter += 1
            if target_patience_counter >= TARGET_PATIENCE:
                target_early_stop_epoch = ep
                if VERBOSE:
                    print(f"\n>>> [Target Early Stop] {model_name} triggered at epoch {ep} (Patience={TARGET_PATIENCE}). Stopping Stage 2 for {model_name}.\n")
                break

    # ========================================================================
    # Phase 2: Fine-tuning after Feature Extraction (same early stopping)
    # ========================================================================
    if FEATURE_EXTRACTION and k > 0 and best_target_state is not None:
        if VERBOSE:
            print(f"\n[{model_name}] >>> Phase 2: Fine-tuning after Feature Extraction <<<")
        # Load best state from Phase 1
        target_model.load_state_dict(best_target_state)
        # Unfreeze ALL layers for fine-tuning
        set_trainable_layers(target_model, k_freeze=0)
        # Fresh optimizer for all trainable parameters
        target_optimizer_ft = torch.optim.RMSprop(
            target_model.parameters(),
            lr=LEARNING_RATE_TARGET,
            alpha=0.9
        )
        # Fine-tuning loop with the SAME early stopping criterion
        best_t_val_loss_ft = best_t_val_loss
        best_t_val_acc_ft = best_t_val_acc
        best_target_state_ft = {
            key: val.cpu().clone() for key, val in target_model.state_dict().items()
        }
        target_patience_counter_ft = 0
        target_early_stop_epoch_ft = None

        for ep in range(1, TARGET_EPOCHS + 1):
            target_model.train()
            tr_loss = 0.0
            for x, y in target_train_subset_loader:
                x, y = x.to(device), y.to(device)
                target_optimizer_ft.zero_grad()
                out = target_model(x)
                loss = target_criterion(out, y)
                loss.backward()
                target_optimizer_ft.step()
                tr_loss += loss.item() * x.size(0)
            avg_train_loss = tr_loss / len(target_train_subset_loader.dataset)
            val_loss, val_acc = default_eval_loss(
                target_model, target_val_subset_loader, target_criterion, device
            )
            test_loss, test_acc = default_eval_loss(
                target_model, target_test_loader, target_criterion, device
            )

            if val_loss < best_t_val_loss_ft:
                best_t_val_loss_ft = val_loss
                best_t_val_acc_ft = val_acc * 100.0
                best_target_state_ft = {
                    key: val.cpu().clone()
                    for key, val in target_model.state_dict().items()
                }
                target_patience_counter_ft = 0
            else:
                target_patience_counter_ft += 1
                if target_patience_counter_ft >= TARGET_PATIENCE:
                    target_early_stop_epoch_ft = ep
                    if VERBOSE:
                        print(
                            f"\n>>> [Target Early Stop FT] {model_name} triggered at epoch {ep} "
                            f"(Patience={TARGET_PATIENCE})."
                        )
                    break

        # Compare Phase 2 vs Phase 1
        if best_t_val_loss_ft < best_t_val_loss:
            improvement = best_t_val_loss - best_t_val_loss_ft
            if VERBOSE:
                print(
                    f"\n>>> [{model_name}] Fine-tuning after Feature Extraction IMPROVED result: "
                    f"Val Loss {best_t_val_loss:.4f} -> {best_t_val_loss_ft:.4f} "
                    f"(delta={improvement:.4f})"
                )
            # Adopt fine-tuned results
            best_t_val_loss = best_t_val_loss_ft
            best_t_val_acc = best_t_val_acc_ft
            best_target_state = best_target_state_ft
            target_early_stop_epoch = (
                target_early_stop_epoch_ft if target_early_stop_epoch_ft else target_early_stop_epoch
            )
        else:
            if VERBOSE:
                print(
                    f"\n>>> [{model_name}] Fine-tuning after Feature Extraction did NOT improve: "
                    f"Val Loss {best_t_val_loss:.4f} -> {best_t_val_loss_ft:.4f}"
                )
    # ========================================================================

    # Save individual model checkpoint
    ft_model_path = make_result_path(f"finetuned_model_{model_name}", "pt", stage2_ts)
    torch.save({
        'k': k,
        'model_name': model_name,
        'best_state_dict': best_target_state,
        'final_state_dict': target_model.state_dict(),
        'best_val_loss': best_t_val_loss,
        'best_test_acc': best_t_val_acc,
        'early_stop_epoch': target_early_stop_epoch
    }, ft_model_path)
    if VERBOSE:
        print(f"[{model_name}] Checkpoint saved to: {ft_model_path}")
    stage2_results[k] = {
        'k': k,
        'model_name': model_name,
        'transferred_layers': k,
        'best_val_loss': float(best_t_val_loss),
        'best_test_acc': float(best_t_val_acc),
        'early_stop_epoch': target_early_stop_epoch,
        'total_epochs': len(stage2_history_k),
        'history': stage2_history_k,
        'checkpoint_path': ft_model_path
    }

# Backward compatibility alias
best_k_idx = min(stage2_results.keys(), key=lambda k: stage2_results[k]['best_val_loss'])
stage2_history = stage2_results[best_k_idx]['history']
target_early_stop_epoch = stage2_results[best_k_idx]['early_stop_epoch']


# Save complete Stage 2 results across all 7 cases
stage2_history_path = make_result_path("stage2_history_all_cases", "json", stage2_ts)
with open(stage2_history_path, 'w', encoding='utf-8') as f:
    json.dump({
        'stage': 'stage2_fine_tuning_all_cases',
        'experiment_case': EXPERIMENT_CASE,
        'target_sample_size': primary_sample_size,
        'target_patience': TARGET_PATIENCE,
        'target_epochs': TARGET_EPOCHS,
        'models': {
            f"M_{k}": {
                'k': res['k'],
                'best_val_loss': res['best_val_loss'],
                'best_test_acc': res['best_test_acc'],
                'early_stop_epoch': res['early_stop_epoch'],
                'total_epochs': res['total_epochs'],
                'history': res['history']
            } for k, res in stage2_results.items()
        }
    }, f, indent=2)
if VERBOSE:
    print("\n" + "=" * 75)
    print(f"Stage 2 (All 7 Cases) completed successfully!")
    print(f"Metrics saved to: {stage2_history_path}")
    print("=" * 75)
# Summary table of all 7 cases
if VERBOSE:
    print(f"{'Model':<8} | {'Transferred (k)':<16} | {'Best Val Loss':<14} | {'Best Test Acc (%)':<18} | {'Early Stop Epoch':<16}")
if VERBOSE:
    print("-" * 80)
for k in range(0, 7):
    res = stage2_results[k]
    es_str = f"Epoch {res['early_stop_epoch']}" if res['early_stop_epoch'] else f"Max ({res['total_epochs']})"
    if VERBOSE:
        print(f"{res['model_name']:<8} | {res['transferred_layers']:<16} | {res['best_val_loss']:<14.4f} | {res['best_test_acc']:<18.2f} | {es_str:<16}")
if VERBOSE:
    print("-" * 80)
if VERBOSE:
    print(f"Optimal Model across all 7 cases: M_{best_k_idx} (k* = {best_k_idx}) with Val Loss = {stage2_results[best_k_idx]['best_val_loss']:.4f}")
print("=" * 75)

