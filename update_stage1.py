import json
import os

NB_PATH = 'd:/googledisk8J/git/Di/pretrained_stat.ipynb'
CELL11_PATH = 'd:/googledisk8J/git/Di/cell_11.py'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# === Update Cell 9: ensure layer_ips is per sample size ===
cell9_src = ''.join(nb['cells'][9]['source'])
old_c9 = """    layer_ips = {l: [] for l in layer_names}
    for n_s, rep_list in subsets_dict.items():
        layer_rep_values = {l: [] for l in layer_names}"""

new_c9 = """    for n_s, rep_list in subsets_dict.items():
        layer_rep_values = {l: [] for l in layer_names}
        layer_ips = {l: [] for l in layer_names}"""

if old_c9 in cell9_src:
    cell9_src = cell9_src.replace(old_c9, new_c9)
    nb['cells'][9]['source'] = cell9_src.splitlines(keepends=True)
    print("Cell 9 updated (layer_ips reset per sample size).")
else:
    print("Cell 9 already updated or pattern not found.")

# === Update Cell 11: def run_stage1_pretraining ===
new_cell11_src = '''def run_stage1_pretraining(
    device: torch.device,
    seed: int,
    loaders: Dict[str, Any],
    num_source_classes: int,
    fixed_target_subsets: Dict[int, List[Dict[str, Any]]],
    primary_sample_size: int,
    num_target_classes: int,
    target_test_loader: Optional[DataLoader] = None,
    pretrain_epochs: int = 50,
    early_stop_patience: int = 5,
    post_early_stop_epochs: int = 1,
    learning_rate_source: float = 0.001,
    best_model_criterion: str = 'pc',
    experiment_case: str = 'imbalance',
    results_dir: Optional[str] = None,
    top_level_func_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Stage 1: Source pretraining with early stopping and model selection.

    1.1 Early stopping (AND logic across 3 conditions):
        - Condition 1: Minimum validation loss + early_stop_patience epochs (5 epochs)
        - Condition 2: Maximum PC + early_stop_patience epochs (5 epochs)
        - Condition 3: Maximum IPC + early_stop_patience epochs (5 epochs)
        Training continues until ALL THREE conditions are met simultaneously.

    1.2 Logging directory:
        Saved in <results_dir>/<top_level_func_name>_<short_hash>/
        where short_hash is up to 8 alphanumeric characters based on parameters.
        Filenames reflect their content. Re-running with identical parameters
        updates (overwrites) existing report files in place.

    1.3 Model logs for all epochs:
        Saved to stage1_model_history.json and pretrained_model.pt.

    1.4 Launch parameters:
        Saved to launch_parameters.txt in human-readable key-value format.

    1.5 CSV table with PC and IPC:
        Saved to pc_ipc_layer_profiles.csv with columns:
        epoch, dataset_id (-1=mean, -2=std/error), PC by layers, IPC by layers.
    """
    import inspect
    import hashlib
    import csv

    # Determine top-level function name if not explicitly provided
    if top_level_func_name is None:
        try:
            stack = inspect.stack()
            for f in reversed(stack):
                name = f.function
                if name and not name.startswith(('<', '_')) and name not in (
                    'run_code', 'run_ast_nodes', 'run_cell', 'run_cell_async', 'run_stage1_pretraining'
                ):
                    top_level_func_name = name
                    break
        except Exception:
            pass
        if not top_level_func_name:
            top_level_func_name = "run_stage1_pretraining"

    # Compute short hash (up to 8 alphanumeric characters) based on launch parameters
    num_reps = len(fixed_target_subsets.get(primary_sample_size, [])) if fixed_target_subsets else 0
    hash_params = {
        'best_model_criterion': str(best_model_criterion),
        'early_stop_patience': int(early_stop_patience),
        'experiment_case': str(experiment_case),
        'learning_rate_source': float(learning_rate_source),
        'num_replications': int(num_reps),
        'num_source_classes': int(num_source_classes),
        'num_target_classes': int(num_target_classes),
        'post_early_stop_epochs': int(post_early_stop_epochs),
        'pretrain_epochs': int(pretrain_epochs),
        'primary_sample_size': int(primary_sample_size),
        'seed': int(seed),
    }
    param_str = json.dumps(hash_params, sort_keys=True)
    short_hash = hashlib.sha256(param_str.encode('utf-8')).hexdigest()[:8]

    # Setup log directory: <results_dir>/<top_level_func_name>_<short_hash>
    if results_dir is None:
        results_dir = os.path.join(os.getcwd(), "pretrained")
    dir_name = f"{top_level_func_name}_{short_hash}"
    log_dir = os.path.join(results_dir, dir_name)
    os.makedirs(log_dir, exist_ok=True)

    # 1.4 Save launch parameters in readable key-value format
    launch_params = dict(hash_params)
    launch_params['device'] = str(device)
    launch_params['top_level_function'] = top_level_func_name
    launch_params['short_hash'] = short_hash
    launch_params['results_dir'] = results_dir
    launch_params['log_dir'] = log_dir

    params_txt_path = os.path.join(log_dir, "launch_parameters.txt")
    with open(params_txt_path, 'w', encoding='utf-8') as f_params:
        f_params.write("=" * 60 + "\\n")
        f_params.write("STAGE 1 LAUNCH PARAMETERS\\n")
        f_params.write("=" * 60 + "\\n")
        for k in sorted(launch_params.keys()):
            f_params.write(f"{k:<25}: {launch_params[k]}\\n")
        f_params.write("=" * 60 + "\\n")

    params_json_path = os.path.join(log_dir, "launch_parameters.json")
    with open(params_json_path, 'w', encoding='utf-8') as f_pjson:
        json.dump(launch_params, f_pjson, indent=2, default=str)

    # 1.5 Setup CSV file for PC and IPC layer profiles
    csv_path = os.path.join(log_dir, "pc_ipc_layer_profiles.csv")
    layer_names = CIFAR6CNN.LAYER_NAMES
    csv_fieldnames = ['epoch', 'dataset_id'] + [f'pc_{l}' for l in layer_names] + [f'ipc_{l}' for l in layer_names]
    csv_rows = []

    print("Stage 1.4")
    print("=" * 75)
    print("[Stage 1] Pretraining 6-layer CNN on Source Dataset")
    print(f"Log directory: {log_dir}")
    print(f"Early stop patience: {early_stop_patience} epochs (val_loss min + {early_stop_patience} AND PC max + {early_stop_patience} AND IPC max + {early_stop_patience})")
    print(f"Post-early-stop control: {post_early_stop_epochs} epochs")
    print("=" * 75)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    source_model = CIFAR6CNN(num_classes=num_source_classes).to(device)
    optimizer = torch.optim.RMSprop(source_model.parameters(), lr=learning_rate_source, alpha=0.9)
    criterion = nn.CrossEntropyLoss()

    stage1_history = []
    epoch_states = []

    best_s_val_loss = float('inf')
    best_val_loss_epoch = 1
    best_source_state = None

    best_pc_epoch = 1
    best_ipc_epoch = 1
    best_pc_value = float('-inf')
    best_ipc_value = float('-inf')

    early_stop_triggered = False
    early_stop_epoch = None
    extra_epochs_left = post_early_stop_epochs
    debug_saved = False

    for ep in range(1, pretrain_epochs + post_early_stop_epochs + 1):
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
            print(f"train time: {time_1 - time_0:.3f}s")

        # Source validation & test accuracy
        s_val_loss, s_test_acc = default_eval_loss(source_model, loaders['source_test'], criterion, device)
        time_2 = time.perf_counter()
        if 1 == ep:
            print(f"eval_loss: {time_2 - time_1:.3f}s")

        # Target validation loss on target test set (if compatible)
        if target_test_loader is not None and num_source_classes == num_target_classes:
            try:
                t_val_loss, _ = default_eval_loss(source_model, target_test_loader, criterion, device)
            except Exception:
                t_val_loss = 0.0
        else:
            t_val_loss = 0.0

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
            print(f"PC evaluation: {time_3 - time_2:.3f}s")

        # 1.5 Record PC and IPC values into CSV table rows:
        # 1) Per-dataset rows (replications 0 .. N-1)
        sub_list = fixed_target_subsets.get(primary_sample_size, [])
        for rep_idx in range(len(sub_list)):
            row = {'epoch': ep, 'dataset_id': rep_idx}
            for l in layer_names:
                row[f'pc_{l}'] = pc_stats[primary_sample_size][l]['values'][rep_idx]
                row[f'ipc_{l}'] = ipc_stats[primary_sample_size][l]['values'][rep_idx]
            csv_rows.append(row)

        # 2) Mean across datasets (-1)
        row_mean = {'epoch': ep, 'dataset_id': -1}
        for l in layer_names:
            row_mean[f'pc_{l}'] = pc_stats[primary_sample_size][l]['mean']
            row_mean[f'ipc_{l}'] = ipc_stats[primary_sample_size][l]['mean']
        csv_rows.append(row_mean)

        # 3) Error / uncertainty across datasets (-2)
        row_err = {'epoch': ep, 'dataset_id': -2}
        for l in layer_names:
            row_err[f'pc_{l}'] = pc_stats[primary_sample_size][l]['std']
            row_err[f'ipc_{l}'] = ipc_stats[primary_sample_size][l]['std']
        csv_rows.append(row_err)

        # Update CSV on disk (updated continuously)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=csv_fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        # Track PC and IPC for early stopping & model selection
        pc_means = [pc_stats[primary_sample_size][l]['mean'] for l in layer_names]
        ipc_means = [ipc_stats[primary_sample_size][l]['mean'] for l in layer_names]
        current_pc = max(pc_means)
        current_ipc = max(ipc_means)

        if current_pc > best_pc_value:
            best_pc_value = current_pc
            best_pc_epoch = ep

        if current_ipc > best_ipc_value:
            best_ipc_value = current_ipc
            best_ipc_epoch = ep

        if s_val_loss < best_s_val_loss:
            best_s_val_loss = s_val_loss
            best_val_loss_epoch = ep

        # 1.1 Check early stopping conditions (AND logic across 3 conditions)
        cond1 = (ep - best_val_loss_epoch) >= early_stop_patience
        cond2 = (ep - best_pc_epoch) >= early_stop_patience
        cond3 = (ep - best_ipc_epoch) >= early_stop_patience
        all_conditions_met = cond1 and cond2 and cond3

        # Save model state dict for this epoch
        epoch_states.append({k: v.cpu().clone() for k, v in source_model.state_dict().items()})

        # Append to history
        prefix_tag = "[Control]" if early_stop_triggered else "[Pretrain]"
        print(f"{prefix_tag} Epoch {ep:02d}/{pretrain_epochs:02d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Source Test Acc: {s_test_acc * 100:.1f}% | "
              f"Source Val Loss: {s_val_loss:.4f} (best: {best_s_val_loss:.4f} @ ep {best_val_loss_epoch}) | "
              f"PC: {current_pc:.4f} (best: {best_pc_value:.4f} @ ep {best_pc_epoch}) | "
              f"IPC: {current_ipc:.4f} (best: {best_ipc_value:.4f} @ ep {best_ipc_epoch}) | "
              f"Patience: ValLoss={ep - best_val_loss_epoch}/{early_stop_patience}, "
              f"PC={ep - best_pc_epoch}/{early_stop_patience}, "
              f"IPC={ep - best_ipc_epoch}/{early_stop_patience}")

        stage1_history.append({
            'epoch': ep,
            'train_loss': float(avg_train_loss),
            'source_val_loss': float(s_val_loss),
            'source_test_acc': float(s_test_acc * 100.0),
            'target_val_loss': float(t_val_loss),
            'current_pc': float(current_pc),
            'current_ipc': float(current_ipc),
            'best_s_val_loss': float(best_s_val_loss),
            'best_val_loss_epoch': int(best_val_loss_epoch),
            'best_pc_value': float(best_pc_value),
            'best_pc_epoch': int(best_pc_epoch),
            'best_ipc_value': float(best_ipc_value),
            'best_ipc_epoch': int(best_ipc_epoch),
            'cond1_val_loss_met': bool(cond1),
            'cond2_pc_met': bool(cond2),
            'cond3_ipc_met': bool(cond3),
            'all_conditions_met': bool(all_conditions_met),
            'pc_profiles': pc_stats,
            'ipc_profiles': ipc_stats,
            'is_control_phase': bool(early_stop_triggered)
        })

        # 1.3 Save model metrics history to disk on every epoch
        history_save_path = os.path.join(log_dir, "stage1_model_history.json")
        with open(history_save_path, 'w', encoding='utf-8') as f_hist:
            json.dump({
                'stage': 'stage1_pretraining',
                'experiment_case': experiment_case,
                'top_level_function': top_level_func_name,
                'short_hash': short_hash,
                'early_stop_epoch': early_stop_epoch,
                'early_stop_triggered': early_stop_triggered,
                'post_early_stop_epochs': post_early_stop_epochs,
                'best_s_val_loss': float(best_s_val_loss),
                'best_val_loss_epoch': int(best_val_loss_epoch),
                'best_pc_value': float(best_pc_value),
                'best_pc_epoch': int(best_pc_epoch),
                'best_ipc_value': float(best_ipc_value),
                'best_ipc_epoch': int(best_ipc_epoch),
                'best_model_criterion': best_model_criterion,
                'epochs_trained': len(stage1_history),
                'history': stage1_history
            }, f_hist, indent=2)

        # 1.1 Early stopping trigger check: all 3 conditions must be satisfied
        if not early_stop_triggered:
            if all_conditions_met:
                early_stop_triggered = True
                early_stop_epoch = ep
                print(f"\\n>>> [Early Stop Triggered] at epoch {ep}: all 3 conditions met (AND logic).")
                print(f"    1) Min Val Loss + {early_stop_patience} epochs: met ({ep - best_val_loss_epoch} epochs since min {best_s_val_loss:.4f} at epoch {best_val_loss_epoch})")
                print(f"    2) Max PC + {early_stop_patience} epochs: met ({ep - best_pc_epoch} epochs since max {best_pc_value:.4f} at epoch {best_pc_epoch})")
                print(f"    3) Max IPC + {early_stop_patience} epochs: met ({ep - best_ipc_epoch} epochs since max {best_ipc_value:.4f} at epoch {best_ipc_epoch})")
                if post_early_stop_epochs > 0:
                    print(f">>> Running {post_early_stop_epochs} additional control epochs...\\n")
                else:
                    print(">>> Stopping Stage 1 pretraining.\\n")
                    break
        else:
            extra_epochs_left -= 1
            if extra_epochs_left <= 0:
                print(f"\\n>>> Completed {post_early_stop_epochs} post-early-stop control epochs. Stopping Stage 1.\\n")
                break

    # === Model selection based on best_model_criterion ===
    if best_model_criterion == 'source_val_loss':
        if best_val_loss_epoch is not None and best_val_loss_epoch <= len(epoch_states):
            best_source_state = epoch_states[best_val_loss_epoch - 1]
        print(f"Best model selected by Source Val Loss (epoch {best_val_loss_epoch}, min loss={best_s_val_loss:.4f})")
    elif best_model_criterion == 'pc':
        if best_pc_epoch is not None and best_pc_epoch <= len(epoch_states):
            best_source_state = epoch_states[best_pc_epoch - 1]
        print(f"Best model selected by PC (epoch {best_pc_epoch}, max PC={best_pc_value:.4f})")
    elif best_model_criterion == 'ipc':
        if best_ipc_epoch is not None and best_ipc_epoch <= len(epoch_states):
            best_source_state = epoch_states[best_ipc_epoch - 1]
        print(f"Best model selected by IPC (epoch {best_ipc_epoch}, max IPC={best_ipc_value:.4f})")
    elif best_model_criterion == 'frankenstein':
        pc_per_epoch = []
        for h in stage1_history:
            pc_per_epoch.append(h['pc_profiles'][primary_sample_size])

        selected_epochs = {}
        selected_epochs['conv1'] = max(range(len(epoch_states)), key=lambda e: pc_per_epoch[e]['conv1']['mean'])
        for j in range(1, len(layer_names)):
            prev_layer = layer_names[j-1]
            curr_layer = layer_names[j]
            selected_epochs[curr_layer] = max(range(len(epoch_states)), key=lambda e: abs(pc_per_epoch[e][prev_layer]['mean'] - pc_per_epoch[e][curr_layer]['mean']))

        frankenstein_state_dict = {}
        for j, layer_name in enumerate(layer_names):
            src_epoch = selected_epochs[layer_name]
            prefixes = (f'layers.{j}.', f'{layer_name}.')
            for key, value in epoch_states[src_epoch].items():
                if key.startswith(prefixes):
                    frankenstein_state_dict[key] = value.clone()
        for key, value in epoch_states[-1].items():
            if key.startswith('classifier'):
                frankenstein_state_dict[key] = value.clone()

        best_source_state = frankenstein_state_dict
        print(f"Frankenstein model assembled: layer epochs = {selected_epochs}")
    else:
        raise ValueError(f"Unknown best_model_criterion: {best_model_criterion}")

    # 1.3 Save model checkpoint to log_dir
    model_save_path = os.path.join(log_dir, "pretrained_model.pt")
    torch.save({
        'best_state_dict': best_source_state,
        'final_state_dict': source_model.state_dict(),
        'early_stop_epoch': early_stop_epoch,
        'best_s_val_loss': float(best_s_val_loss),
        'best_val_loss_epoch': int(best_val_loss_epoch),
        'best_pc_value': float(best_pc_value),
        'best_pc_epoch': int(best_pc_epoch),
        'best_ipc_value': float(best_ipc_value),
        'best_ipc_epoch': int(best_ipc_epoch),
        'best_model_criterion': best_model_criterion,
    }, model_save_path)
    print(f"Pretrained model checkpoint saved to: {model_save_path}")

    stage1_ts = get_timestamp()
    print(f"Stage 1 artifacts saved in: {log_dir}")
    print(f"  - Model checkpoint: {model_save_path}")
    print(f"  - Metrics history : {history_save_path}")
    print(f"  - Launch params   : {params_txt_path}")
    print(f"  - PC/IPC CSV table: {csv_path}")

    return {
        'source_model': source_model,
        'best_source_state': best_source_state,
        'stage1_history': stage1_history,
        'early_stop_epoch': early_stop_epoch,
        'stage1_ts': stage1_ts,
        'best_s_val_loss': best_s_val_loss,
        'best_val_loss_epoch': best_val_loss_epoch,
        'best_pc_value': best_pc_value,
        'best_pc_epoch': best_pc_epoch,
        'best_ipc_value': best_ipc_value,
        'best_ipc_epoch': best_ipc_epoch,
        'model_save_path': model_save_path,
        'history_save_path': history_save_path,
        'log_dir': log_dir,
        'csv_path': csv_path,
        'params_path': params_txt_path,
        'short_hash': short_hash,
        'top_level_func_name': top_level_func_name,
    }
'''

nb['cells'][11]['source'] = new_cell11_src.splitlines(keepends=True)
print("Cell 11 updated.")

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Saved notebook:", NB_PATH)

with open(CELL11_PATH, 'w', encoding='utf-8') as f:
    f.write(new_cell11_src)
print("Saved cell_11.py:", CELL11_PATH)
