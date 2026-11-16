def select_fixed_subsets(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    sample_sizes: List[int],
    num_replications: int,
    base_seed: int = 42
) -> Dict[int, List[Dict[str, Any]]]:
    """Pre-sample and fix deterministic subsets for PC and target validation."""
    n_total = x_data.shape[0]
    subsets_by_size = {}

    for n_s in sample_sizes:
        subsets_by_size[n_s] = []
        eff_n = min(n_s, n_total)
        for rep in range(num_replications):
            seed = base_seed + rep * 1000 + n_s
            gen = torch.Generator().manual_seed(seed)
            indices = torch.randperm(n_total, generator=gen)[:eff_n]
            sub_x, sub_y = x_data[indices], y_data[indices]
            sub_loader = DataLoader(
                TensorDataset(sub_x, sub_y),
                batch_size=64,
                shuffle=False
            )
            subsets_by_size[n_s].append({
                'rep': rep,
                'seed': seed,
                'indices': indices,
                'x': sub_x,
                'y': sub_y,
                'loader': sub_loader
            })
    return subsets_by_size

# Pre-select and fix target subsets
fixed_target_subsets = select_fixed_subsets(
    target_train_x,
    target_train_y,
    sample_sizes=SAMPLE_SIZES,
    num_replications=NUM_REPLICATIONS,
    base_seed=42
)

print(f"Fixed {len(fixed_target_subsets)} sample size configurations with {NUM_REPLICATIONS} replications each.")

def evaluate_layer_pc_profiles(
    model: nn.Module,
    subsets_dict: Dict[int, List[Dict[str, Any]]],
    device: torch.device,
    pooling: str = 'gap',
    Debug_Info: int = 0,
    epoch: int = 0,
    output_dir: str = "debug_results",
    one_hot: bool = False,
    num_classes: Optional[int] = None
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Compute layer-wise PC on the pre-fixed subsets across replications."""
    model.eval()
    layer_names = CIFAR6CNN.LAYER_NAMES
    ipc_summary_by_size = {}
    pc_summary_by_size = {}
    layer_ips = {l: [] for l in layer_names}
    for n_s, rep_list in subsets_dict.items():
        layer_rep_values = {l: [] for l in layer_names}
        for item in rep_list:
            acts, labels = extract_layer_activations(
                model=model,
                dataloader=item['loader'],
                layer_names=layer_names,
                pooling=pooling,
                device=device,
                max_samples=n_s
            )
            for l in layer_names:
                if one_hot and num_classes is not None:
                    labels_one_hot = torch.zeros(len(labels), num_classes, device=device)
                    labels_one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
                    labels_for_pc = labels_one_hot
                else:
                    labels_for_pc = labels
                pc_val = projection_correlation(acts[l], labels_for_pc, mode='pcor')
                ipc_val = projection_correlation(acts[l], labels_for_pc, mode='ipcor')
                layer_rep_values[l].append(float(pc_val))
                layer_ips[l].append(float(ipc_val))
                if Debug_Info == 1:
                    experiment_name = f"CNN6_{epoch}_{l}"
                    save_experiment_results(
                        experiment_name=experiment_name,
                        X=acts[l],
                        Y=labels,
                        pc={'PC': float(pc_val)},
                        output_dir=output_dir,
                    )
        stats = {}
        for l in layer_names:
            arr = np.array(layer_rep_values[l])
            stats[l] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'values': [float(v) for v in arr]
            }
        pc_summary_by_size[n_s] = stats
        ipc_stats = {}
        for l in layer_names:
            ipc_arr = np.array(layer_ips[l])
            ipc_stats[l] = {
                'mean': float(np.mean(ipc_arr)),
                'std': float(np.std(ipc_arr)),
                'values': [float(v) for v in ipc_arr]
            }
        ipc_summary_by_size[n_s] = ipc_stats
    return pc_summary_by_size, ipc_summary_by_size

def format_pc_summary(pc_stats: Dict[int, Dict[str, Dict[str, float]]], ipc_stats: Optional[Dict[int, Dict[str, Dict[str, float]]]] = None, target_size: int = None) -> str:
    """Format PC stats as a readable string for logging."""
    if target_size is None:
        target_size = list(pc_stats.keys())[0] if pc_stats else 0
    stats = pc_stats.get(target_size, {})
    parts = []
    for l in CIFAR6CNN.LAYER_NAMES:
        if l in stats:
            parts.append(f"{l}={stats[l]['mean']:.3f}\u00b1{stats[l]['std']:.3f}")
    if ipc_stats is not None:
        ipc = ipc_stats.get(target_size, {})
        ipc_parts = []
        for l in CIFAR6CNN.LAYER_NAMES:
            if l in ipc:
                ipc_parts.append(f"{l}={ipc[l]['mean']:.3f}\u00b1{ipc[l]['std']:.3f}")
        if ipc_parts:
            parts.append("IPC: " + " | ".join(ipc_parts))
    return " | ".join(parts)
