from models import CIFAR6CNN, create_transferred_model, set_trainable_layers
from dataset_splits import (
    split_cifar_imbalance,
    split_cifar_balance,
    generate_synthetic_transfer_dataset,
    make_dataloaders,
    create_ft_dataloaders_from_subsets
)
from di_algorithm import default_eval_loss
from projection_correlation import projection_correlation, save_experiment_results
from layer_extractor import extract_layer_activations

if USE_REAL_CIFAR10:
    from cifar10_experiment import load_cifar10_raw
    print("Loading CIFAR-10 dataset...")
    raw_x, raw_y = load_cifar10_raw(data_dir='./data')
    if EXPERIMENT_CASE == 'imbalance':
        splits = split_cifar_imbalance(raw_x, raw_y, seed=SEED)
    else:
        splits = split_cifar_balance(raw_x, raw_y, seed=SEED)
else:
    print("Generating synthetic transfer dataset...")
    splits = generate_synthetic_transfer_dataset(
        n_source=1200, n_target=800,
        num_classes=5 if EXPERIMENT_CASE == 'imbalance' else 10,
        transition_layer=3 if EXPERIMENT_CASE == 'imbalance' else 4,
        seed=SEED
    )

generator = torch.Generator().manual_seed(SEED)
loaders = make_dataloaders(splits, batch_size=64, generator=generator)
target_train_x, target_train_y = splits['target_train']
target_test_loader = loaders['target_test']

print(f"Source train size: {len(splits['source_train'][0])} | test: {len(splits['source_test'][0])}")
print(f"Target train size: {len(target_train_x)} | test: {len(splits['target_test'][0])}")
print(f"Source classes: {splits['num_source_classes']} | Target classes: {splits['num_target_classes']}")
