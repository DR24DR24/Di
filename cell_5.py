# Experiment Mode and Subsets
EXPERIMENT_CASE = 'imbalance'   # 'imbalance' (5 animal -> 5 vehicle classes) or 'balance' (50%/50% split)
SAMPLE_SIZES = [200]      # Target sample sizes for PC tracking and fine-tuning
NUM_REPLICATIONS = 10            # Number of replications for PC confidence interval estimation
USE_REAL_CIFAR10 = True         # Set True for official CIFAR-10; False for quick synthetic mode

# Stage 1: Pretraining Hyperparameters
PRETRAIN_EPOCHS = 50            # Maximum pretraining epochs
EARLY_STOP_PATIENCE = 5         # Patience for early stopping on source validation loss
POST_EARLY_STOP_EPOCHS = 1     # Named constant: extra control epochs after early stopping trigger
LEARNING_RATE_SOURCE = 0.001    # RMSProp learning rate for pretraining (Di et al. 2025 Sec 4.1)

# Stage 2: Target Fine-Tuning Hyperparameters
TARGET_EPOCHS = 50              # Maximum target adaptation epochs
TARGET_PATIENCE = 5             # Patience for target early stopping
LEARNING_RATE_TARGET = 0.001    # RMSProp learning rate for target fine-tuning
# Feature Extraction mode: strict freezing of first k layers
#   False  -> Fine-tuning (all layers trainable, first k initialized from source)
#   True   -> Feature Extraction (first k layers frozen: requires_grad=False)
FEATURE_EXTRACTION = True
# Размер выборки для дообучения (Fine-Tuning), независимо от размера для PC
FT_SAMPLE_SIZE = 1000   # None -> использовать primary_sample_size (SAMPLE_SIZES[0])

# Directory for experiment results (matches notebook name: 'pretrained')
NOTEBOOK_NAME = "pretrained"
RESULTS_DIR = os.path.join(os.getcwd(), NOTEBOOK_NAME)
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_timestamp() -> str:
    """Return a compact timestamp string YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def make_result_path(base_name: str, ext: str, ts: Optional[str] = None) -> str:
    """Generate a full output file path with timestamp in the results directory."""
    if ts is None:
        ts = get_timestamp()
    filename = f"{base_name}_{ts}.{ext}"
    return os.path.join(RESULTS_DIR, filename)

print(f"Results Directory: {RESULTS_DIR}")
print(f"Mode: {EXPERIMENT_CASE} | Sample sizes: {SAMPLE_SIZES} | Replications: {NUM_REPLICATIONS}")
print(f"Post early-stopping control epochs: {POST_EARLY_STOP_EPOCHS}")
