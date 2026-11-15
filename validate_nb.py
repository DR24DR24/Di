import json
with open('d:/googledisk8J/git/Di/pretrained_fr.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
print('Notebook JSON is valid')
cell = [c for c in nb['cells'] if c.get('execution_count') == 20]
if cell:
    src = ''.join(cell[0]['source'])
    checks = [
        ('BEST_MODEL_CRITERION = \'frankenstein\'', 'config variable'),
        ('best_pc_epoch = None', 'pc tracking var'),
        ('best_ipc_epoch = None', 'ipc tracking var'),
        ('mean_pc = sum(pc_means) / len(pc_means)', 'mean PC calc'),
        ('mean_ipc = sum(ipc_means) / len(ipc_means)', 'mean IPC calc'),
        ("if BEST_MODEL_CRITERION == 'source_val_loss':", 'source_val_loss branch'),
        ("elif BEST_MODEL_CRITERION == 'pc':", 'pc branch'),
        ("elif BEST_MODEL_CRITERION == 'ipc':", 'ipc branch'),
        ("elif BEST_MODEL_CRITERION == 'frankenstein':", 'frankenstein branch'),
    ]
    for pat, desc in checks:
        print('  [{}] {}'.format(pat in src, desc))