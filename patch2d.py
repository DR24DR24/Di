import json

path = 'd:/googledisk6/user/goit/Diplom/var1_Di/code/Di/pretrained500.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# CELL 16: Update markdown to mention IPC
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'Layer-wise PC Trajectories' in src and 'IPC' not in src:
        old = "3. **Layer-wise PC Trajectories for all 7 cases**:"
        new = """3. **Layer-wise PC and IPC Trajectories for all 7 cases**:
   - Evolution of Projection Correlation (PC) and Improved Projection Correlation (IPC) for all layers (`conv1`..`conv6`) during target fine-tuning for each model."""
        if old in src:
            cell['source'] = src.replace(old, new).split('\n')
            print('Cell 16: updated markdown to mention IPC')
            break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Patch 2d done (cell 16 markdown)')
