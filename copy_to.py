import json
import os

path = 'd:/googledisk6/user/goit/Diplom/var1_Di/code/Di/pretrained500.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Create a combined copy of code cells
combined = []
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if cell['cell_type'] == 'code' and len(src) > 100:
        combined.append(f"===== CELL {i} (code) =====")
        combined.append(src)
        combined.append("")

with open('d:/googledisk6/user/goit/Diplom/var1_Di/code/Di/all_code.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(combined))

print("Created all_code.txt")
