import json

path = 'd:/googledisk6/user/goit/Diplom/var1_Di/code/Di/pretrained500.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Print cell 9
print('=== CELL 9 ===')
print(''.join(nb['cells'][9].get('source', [])))
print()

# Print cell 10 (first 500 chars)
print('=== CELL 10 (first 500) ===')
print(''.join(nb['cells'][10].get('source', []))[:500])
print()
