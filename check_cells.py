import json

path = 'd:/googledisk6/user/goit/Diplom/var1_Di/code/Di/pretrained500.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Print cell 10
print('===== CELL 10 =====')
src10 = ''.join(nb['cells'][10].get('source', []))
print(src10[:3000])
print()

# Print cell 12
print('===== CELL 12 =====')
src12 = ''.join(nb['cells'][12].get('source', []))
print(src12[:3000])
print()

# Print cell 15
print('===== CELL 15 =====')
src15 = ''.join(nb['cells'][15].get('source', []))
print(src15[:3000])
