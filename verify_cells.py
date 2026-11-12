import json

path = 'd:/googledisk6/user/goit/Diplom/var1_Di/code/Di/pretrained500.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Verify cells 9, 10, 12, 15
for i in [9, 10, 12, 15]:
    print(f'===== CELL {i} =====')
    src = ''.join(nb['cells'][i].get('source', []))
    print(src[:2500])
    print()
    print('-----')
    print()
