import json

path = 'd:/googledisk6/user/goit/Diplom/var1_Di/code/Di/pretrained500.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Print cell 17
print('===== CELL 17 =====')
src = ''.join(nb['cells'][17].get('source', []))
print(src)
