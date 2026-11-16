import json
with open(r'd:/googledisk8J/git/Di/pretrained_stat.ipynb', 'r', encoding='utf-8-sig') as f:
    nb = json.load(f)
for i in range(len(nb['cells'])):
    cell = nb['cells'][i]
    src = ''.join(cell.get('source', []))
    print('Cell {} (exec={}): len={}'.format(i, cell.get('execution_count'), len(src)))
    if src:
        print('  First 100 chars: ' + src[:100].replace('\n', ' '))