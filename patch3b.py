import json

path = 'd:/googledisk6/user/goit/Diplom/var1_Di/code/Di/pretrained500.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# CELL 15: Stage 2 logging - update format_pc_summary call to include ipc_stats
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if i == 15 and 'pc_str = format_pc_summary' in src:
        old = "pc_str = format_pc_summary(pc_stats, target_size=primary_sample_size)"
        new = "pc_str = format_pc_summary(pc_stats, ipc_stats, target_size=primary_sample_size)"
        if old in src:
            cell['source'] = src.replace(old, new).split('\n')
            print('Cell 15: pc_str formatting updated')
            break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Patch 3b (cell 15) done')
