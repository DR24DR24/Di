p = '_translate_ru.py'
with open(p, encoding='utf-8') as f:
    lines = f.readlines()
# Fix line 42 (0-indexed 41) which has escaped apostrophe issue
lines[41] = (
    "repls.append(('Она способна обнаруживать любые нелинейные зависимости "
    "(в отличие от линейной корреляции Пирсона', "
    "\"It can detect any nonlinear dependencies (unlike Pearson's linear correlation)\"))\n"
)
with open(p, 'w', encoding='utf-8') as f:
    f.writelines(lines)
print('Fixed line 42')
