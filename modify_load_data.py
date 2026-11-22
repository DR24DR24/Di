import json

nb_path = r"d:\googledisk6\user\goit\Diplom\var1_Di\code\Di2\pretrained_stat.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# --- 2. Modify call site in cell 43 ---
for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        text = "".join(cell["source"])
        if "data = load_experiment_data(" in text:
            # The JSON has escaped quotes, so we match with backslash-quotes
            old_call = 'data = load_experiment_data(\\n    experiment_case=params[\\\"experiment_case\\\"],\\n    seed=params[\\\"seed\\\"],\\n    use_real_cifar10=params[\\\"use_real_cifar10\\\"],\\n)'
            new_call = 'data = load_experiment_data(params)'
            
            if old_call in text:
                text = text.replace(old_call, new_call, 1)
                cell["source"] = text.splitlines(True)
                print("Call site: load_experiment_data(params)")
            else:
                print("Call site: old call not found with escaped quotes")
                # Try matching without escapes
                old_call2 = 'data = load_experiment_data(\n    experiment_case=params["experiment_case"],\n    seed=params["seed"],\n    use_real_cifar10=params["use_real_cifar10"],\n)'
                if old_call2 in text:
                    text = text.replace(old_call2, new_call, 1)
                    cell["source"] = text.splitlines(True)
                    print("Call site: load_experiment_data(params) (matched without escapes)")
                else:
                    idx = text.find("data = load_experiment_data")
                    if idx >= 0:
                        print("Found at index", idx)
                        print(repr(text[idx:idx+250]))
            break

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Done.")
