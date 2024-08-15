import csv
import json
import sys
from pathlib import Path

TASK_TO_KS = {
    'cola': (['eval_matthews_correlation'], 100.0),
    'sst2': (['eval_accuracy'], 100.0),
    'mrpc': (['eval_f1'], 100.0),
    'stsb': (['eval_spearmanr'], 100.0),
    'qqp': (['eval_f1'], 100.0),
    'mnli': (['eval_accuracy'], 100.0),  # 'eval_accuracy_mm'
    'qnli': (['eval_accuracy'], 100.0),
    'rte': (['eval_accuracy'], 100.0),
    # 'wnli': (['eval_accuracy'], 100.0)
}


def is_glue_dir(p):
    return p.joinpath('mnli').is_dir() or p.joinpath('mnli.json').is_file()


output_dir = sys.argv[1]
sub_dirs = [p for p in Path(output_dir).glob('*') if p.is_dir() and is_glue_dir(p)]
if len(sub_dirs) < 1:
    sub_dirs = [Path(output_dir)]
    if not is_glue_dir(sub_dirs[0]):
        print('Miss results, exit')
        exit()

table = list()

for p in sub_dirs:
    name = p.stem
    row = list()
    for t, (ks, facotr) in TASK_TO_KS.items():
        try:
            rp = Path(p, t, 'all_results.json')
            if not rp.exists():
                rp = Path(p, t + '.json')
            with open(rp) as f:
                result = json.load(f)
        except FileNotFoundError as e:
            print(e)
            result = {k: 0 for k in ks}
        for k in ks:
            row.append(result[k] * facotr)
    avg = sum(row) / len(row)
    row = [name, avg] + row
    print(p, avg)
    table.append(row)

with open(Path(output_dir, 'glue.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerows(table)
    print('write to', f.name)

"""
python tools/gather_glue.py results/baiy-n29p25
"""
