from collections import defaultdict
import csv
import json
import sys
from pathlib import Path


TASK_TO_KS = {
    'xnli': (['predict_accuracy'], 100.0, 'ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh'),
    'paws-x': (['predict_accuracy'], 100.0, 'de,en,es,fr,ja,ko,zh'),
    'udpos': (['predict_f1'], 100.0, {
        "udpos.Afrikaans": "af",
        "udpos.Arabic": "ar",
        "udpos.Basque": "eu",
        "udpos.Bulgarian": "bg",
        "udpos.Dutch": "nl",
        "udpos.English": "en",
        "udpos.Estonian": "et",
        "udpos.Finnish": "fi",
        "udpos.French": "fr",
        "udpos.German": "de",
        "udpos.Greek": "el",
        "udpos.Hebrew": "he",
        "udpos.Hindi": "hi",
        "udpos.Hungarian": "hu",
        "udpos.Indonesian": "id",
        "udpos.Italian": "it",
        "udpos.Japanese": "ja",
        "udpos.Kazakh": "kk",
        "udpos.Korean": "ko",
        "udpos.Chinese": "zh",
        "udpos.Marathi": "mr",
        "udpos.Persian": "fa",
        "udpos.Portuguese": "pt",
        "udpos.Russian": "ru",
        "udpos.Spanish": "es",
        "udpos.Tagalog": "tl",
        "udpos.Tamil": "ta",
        "udpos.Telugu": "te",
        "udpos.Thai": "th",
        "udpos.Turkish": "tr",
        "udpos.Urdu": "ur",
        "udpos.Vietnamese": "vi",
        "udpos.Yoruba": "yo"
        }),
    'wikiann': (
        ['predict_f1'], 100.0,
        "en,af,ar,bg,bn,de,el,es,et,eu,fa,fi,fr,he,hi,hu,id,it,ja,jv,"
        "ka,kk,ko,ml,mr,ms,my,nl,pt,ru,sw,ta,te,th,tl,tr,ur,vi,yo,zh"
    ),
    'xquad': (['test_f1', 'test_exact_match'], 1.0,{
        "xquad.ar": 'ar', "xquad.de": 'de', "xquad.el": 'el', "xquad.en": 'en', "xquad.es": 'es', "xquad.ru": 'ru',
        "xquad.th": 'th', "xquad.tr": 'tr', "xquad.vi": 'vi', "xquad.zh": 'zh', "xquad.ro": 'ro'
    }),
    'mlqa': (['test_f1', 'test_exact_match'], 1.0, {
        'mlqa.ar.ar': 'ar', 'mlqa.de.de': 'de', 'mlqa.en.en': 'en', 'mlqa.es.es': 'es', 'mlqa.hi.hi': 'hi',
        'mlqa.vi.vi': 'vi', 'mlqa.zh.zh': 'zh'
    }),
    'tydiqa': (['test_f1', 'test_exact_match'], 1.0, 'en,ar,bn,fi,id,ko,ru,sw,te'),
    'bucc18_first_token': (['predict_f1'], 100.0, {
        'bucc18.de': 'de','bucc18.fr': 'fr','bucc18.ru': 'ru', 'bucc18.zh': 'zh'
    }),
    'tatoeba_first_token': (['predict_accuracy'], 100.0, {  # ',aze-eng,lit-eng,pol-eng,ukr-eng,ron-eng'
        "ara-eng": "ar", "heb-eng": "he", "vie-eng": "vi", "ind-eng": "id", "jav-eng": "jv", "tgl-eng": "tl",
        "eus-eng": "eu", "mal-eng": "ml", "tam-eng": "ta", "tel-eng": "te", "afr-eng": "af", "nld-eng": "nl",
        "deu-eng": "de", "ell-eng": "el", "ben-eng": "bn", "hin-eng": "hi", "mar-eng": "mr", "urd-eng": "ur",
        "fra-eng": "fr", "ita-eng": "it", "por-eng": "pt", "spa-eng": "es", "bul-eng": "bg", "rus-eng": "ru",
        "jpn-eng": "ja", "kat-eng": "ka", "kor-eng": "ko", "tha-eng": "th", "swh-eng": "sw", "cmn-eng": "zh",
        "kaz-eng": "kk", "tur-eng": "tr", "est-eng": "et", "fin-eng": "fi", "hun-eng": "hu", "pes-eng": "fa",
        # "aze-eng": "az", "lit-eng": "lt", "pol-eng": "pl", "ukr-eng": "uk", "ron-eng": "ro"
    }),
    ### XTREME-R
    'xcopa': (['predict_accuracy'], 100.0, 'et,ht,id,it,qu,sw,ta,th,tr,vi,zh'),
    'udpos_v27': (
        ['predict_f1'], 100.0,
        'af,ar,bg,de,el,en,es,et,eu,fa,fi,fr,he,hi,hu,id,it,ja,kk,ko,lt,mr,nl,pl,pt,ro,ru,ta,te,th,tl,tr,uk,ur,vi,wo,yo,zh'
    ),
    'wikiann-47': (
        ['predict_f1'], 100.0,
        "ar,he,vi,id,jv,ms,tl,eu,ml,ta,te,af,nl,en,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,"
        "ka,ko,th,sw,yo,my,zh,kk,tr,et,fi,hu,qu,pl,uk,az,lt,pa,gu,ro"
    ),
    'tatoeba-xtreme-r': (['predict_accuracy'], 100.0, {
        "ara-eng": "ar", "heb-eng": "he", "vie-eng": "vi", "ind-eng": "id", "jav-eng": "jv", "tgl-eng": "tl",
        "eus-eng": "eu", "mal-eng": "ml", "tam-eng": "ta", "tel-eng": "te", "afr-eng": "af", "nld-eng": "nl",
        "deu-eng": "de", "ell-eng": "el", "ben-eng": "bn", "hin-eng": "hi", "mar-eng": "mr", "urd-eng": "ur",
        "fra-eng": "fr", "ita-eng": "it", "por-eng": "pt", "spa-eng": "es", "bul-eng": "bg", "rus-eng": "ru",
        "jpn-eng": "ja", "kat-eng": "ka", "kor-eng": "ko", "tha-eng": "th", "swh-eng": "sw", "cmn-eng": "zh",
        "kaz-eng": "kk", "tur-eng": "tr", "est-eng": "et", "fin-eng": "fi", "hun-eng": "hu", "pes-eng": "fa",
        "aze-eng": "az", "lit-eng": "lt", "pol-eng": "pl", "ukr-eng": "uk", "ron-eng": "ro"
    }),
    'mewslix': (['map_at_20'], 100.0, 'ar,de,en,es,fa,ja,pl,ro,ta,tr,uk'),
    'lareqa-xquadr': (['map_at_20'], 100.0, 'ar,de,el,en,es,hi,ru,th,tr,vi,zh'),
}


def is_xtreme_dir(p):
    return p.joinpath('xnli').is_dir() or p.joinpath('xnli.json').is_file()


baseline_paths = [
    'results/lm/google-bert/bert-base-multilingual-cased',
    'results/lm/FacebookAI/xlm-roberta-base',
]

if len(sys.argv) > 1:
    target_path = sys.argv[1]
    sub_dirs = [p for p in Path(target_path).glob('*') if p.is_dir() and is_xtreme_dir(p)]
    if len(sub_dirs) < 1:
        sub_dirs = [Path(target_path)]
        if not is_xtreme_dir(sub_dirs[0]):
            print('Miss results, exit')
            exit()
else:
    target_path = 'results/'
    sub_dirs = [Path(p) for p in baseline_paths]


ts_xtreme = [
    'xnli', 'paws-x', 'udpos', 'wikiann', 'xquad', 'mlqa', 'tydiqa', 'bucc18_first_token', 'tatoeba_first_token'
]
ts_xtremer = [
    'xnli', 'xcopa', 'udpos_v27', 'wikiann-47', 'xquad', 'mlqa', 'tydiqa', 'tatoeba-xtreme-r', 'mewslix', 'lareqa-xquadr'
]

table = [[]]
all_details = dict()

for p in sub_dirs:
    model_name = p.stem
    head = list()
    row = list()
    details = defaultdict(dict)
    for t, (ks, facotr, names) in TASK_TO_KS.items():
        if isinstance(names, str):
            names = {n: n for n in names.split(',')}
        try:
            _t = t if t != 'wikiann-47' else 'wikiann'
            rp = Path(p, _t, 'all_results.json')
            if not rp.exists():
                rp = Path(p, _t + '.json')
            with open(rp) as f:
                try:
                    all_metrics = json.load(f)['all_metrics']
                except KeyError:
                    print(t, p)
                    raise
            result = {k: sum(all_metrics[n][k] for n in names) / len(names) for k in ks}
        except FileNotFoundError as e:
            print(e)
            result = {k: 0 for k in ks}
        for k in ks:
            row.append(result[k] * facotr)
            head.append(f'{t}-{k}')
        dk = f'{t}-{ks[0]}'
        for n, l in names.items():
            details[dk][l] = ((all_metrics[n][ks[0]] if ks[0] in all_metrics[n] else 0) * facotr) if n in all_metrics else 0
    all_details[model_name] = details
    k_to_idx = {k: i for i, k in enumerate(head)}
    avg = sum(row[k_to_idx[f'{t}-{TASK_TO_KS[t][0][0]}']] for t in ts_xtreme) / len(ts_xtreme)
    avg_r = sum(row[k_to_idx[f'{t}-{TASK_TO_KS[t][0][0]}']] for t in ts_xtremer) / len(ts_xtremer)
    head = ['model', 'XTREME Avg.'] + head[:12] + ['XTREME-R Avg.'] + head[12:]
    row = [model_name, avg] + row[:12] + [avg_r] + row[12:]
    print(p, head[1], row[1], head[14], row[14])
    table.append(row)
    table[0] = head

all_langs = [['model'] + sorted(set(sum([list(m.keys()) for m in details.values()], [])))]
for model, details in all_details.items():
    by_langs = defaultdict(dict)
    for task, metrics in details.items():
        for lang, score in metrics.items():
            by_langs[lang][task] = score
    all_langs.append([model] + [sum(by_langs[lang].values()) / len(by_langs[lang]) for lang in all_langs[0][1:]])

detail_by_langs = []
for task, metrics in list(all_details[model_name].items()):
    task_head = [task] + list(metrics)
    detail_by_langs.append(task_head)
    for model, d in all_details.items():
        detail_by_langs.append([model] + [d[task][l] for l in task_head[1:]])
    detail_by_langs.append(['',])

table = table + ['', ] + all_langs + ['',] + detail_by_langs

with open(Path(target_path, 'xtreme.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerows(table)
    print('write to', f.name)

"""
python tools/gather_xtreme.py xxxx
"""
