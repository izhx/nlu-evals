import logging
import os
import time

os.environ['HF_HOME'] = './.cache_hf'
# os.environ['ALL_PROXY'] = 'http://localhost:1081'

import datasets

datasets.utils.logging.set_verbosity(logging.INFO)
for handler in logging.getLogger('datasets').handlers:
    formatter = logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s")
    handler.setFormatter(formatter)

# # Setup logging
# logging.basicConfig(
#     format="%(asctime)s|%(levelname)s|%(name)s:%(lineno)s|=> %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     handlers=[logging.StreamHandler(sys.stdout)],
# )
# logger = logging.getLogger('cache_data.py')
# logger.setLevel(logging.INFO)

def printf(*args):
    print(time.asctime(), '|', *args)


def load_by_names(path: str, names=None):
    printf('path:', path)
    if names is None or len(names)< 1:
        names = datasets.get_dataset_config_names(path)
    for name in names:
        printf(name, '->', datasets.load_dataset(path, name))


load_by_names('nyu-mll/glue')

load_by_names('facebook/xnli', 'ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh'.split(','))
# no all_languages

load_by_names('cambridgeltl/xcopa', 'et,ht,id,it,qu,sw,ta,th,tr,vi,zh'.split(','))
# no translation-*

load_by_names('aps/super_glue', ['copa'])

printf(datasets.load_dataset('allenai/social_i_qa'))

load_by_names('izhx/xtreme-r-udpos')

load_by_names(
    'unimelb-nlp/wikiann',
    "ar,he,vi,id,jv,ms,tl,eu,ml,ta,te,af,nl,en,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,"
    "bg,ru,ja,ka,ko,th,sw,yo,my,zh,kk,tr,et,fi,hu,qu,pl,uk,az,lt,pa,gu,ro".split(',')
)

printf(datasets.load_dataset('rajpurkar/squad'))

load_by_names('google/xquad', ['xquad.' + l for l in 'ar,de,el,en,es,hi,ru,th,tr,vi,zh,ro'.split(',')])

load_by_names('facebook/mlqa')  # [f'mlqa.{l}.{l}' for l in 'en es de ar hi vi zh'.split(' ')]

load_by_names('juletxara/tydiqa_xtreme')

load_by_names('izhx/mewsli-x')

load_by_names('google-research-datasets/xquad_r')

load_by_names('mteb/tatoeba-bitext-mining')

###########  XTREME

load_by_names('google/xtreme', ['bucc18.' + l for l in 'de,fr,ru,zh'.split(',')])

# ud v2.5 pos, 33 languages
xtreme_names = datasets.get_dataset_config_names('google/xtreme')
load_by_names('google/xtreme', [n for n in xtreme_names if n.startswith('udpos')])

load_by_names('google-research-datasets/paws-x')


print('ok')
