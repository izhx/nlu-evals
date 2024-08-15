## `nlu-evals`

Evaluate BERT-style encoder models based on `transformers` library, currently support:
 - [GLUE](https://aclanthology.org/W18-5446/)
 - [XTREME](https://proceedings.mlr.press/v119/hu20b.html) and [XTREME-R](https://aclanthology.org/2021.emnlp-main.802/)


### Environment

```shell
conda create -n nlu_eval python=3.11
conda activate nlu_eval

pip install 'transformers>=4.41.0' 'datasets<2.20.0,>=2.19.0' accelerate evaluate scikit-learn seqeval pytrec_eval
```

### Usage

Directly run each script under `scripts/`, and those with dependencies will be automatically executed in order.
After completion, use the script under `tools/` to obtain the result statistics.

Run all datasets:
```shell
# Run all subsets sequentially
for (( group = 0; group <= 5; group += 1 ))
do
    bash scripts/entry.sh Alibaba-NLP/gte-multilingual-mlm-base results/mgte-mlm-base $group
done

# Get result
python tools/gather_glue.py results/mgte-mlm-base
python tools/gather_xtreme.py results/mgte-mlm-base
```


### Offline Setting

Please cache all required datasets manually and then set environment variables:
1. `HF_HOME="./.cache_hf"`
2. `HF_DATASETS_OFFLINE="1"`

Then
```
HF_HOME="./.cache_hf" HF_DATASETS_OFFLINE="1" bash scripts/entry.sh Alibaba-NLP/gte-multilingual-mlm-base results/mgte-mlm-base 0
```


### Citation
If you use the code in this repo, please cite our paper \cite{zhang2024mgte} and the corresponding benchmarks.
```
@misc{zhang2024mgte,
  title={mGTE: Generalized Long-Context Text Representation and Reranking Models for Multilingual Text Retrieval}, 
  author={Xin Zhang and Yanzhao Zhang and Dingkun Long and Wen Xie and Ziqi Dai and Jialong Tang and Huan Lin and Baosong Yang and Pengjun Xie and Fei Huang and Meishan Zhang and Wenjie Li and Min Zhang},
  year={2024},
  eprint={2407.19669},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2407.19669}, 
}

@inproceedings{wang-etal-2018-glue,
    title = "{GLUE}: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding",
    author = "Wang, Alex  and
      Singh, Amanpreet  and
      Michael, Julian  and
      Hill, Felix  and
      Levy, Omer  and
      Bowman, Samuel",
    editor = "Linzen, Tal  and
      Chrupa{\l}a, Grzegorz  and
      Alishahi, Afra",
    booktitle = "Proceedings of the 2018 {EMNLP} Workshop {B}lackbox{NLP}: Analyzing and Interpreting Neural Networks for {NLP}",
    month = nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W18-5446",
    doi = "10.18653/v1/W18-5446",
    pages = "353--355",
}

@InProceedings{pmlr-v119-hu20b,
  title = 	 {{XTREME}: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalisation},
  author =       {Hu, Junjie and Ruder, Sebastian and Siddhant, Aditya and Neubig, Graham and Firat, Orhan and Johnson, Melvin},
  booktitle = 	 {Proceedings of the 37th International Conference on Machine Learning},
  pages = 	 {4411--4421},
  year = 	 {2020},
  editor = 	 {III, Hal Daumé and Singh, Aarti},
  volume = 	 {119},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {13--18 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v119/hu20b/hu20b.pdf},
  url = 	 {https://proceedings.mlr.press/v119/hu20b.html},
}

@inproceedings{ruder-etal-2021-xtreme,
    title = "{XTREME}-{R}: Towards More Challenging and Nuanced Multilingual Evaluation",
    author = "Ruder, Sebastian  and
      Constant, Noah  and
      Botha, Jan  and
      Siddhant, Aditya  and
      Firat, Orhan  and
      Fu, Jinlan  and
      Liu, Pengfei  and
      Hu, Junjie  and
      Garrette, Dan  and
      Neubig, Graham  and
      Johnson, Melvin",
    editor = "Moens, Marie-Francine  and
      Huang, Xuanjing  and
      Specia, Lucia  and
      Yih, Scott Wen-tau",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.802",
    doi = "10.18653/v1/2021.emnlp-main.802",
    pages = "10215--10245",
}
```
