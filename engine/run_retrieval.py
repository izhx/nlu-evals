#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Training dual encoder models.
"""

import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import torch
import datasets
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import ModelOutput
# from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.39.0.dev0")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    pooler_type: Optional[str] = field(
        default=None, metadata={"help": "Pooler name (first_token/last_token/mean) for the embeddings."}
    )
    loss_kwargs: Optional[str] = field(default=None, metadata={"help": "The kwargs to loss."})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: Optional[bool] = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_type: Optional[str] = field(default="bitext", metadata={"help": "bitext/retrieval."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column_names: Optional[str] = field(
        default=None, metadata={"help": "The column names of text to input."}
    )
    corpus_dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    corpus_split: Optional[str] = field(
        default=None, metadata={"help": "The split name of the dataset to use (via the datasets library)."}
    )
    corpus_dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    corpus_text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of the positive candidate text."}
    )
    test_config_names: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    positive_id_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of the positive candidate id."}
    )
    merge_test: bool = field(default=False, metadata={"help": "Merge queries and corpus of all test configs."})
    test_split: Optional[str] = field(
        default=None, metadata={"help": "The split name of the dataset to test (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    corpus_file: Optional[str] = field(default=None, metadata={"help": "The input corpus file (a text file)."})
    negatives_per_instance: int = field(
        default=0,
        metadata={"help": "num of negative texts per training instance."},
    )
    safe_negatives: bool = field(
        default=True, metadata={"help": "if True, up sample negatives if they are less than `negatives_per_instance`"}
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_query_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_doc_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_query_chunk_size: int = field(default=0)
    gc_doc_chunk_size: int = field(default=0)
    score_fct: str = field(default='cos_sim', metadata={"help": "dot/cos_sim"})
    top_k: int = field(default=100, metadata={"help": "Keep top k candidates"})
    num_batch_per_chunk: int = field(default=100, metadata={"help": "corpus_chunk for eval"})
    ignore_identical: bool = field(default=False, metadata={"help": "ignore same query_id and doc_id in eval"})

    def __post_init__(self):
        assert self.task_type in {'bitext', 'retrieval'}
        assert self.score_fct in {'cos_sim', 'dot'}
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training/validation file/test_file.")
        # else:
        #     if self.train_file is not None:
        #         extension = self.train_file.split(".")[-1]
        #         assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        #     if self.validation_file is not None:
        #         extension = self.validation_file.split(".")[-1]
        #         assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        #     if self.test_file is not None:
        #         extension = self.test_file.split(".")[-1]
        #         assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."


#### from https://github.com/izhx/uni-rep
import os
import random
from typing import Dict, List, Optional

from torch.utils.data import Dataset


class TrainDataset(Dataset):
    """
    We create a custom dataset that returns tuples (query, positive, *negatives)
    on-the-fly based on the information from the mined hard-negatives.

    Remember to shuffle mined hard-negatives when formatting data .

    instances: [{'query': q, 'positives': [xx, ...], 'negatives': [xxx, ...]}, ...]
    """

    def __init__(
        self,
        instances: List[Dict],
        neg_per_ins: int = 1,
        shuffle_positives: bool = True,
        shuffle_negatives: bool = True,
        safe_negatives: bool = True,  # if True, up sample negatives if they are less than neg_per_ins
    ):
        self.instances = instances
        self.neg_per_ins = neg_per_ins
        self.shuffle_positives = shuffle_positives
        self.shuffle_negatives = shuffle_negatives
        self.safe_negatives = safe_negatives
        self.trainer = None

    def __getitem__(self, item):
        ins = self.instances[item]

        epoch = int(self.trainer.state.epoch)
        _hashed_seed = hash(item + self.trainer.args.seed)

        positives = ins['positives']
        if self.shuffle_positives:
            pos_text = positives[(_hashed_seed + epoch) % len(positives)]
        else:
            # pos_psg = group_positives[0]  # tevatron
            pos_text = positives.pop(0)  # get one positive and move it to the end
            positives.append(pos_text)

        rand = random.Random(_hashed_seed)
        negatives = ins.get('negatives', list())
        if len(negatives) < self.neg_per_ins:
            if self.safe_negatives and len(negatives) > 0:
                neg_texts = rand.choices(negatives, k=self.neg_per_ins)
            else:
                neg_texts = [ _ for _ in negatives]
                indices = rand.sample(range(len(self.instances)), k=self.neg_per_ins * 2)
                for index in indices:
                    if index == item:
                        continue
                    other = self.instances[index]
                    candidates = other['positives'] + other.get('negatives', list())
                    neg_texts.append(candidates[(_hashed_seed + epoch) % len(candidates)])
                    if len(neg_texts) >= self.neg_per_ins:
                        break
        elif self.neg_per_ins < 1:
            neg_texts = []
        elif self.shuffle_negatives:
            _offset = epoch * self.neg_per_ins % len(negatives)
            neg_texts = negatives.copy()
            rand.shuffle(neg_texts)
            neg_texts = neg_texts * 2
            neg_texts = neg_texts[_offset: _offset + self.neg_per_ins]
        else:
            # neg_texts = negatives[:self.neg_per_ins]  # tevatron
            neg_texts = list()
            for _ in range(self.neg_per_ins):
                neg = negatives.pop(0)
                neg_texts.append(neg)
                negatives.append(neg)

        return {'query': ins['query'], 'docs': [pos_text, *neg_texts]}

    def __len__(self):
        return len(self.instances)


#### from https://github.com/izhx/uni-rep
from typing import Any, Dict, List

from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

@dataclass
class CollatorForRetrieval:
    tokenizer: PreTrainedTokenizerBase
    max_length_query: int
    max_length_doc: int

    def __call__(self, features: List[Dict[str, Any]]) -> dict:
        if 'query' in features[0]:
            queries = [f['query'] for f in features]
            query_encoding = self.encode(queries, self.max_length_query)
        else:
            query_encoding = None

        if 'docs' in features[0]:
            docs = sum([f['docs'] for f in features], list())
            doc_encoding = self.encode(docs, self.max_length_doc)
        else:
            doc_encoding = None
        return dict(query_encoding=query_encoding, doc_encoding=doc_encoding)

    def encode(self, texts, max_length) -> BatchEncoding:
        output = self.tokenizer(texts, max_length=max_length, truncation=True, padding=True, return_tensors='pt')
        return output


#### from https://github.com/izhx/uni-rep
from transformers import PreTrainedModel

@dataclass
class RetrievalModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    query_rep: Optional[torch.FloatTensor] = None
    doc_rep: Optional[torch.FloatTensor] = None
    rep: Optional[torch.FloatTensor] = None


class ModelForRetrieval(torch.nn.Module):
    def __init__(self, base_model, pooler_type: str = 'first_token', loss_fn=None):
        super().__init__()
        self.base_model = base_model
        self.pooler_type = pooler_type
        self.loss_fn = loss_fn or MultipleNegativesRankingLoss()
        assert pooler_type in {'first_token', 'last_token', 'mean', 'weightedmean'}

    def forward(
        self,
        query_encoding: Optional[dict] = None,
        doc_encoding: Optional[dict] = None,
    ) -> RetrievalModelOutput:
        """
        """
        query_rep, doc_rep, loss, rep = None, None, None, None

        if query_encoding is not None:
            query_rep = rep = self.encode_query(**query_encoding)

        if doc_encoding is not None:
            doc_rep = rep = self.encode_doc(**doc_encoding)

        if query_rep is not None and doc_rep is not None:
            assert callable(self.loss_fn)
            loss = self.loss_fn(query_rep, doc_rep)
            rep = None

        return RetrievalModelOutput(loss=loss, query_rep=query_rep, doc_rep=doc_rep, rep=rep)

    def encode_query(self, **kwargs) -> torch.Tensor:
        # Encode query inputs into query representations (via `_encode` by default)
        return self._encode(self.base_model, **kwargs)

    def encode_doc(self, **kwargs) -> torch.Tensor:
        # Encode doc inputs into doc representations (via `_encode` by default)
        return self._encode(self.base_model, **kwargs)

    def _encode(
        self,
        model: PreTrainedModel,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        # Encode inputs into neural representations by the give model.
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs
        )
        embedding = self.pooling(output.last_hidden_state, attention_mask)
        return embedding

    def pooling(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.pooler_type == 'first_token':
            return hidden_state[:, 0]
        else:
            assert attention_mask is not None, f"pooling {self.pooler_type} needs attention_mask"

        if attention_mask.ndim == 2:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size())
        elif attention_mask.ndim == 3:
            mask_expanded = attention_mask
        else:
            raise RuntimeError(f"Unexpected {attention_mask.ndim=}")

        hidden_state = hidden_state * mask_expanded

        if self.pooler_type == 'last_token':
            n, l, h = hidden_state.shape

            # Get shape [n] indices of the last token (i.e. the last token for each batch item)
            # Any sequence where min == 1, we use the entire sequence lenth since argmin = 0
            values, indices = torch.min(attention_mask, 1, keepdim=False)
            gather_indices = torch.where(values == 0, indices, l) - 1 # Shape [n]

            # There are empty sequences, where the index would become -1 which will crash
            gather_indices = torch.clamp(gather_indices, min=0)

            # Turn indices from shape [n] --> [n, 1, h]
            gather_indices = gather_indices.unsqueeze(1).unsqueeze(1).expand(n, 1, h)

            # Gather along the 1st dim (l) (n, l, h -> n, h)
            pooled_output = torch.gather(hidden_state, 1, gather_indices).squeeze(dim=1)

        elif self.pooler_type == 'mean':
            lengths = mask_expanded.sum(1).clamp(min=1e-9)
            pooled_output = hidden_state.sum(dim=1) / lengths

        elif self.pooler_type == 'weightedmean':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
            # hidden_state shape: [bs, seq, hidden_dim]
            weights = (
                    torch.arange(start=1, end=hidden_state.shape[1] + 1)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .expand(hidden_state.size())
                    .float().to(hidden_state.device)
                )
            assert weights.shape == hidden_state.shape == input_mask_expanded.shape
            input_mask_expanded = input_mask_expanded * weights
            
            sum_embeddings = torch.sum(hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            pooled_output = sum_embeddings / sum_mask

        else:
            raise ValueError(f"Wrong pooler : {self.pooler_type}")

        return pooled_output


def cos_sim(a: torch.Tensor, b: torch.Tensor, do_norm=True):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    if do_norm:
        a = torch.nn.functional.normalize(a, p=2, dim=1)
        b = torch.nn.functional.normalize(b, p=2, dim=1)

    return torch.mm(a, b.transpose(0, 1))


#### from https://github.com/izhx/uni-rep
from functools import partial

import torch.distributed as dist

class MultipleNegativesRankingLoss(torch.nn.Module):
    def __init__(
        self,
        scale: float = 20.0,
        score: str = 'cos_sim',
        reduction: str = 'mean',
    ):
        super().__init__()
        self.scale = scale
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction=reduction)
        if score == 'cos_sim':
            self.score_fct = cos_sim
        elif score == 'dot':
            self.score_fct = partial(cos_sim, do_norm=False)
        else:
            raise ValueError('Unsupported score function: %s', score)

    def forward(self, reps_a, reps_b):
        space = reps_b.size(0) // reps_a.size(0)  # num docs per instance
        offset = 0

        if dist.is_initialized():
            # Because we do not gather reps_a (i.e. we only have part of them)
            # we need to move the labels to the corresponding part of reps_b we have
            # i.e. on rank 0, it is the first len(reps_b); on rank 1 it's the
            # second len(reps_b). so we add `offset = len(reps_b) * rank` to labels
            offset = reps_b.size(0) * dist.get_rank()
            full_reps_b = [reps_b.new_zeros(reps_b.size()) for _ in range(dist.get_world_size())]
            all_gather_with_grad(full_reps_b, reps_b.contiguous())
            reps_b = torch.cat(full_reps_b)

        labels = torch.arange(reps_a.size(0), device=reps_a.device) * space + offset
        scores = self.score_fct(reps_a, reps_b) * self.scale
        loss = self.cross_entropy(scores, labels)
        return loss


# from https://github.com/vlkit/vlkit/blob/main/vlkit/ops/distributed.py
class AllGather(torch.autograd.Function):
    """
    all_gather with gradient back-propagation
    """

    @staticmethod
    def forward(ctx, tensor_list, tensor, group=None, async_op=False):
        dist.all_gather(tensor_list, tensor, group=group, async_op=async_op)
        return tuple(tensor_list)

    @staticmethod
    def backward(ctx, *grad_list):
        grad_list = list(grad_list)
        rank = dist.get_rank()

        dist_ops = [
            dist.reduce(grad_list[i], i, async_op=True) for i in range(dist.get_world_size())
        ]

        for op in dist_ops:
            op.wait()

        return None, grad_list[rank], None, None


all_gather_with_grad = AllGather.apply
####

import heapq
from itertools import chain, repeat
from typing import Dict, Optional

from tqdm.autonotebook import tqdm

from torch.utils.data import DataLoader
import torch.distributed as dist

from transformers.trainer import (
    TRAINING_ARGS_NAME, Trainer, EvalLoopOutput, EvalPrediction, denumpify_detensorize
)
from transformers.training_args import ParallelMode
from transformers.utils import is_sagemaker_mp_enabled


def gather_obj_to_list(obj):
    array = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(array, obj)
    return array


class RetrievalTrainer(Trainer):
    """
        model: ModelForRetrieval,
        args: TrainingArguments,
        data_collator: CollatorForRetrieval,
        train_dataset: Optional[TrainDataset] = None,
    """

    def __init__(self, **kwargs):
        self.data_args = kwargs.pop('data_args')
        self.score_fct = kwargs.pop('score_fct', cos_sim)

        super().__init__(**kwargs)

        if self.data_args.grad_cache:
            logger.warning("Grad cache should be the last method to enlarge the batch_size...")
            if self.is_deepspeed_enabled or is_sagemaker_mp_enabled() or self.use_apex:
                raise ValueError(
                    "Using grad_cache together with deepspeed/sagemaker_mp/apex is not possible."
                )

            from grad_cache import GradCache

            scaler = self.scaler if getattr(self, 'do_grad_scaling', False) else self.accelerator.scaler
            self.gc = GradCache(
                models=[self.model, self.model],
                chunk_sizes=[self.data_args.gc_query_chunk_size, self.data_args.gc_doc_chunk_size],
                loss_fn=self.model.loss_fn,
                split_input_fn=split_inputs,
                get_rep_fn=get_rep,
                fp16=self.args.fp16,
                scaler=scaler if self.args.fp16 else None
            )

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.model.base_model.save_pretrained(
            output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if model is None:
            model = self.model.base_model
        super()._load_from_checkpoint(resume_from_checkpoint, model)

    def _load_best_model(self):
        self_model = self.model
        self.model = self_model.base_model
        super()._load_best_model()
        self.model = self_model

    def training_step(self, model: torch.nn.Module, inputs) -> torch.Tensor:
        if not self.data_args.grad_cache:
            return super().training_step(model, inputs)

        model.train()
        inputs = self._prepare_inputs(inputs)

        inputs_doc = dict(doc_encoding=inputs.pop('doc_encoding'))
        _distributed = self.args.parallel_mode == ParallelMode.DISTRIBUTED
        self.gc.models = [model, model]  # AssertionError: Some of models are not wrapped in DistributedDataParallel. Make sure you are running DDP with proper initializations
        loss = self.gc(inputs, inputs_doc, no_sync_except_last=_distributed)

        return loss.detach() / self.args.gradient_accumulation_steps

    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return eval_dataset

    def get_test_dataloader(self, test_dataset):
        return test_dataset

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        def shard_data(data: list):
            shard_size = len(data) // self.accelerator.num_processes
            start = shard_size * self.accelerator.process_index
            end = len(data) if self.accelerator.is_last_process else start + shard_size
            data = data[start: end]
            logger.warning(f"[rank {self.accelerator.process_index}] data slice {start}:{end}")
            return data

        def encode(data, input_name, max_length, chunk_size=None):
            # Encode data
            def collate_fn(features) -> tuple:
                ids = [f['id'] for f in features]
                texts = [f['text'] for f in features]
                encoding = self.data_collator.tokenizer(
                    texts,
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                    return_tensors='pt'
                )
                return ids, encoding

            loader = DataLoader(
                data,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=1,
                collate_fn=collate_fn,
            )
            ids, embeds = list(), list()
            with torch.inference_mode():
                for (batch_ids, batch) in loader:
                    intputs = {input_name: batch.to(self.accelerator.device)}
                    output = model(**intputs)
                    reps = output.rep
                    ids.append(batch_ids)
                    embeds.append(reps)
                    if chunk_size and len(ids) >= chunk_size:
                        ids = sum(ids, start=list())
                        embeds = torch.cat(embeds)
                        yield ids, embeds
                        ids, embeds = list(), list()
                if len(ids):
                    ids = sum(ids, start=list())
                    embeds = torch.cat(embeds)
                    yield ids, embeds

        queries, corpus, qrels = dataloader[0], dataloader[1], dataloader[2]
        prefix = ""
        if self.accelerator.num_processes > 1:
            queries = shard_data(queries)
            prefix = f"[Rank {self.accelerator.process_index}]"

        logger.info(f"{prefix} Encoding {len(queries)} queries")
        query_ids, query_embed = list(encode(queries, 'query_encoding', self.data_args.max_query_length))[0]
        if self.accelerator.num_processes > 1:
            padded = self.accelerator.pad_across_processes(query_embed)
            output_tensors = [torch.empty_like(padded) for _ in range(self.accelerator.num_processes)]
            dist.all_gather(output_tensors, padded)
            lengths = gather_obj_to_list(query_embed.size(0))
            output_tensors = [t[:l] for t, l in zip(output_tensors, lengths)]
            query_embed = torch.cat(output_tensors)
            all_ids = gather_obj_to_list(query_ids)
            query_ids = sum(all_ids, start=list())
        logger.info(f"{prefix} get query_embed {query_embed.shape}")

        top_k = self.data_args.top_k  # 1000
        # Sort by text length
        logger.info(f"{prefix} Sorting corpus ({len(corpus)}) by text length (Longest first)...")
        corpus = sorted(corpus, key=lambda x: x['text'], reverse=True)
        # Keep only the top-k docs for each query
        result_heaps = {qid: [] for qid in query_ids}
        chunk_num = math.ceil(math.ceil(len(corpus) / batch_size) / self.data_args.num_batch_per_chunk)
        for no, (doc_ids, chunk_embed) in tqdm(enumerate(encode(
            corpus, 'doc_encoding', self.data_args.max_doc_length, self.data_args.num_batch_per_chunk
        )), total=chunk_num, disable=not self.accelerator.is_main_process):
            logger.info(f"{prefix} Encoded corpus chunk {no + 1}/{chunk_num} size {len(doc_ids)}")

            # Compute similarites using either cosine-similarity or dot product
            scores = self.score_fct(query_embed, chunk_embed)
            scores[torch.isnan(scores)] = -1

            # Get top-k values
            top_k_values, top_k_idx = torch.topk(scores, min(top_k+1, scores.size(1)), dim=1, largest=True)
            top_k_values = top_k_values.cpu().tolist()
            top_k_idx = top_k_idx.cpu().tolist()

            for i, query_id in enumerate(query_ids):
                for j, score in zip(top_k_idx[i], top_k_values[i]):
                    doc_id = doc_ids[j]
                    if self.data_args.ignore_identical and doc_id == query_id:
                        continue
                    if len(result_heaps[query_id]) < top_k:
                        # Push item on the heap
                        heapq.heappush(result_heaps[query_id], (score, doc_id))
                    else:
                        # If item is larger than the smallest in the heap,
                        # push it on the heap then pop the smallest element.
                        # Tuples are compared by each item.
                        heapq.heappushpop(result_heaps[query_id], (score, doc_id))

        logger.info(prefix+ " Done encoding and searching, calculating results...")
        sort_by_score = partial(sorted, key=lambda x: x[0], reverse=True)
        if self.accelerator.num_processes > 1:
            results = gather_obj_to_list(result_heaps)
            result_heaps = {
                k: sort_by_score(chain(*[r[k] for r in results]))[:top_k] for k in query_ids
            }
            logger.info(prefix + " Gathered all results.")
        else:
            # Sort for single process runing
            result_heaps = {k: sort_by_score(v) for k, v in result_heaps.items()}

        # Metrics!
        metrics = {}
        if self.compute_metrics is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=result_heaps, label_ids=qrels))

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=result_heaps, label_ids=None, metrics=metrics, num_samples=len(query_ids))


def get_rep(x):
    return x.rep


def split_inputs(model_input: dict, chunk_size: int):
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]

    keys = list(arg_val.keys())
    chunked_tensors = [arg_val[k].split(chunk_size, dim=0) for k in keys]
    chunked_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]

    return [{arg_key: c} for c in chunked_arg_val]
####


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 4. Load dataset
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full image path and the second column for the
    # captions (unless you specify column names for this with the `image_column` and `caption_column` arguments).
    #
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            keep_in_memory=False,
            # data_dir=data_args.data_dir,
            token=model_args.token,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # 5. Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        # do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    base_model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        # False,  # add_pooler
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    loss_fn = None
    if model_args.loss_kwargs is not None:
        loss_fn = MultipleNegativesRankingLoss(**model_args.loss_kwargs)
    model = ModelForRetrieval(base_model, model_args.pooler_type, loss_fn)

    # 6. Preprocessing the datasets.
    if data_args.text_column_names is None:
        query_column_name = 'query'
        positive_column_name = 'positives'
        negative_column_name = 'negatives'
    else:
        if isinstance(data_args.text_column_names, str):
            data_args.text_column_names = data_args.text_column_names.split(',')
        query_column_name, *other_names = data_args.text_column_names
        if len(other_names) < 2:
            other_names = other_names + [None]
        positive_column_name, negative_column_name = other_names

    def check_column_names(dataset):
        assert query_column_name in dataset.column_names
        assert positive_column_name in dataset.column_names
        if isinstance(data_args.text_column_names, list) and len(data_args.text_column_names) > 2:
            assert negative_column_name in dataset.column_names

    # Load trainset
    # doc_is_idx = False
    if training_args.do_train:
        import json

        check_column_names(raw_datasets['train'])
        instances = list()
        for row in raw_datasets["train"]:
            pos = row[positive_column_name]
            ins = {'query': row[query_column_name], 'positives': pos if isinstance(pos, list) else [pos]}
            if negative_column_name in row:
                neg = row[negative_column_name]
                ins['negatives'] = neg if isinstance(neg, list) else [neg]
            if len(instances) < 2:
                logger.info("Training instance: %s", json.dumps(ins, ensure_ascii=False))
            instances.append(ins)
        logger.info(f"Total {len(instances)} training instances")

        if data_args.max_train_samples is not None:
            max_train_samples = min(len(instances), data_args.max_train_samples)
            instances = instances[:max_train_samples]

        train_dataset = TrainDataset(
            instances,
            data_args.negatives_per_instance,
            safe_negatives=data_args.safe_negatives
        )
        training_args.remove_unused_columns = False

    def prepare_eval(dataset, all_queries=None, all_corpus=None, all_qrels=None, source=None, only_query=False):
        queries = list() if all_queries is None else all_queries
        corpus = dict() if all_corpus is None else all_corpus
        qrels = dict() if all_qrels is None else all_qrels
        if only_query:
            for ins in dataset:
                qid = len(queries)
                queries.append({'id': qid, 'text': ins[query_column_name], 'source': source})
                if data_args.positive_id_column_name is not None:
                    qrels[qid] = {ins[data_args.positive_id_column_name]: 1}
        else:
            for q, d in zip(dataset[query_column_name], dataset[positive_column_name]):
                qid = len(queries)
                queries.append({'id': qid, 'text': q, 'source': source})
                if d not in corpus:
                    corpus[d] = len(corpus)
                qrels[qid] = {corpus[d]: 1}
            if all_corpus is None:
                corpus = [{'id': i, 'text': t} for t, i in corpus.items()]
        return queries, corpus, qrels

    if training_args.do_eval:
        eval_dataset = prepare_eval(raw_datasets['validation'])

    # 7. Init data_collator and compute_metrics
    if data_args.max_query_length is None:
        data_args.max_query_length = data_args.max_seq_length
    if data_args.max_doc_length is None:
        data_args.max_doc_length = data_args.max_seq_length
    data_collator = CollatorForRetrieval(tokenizer, data_args.max_query_length, data_args.max_doc_length)

    k_values = [k for k in [1, 3, 5, 10, 100, 1000] if k < data_args.top_k] + [data_args.top_k]

    def compute_metrics(p):
        result_heaps, qrels = p
        if data_args.task_type == 'bitext': # or global_corpus is None:
            n_correct = 0
            recall_set = set()
            for qid, v in result_heaps.items():
                score, docid = v[0]
                recall_set.add(docid)
                n_correct += (qrels[qid][docid] if docid in qrels[qid] else 0)

            recall = accuracy = n_correct / len(qrels)
            precision = n_correct / len(recall_set)
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            metrics = {"accuracy": accuracy, 'f1': f1}
            return metrics
        else:
            import pandas as pd
            import pytrec_eval

            # https://github.com/beir-cellar/beir/blob/main/beir/retrieval/custom_metrics.py
            def mrr(qrels: Dict[str, Dict[str, int]],
                    results: Dict[str, Dict[str, float]],
                    k_values: List[int]
                ) -> Dict[str, float]:

                MRR = {}
                
                for k in k_values:
                    MRR[f"mrr_at_{k}"] = 0.0
                
                k_max, top_hits = max(k_values), {}
                
                for query_id, doc_scores in results.items():
                    # top_hits[query_id] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]
                    top_hits[query_id] = list(doc_scores.items())[0:k_max]

                for query_id, hits in top_hits.items():
                    relevant_docs = set([doc_id for doc_id, score in qrels[query_id].items() if score > 0])
                    for k in k_values:
                        for rank, hit in enumerate(hits[0:k]):
                            if hit[0] in relevant_docs:
                                MRR[f"mrr_at_{k}"] += 1.0 / (rank + 1)
                                break

                for k in k_values:
                    MRR[f"mrr_at_{k}"] = MRR[f"mrr_at_{k}"] / len(qrels)

                return MRR

            map_string = "map_cut." + ",".join([str(k) for k in k_values])
            ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
            recall_string = "recall." + ",".join([str(k) for k in k_values])
            precision_string = "P." + ",".join([str(k) for k in k_values])
            qrels = {str(qid): {str(docid): s for docid, s in v.items()} for qid, v in qrels.items()}
            results = {str(qid): {str(docid): s for s, docid in v} for qid, v in result_heaps.items()}
            evaluator = pytrec_eval.RelevanceEvaluator(
                qrels, {map_string, ndcg_string, recall_string, precision_string}
            )
            scores_by_query = evaluator.evaluate(results)
            scores = pd.DataFrame.from_dict(scores_by_query.values()).mean()
            metrics = mrr(qrels, results, k_values)
            for prefix in ('map_cut', 'ndcg_cut', 'recall', 'P'):
                name = 'precision' if prefix == 'P' else prefix.split('_')[0]
                for k in k_values:
                    metrics[f'{name}_at_{k}'] = scores[f'{prefix}_{k}']
            return metrics

    # 8. Initialize our trainer
    trainer = RetrievalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        data_args=data_args,
        score_fct=cos_sim if data_args.score_fct == 'cos_sim' else partial(cos_sim, do_norm=False)
    )

    # 9. Training
    if training_args.do_train:
        train_dataset.trainer = trainer
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        if data_args.dataset_config_name is not None:
            train_result.metrics['config'] = data_args.dataset_config_name
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # 10. Evaluation and Prediction
    if training_args.do_eval:
        metrics = trainer.evaluate()
        if data_args.dataset_config_name is not None:
            metrics['config'] = data_args.dataset_config_name
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        if data_args.test_config_names is None:
            config_names = [data_args.dataset_config_name or 'defualt']

        # only for datasets on HF hub
        elif data_args.test_config_names == 'all_config':
            if os.environ['HF_DATASETS_OFFLINE'] == '1':
                raise ValueError("`all_config` is not working in offline mode.")
            config_names = datasets.get_dataset_config_names(data_args.dataset_name)
        else:
            config_names = data_args.test_config_names.split(',')

        test_split = data_args.test_split or 'test'

        def get_testset(config_name: str):
            if data_args.dataset_name is None or (
                test_split in raw_datasets and config_name == raw_datasets[test_split].config_name
            ):
                test_ds = raw_datasets[test_split]
            else:
                test_ds = load_dataset(
                    data_args.dataset_name,
                    config_name,
                    split=test_split,
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                )
            return test_ds

        only_query = False
        if data_args.corpus_dataset_name or data_args.corpus_dataset_config_name or data_args.corpus_split:
            only_query = True
            if (
                data_args.corpus_dataset_name is None and
                data_args.corpus_dataset_config_name is None and
                data_args.corpus_split is not None and
                data_args.corpus_split in raw_datasets
            ):
                corpus_ds = raw_datasets[data_args.corpus_split]
            else:
                corpus_ds = load_dataset(
                    data_args.corpus_dataset_name or data_args.dataset_name,
                    data_args.corpus_dataset_config_name or data_args.dataset_config_name,
                    split=data_args.corpus_split or 'test',
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                )
            doc_id_column = data_args.positive_id_column_name or 'id'
            doc_text_column = data_args.corpus_text_column_name or 'text'
            corpus = [{'id': i[doc_id_column], 'text': i[doc_text_column]} for i in corpus_ds]

        all_queries, all_corpus, all_qrels = list(), dict(), dict()
        if data_args.merge_test:
            for config_name in config_names:
                test_ds = get_testset(config_name)
                prepare_eval(test_ds, all_queries, all_corpus, all_qrels, config_name, only_query)
            if only_query:
                all_corpus = corpus
            else:
                all_corpus = [{'id': i, 'text': t} for t, i in all_corpus.items()]
            logger.info(f"*** Predict merged ***")
            predictions, labels, metrics = trainer.predict(
                (all_queries, all_corpus, all_qrels), metric_key_prefix="predict"
            )
        else:
            metrics = dict()

        all_metrics = dict()
        for config_name in config_names:
            logger.info(f"*** Predict {config_name} ***")
            if data_args.merge_test:
                results, qrels = dict(), dict()
                for q in all_queries:
                    if q['source'] == config_name:
                        results[q['id']] = predictions[q['id']]
                        qrels[q['id']] = all_qrels[q['id']]
                sub_metrics = compute_metrics((results, qrels))
            else:
                test_ds = get_testset(config_name)
                test_ds = prepare_eval(test_ds)
                *_, sub_metrics = trainer.predict(test_ds, metric_key_prefix="predict")

            sub_metrics['config'] = config_name
            # trainer.log_metrics("predict", metrics)
            # trainer.save_metrics("predict", metrics)
            all_metrics[config_name] = sub_metrics

        if data_args.test_config_names is not None:
            metrics['all_metrics'] = all_metrics
        trainer.save_metrics("predict", metrics)

    # 11. Write Training Stats and push to hub.
    trainer.model = trainer.model.base_model
    finetuned_from = model_args.model_name_or_path
    # If from a local directory, don't set `finetuned_from` as this is required to be a valid repo. id on the Hub.
    if os.path.isdir(finetuned_from):
        finetuned_from = None
    kwargs = {"finetuned_from": finetuned_from, "tasks": data_args.task_type}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()

"""
pip install pytrec_eval
"""
