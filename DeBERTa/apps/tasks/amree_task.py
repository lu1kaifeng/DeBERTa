#
# Author: penhe@microsoft.com
# Date: 01/25/2019
#

from glob import glob
from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from bisect import bisect
import copy
import math

import numpy
from scipy.special import softmax
import numpy as np
import pdb
import os
import sys
import csv

import random
import torch
import re
import ujson as json
from torch.utils.data import DataLoader

from ace2005_module.data_load import ACE2005Dataset, idx2trigger, pad
from ace2005_module.model import Net
from ace2005_module.utils import find_triggers, calc_metric
from .metrics import *
from .task import EvalData, Task
from .task_registry import register_task
from ...data.static_dataset import StaticDataset
from ...utils import xtqdm as tqdm
from ...training import DistributedTrainer, batch_to
from ...data import DistributedBatchSampler, SequentialSampler, BatchSampler, AsyncDataLoader
from ...data import ExampleInstance, ExampleSet, DynamicDataset, example_to_feature
from ...data.example import _truncate_segments
from ...data.example import *
from ...utils import get_logger
from ..models import MaskedLanguageModel
from .._utils import merge_distributed, join_chunks

logger = get_logger()

__all__ = ["AMREETask"]


@register_task(name="AMREE", desc="AMR event extraction task")
class AMREETask(Task):
    def __init__(self, data_dir, tokenizer, args, **kwargs):
        super().__init__(tokenizer, args, **kwargs)
        self.roles = None
        self.data_dir = data_dir
        self.split = 0.9

    def _list_splitter(self,list_to_split, ratio):
        elements = len(list_to_split)
        middle = int(elements * ratio)
        return [list_to_split[:middle], list_to_split[middle:]]
    def train_data(self, max_seq_len=512, **kwargs):
        return ACE2005Dataset(os.path.join(self.data_dir, 'train_mat.json'),None,os.path.join(self.data_dir, 'train_mat.json.cache'),os.path.join(self.data_dir, 'roles.txt'),self.tokenizer)


    def get_labels(self):
        return list(self.tokenizer.vocab.values())

    def eval_data(self, max_seq_len=512, **kwargs):
        return self._data('eval',ACE2005Dataset(os.path.join(self.data_dir, 'test_mat.json'),None,os.path.join(self.data_dir, 'test_mat.json.cache'),os.path.join(self.data_dir, 'roles.txt'),self.tokenizer))

    def test_data(self, max_seq_len=512, **kwargs):
        """See base class."""
        raise NotImplemented('This method is not implemented yet.')

    def _data(self, name, examples, type_name='dev', ignore_metric=False):
        predict_fn = self.get_predict_fn()
        return EvalData(name, examples,
                        metrics_fn=self.get_metrics_fn(), predict_fn=predict_fn, ignore_metric=ignore_metric,
                        critial_metrics=['accuracy'])

    def get_metrics_fn(self):
        """Calcuate metrics based on prediction results"""

        def metrics_fn(logits, labels):
            preds = logits
            acc = (preds == labels).sum() / len(labels)
            metrics = OrderedDict(accuracy=acc)
            return metrics

        return metrics_fn

    def get_loss_fn(self, *args, **kwargs):
        def _loss_fn(trainer, model: Net, data):
            tokens_x_2d, entities_x_3d, postags_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d, adjm, task_id,amrMat = data
            trigger_loss, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys = model.predict_triggers(
                tokens_x_2d=tokens_x_2d, entities_x_3d=entities_x_3d,
                postags_x_2d=postags_x_2d, head_indexes_2d=head_indexes_2d,
                triggers_y_2d=triggers_y_2d, arguments_2d=arguments_2d,amrMat=amrMat)
            if len(argument_keys) > 0:
                argument_loss, arguments_y_2d, argument_hat_1d, argument_hat_2d = model.predict_arguments(
                    argument_hidden, argument_keys, arguments_2d)
                loss = trigger_loss + argument_loss


            else:
                loss = trigger_loss
            return loss,1

        return _loss_fn

    def get_eval_fn(self):

        def validation(args, model, device, iterator: EvalData, prefix=None, tag=None, steps=None, return_logits=False):
            model.eval()
            cum_loss = 0
            argument_count = 1
            trigger_count = 1
            cum_argument_loss = 0
            cum_trigger_loss = 0
            words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all = [], [], [], [], []
            logits_list = [[], []]
            with torch.no_grad():
                eval_loader = DataLoader(iterator.data,batch_size=args.eval_batch_size,collate_fn=pad)
                for i, batch in enumerate(eval_loader):
                    tokens_x_2d, entities_x_3d, postags_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d, adjm, task_id,amrMat = batch
                    if return_logits:
                        trigger_loss, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys, trigger_logits = model.predict_triggers(
                            tokens_x_2d=tokens_x_2d, entities_x_3d=entities_x_3d,
                            postags_x_2d=postags_x_2d, head_indexes_2d=head_indexes_2d,
                            triggers_y_2d=triggers_y_2d, arguments_2d=arguments_2d, return_logits=True,amrMat=amrMat)
                        logits_list[0].append(
                            (trigger_loss, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys,
                             trigger_logits))
                    else:
                        trigger_loss, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys = model.predict_triggers(
                            tokens_x_2d=tokens_x_2d, entities_x_3d=entities_x_3d,
                            postags_x_2d=postags_x_2d, head_indexes_2d=head_indexes_2d,
                            triggers_y_2d=triggers_y_2d, arguments_2d=arguments_2d,amrMat=amrMat)

                    words_all.extend(words_2d)
                    triggers_all.extend(triggers_2d)
                    triggers_hat_all.extend(trigger_hat_2d.cpu().numpy().tolist())
                    arguments_all.extend(arguments_2d)

                    if len(argument_keys) > 0:
                        if return_logits:
                            argument_loss, arguments_y_2d, argument_hat_1d, argument_hat_2d, argument_logits = model.predict_arguments(
                                argument_hidden, argument_keys, arguments_2d, return_logits=True)
                            logits_list[1].append(
                                (argument_loss, arguments_y_2d, argument_hat_1d, argument_hat_2d, argument_logits))
                        else:
                            argument_loss, arguments_y_2d, argument_hat_1d, argument_hat_2d = model.predict_arguments(
                                argument_hidden, argument_keys, arguments_2d)
                        arguments_hat_all.extend(argument_hat_2d)
                        cum_trigger_loss += trigger_loss
                        cum_loss += trigger_loss + argument_loss
                        cum_argument_loss += argument_loss
                        trigger_count += 1
                        argument_count += 1
                        # if i == 0:

                        #     print("=====sanity check for triggers======")
                        #     print('triggers_y_2d[0]:', triggers_y_2d[0])
                        #     print("trigger_hat_2d[0]:", trigger_hat_2d[0])
                        #     print("=======================")

                        #     print("=====sanity check for arguments======")
                        #     print('arguments_y_2d[0]:', arguments_y_2d[0])
                        #     print('argument_hat_1d[0]:', argument_hat_1d[0])
                        #     print("arguments_2d[0]:", arguments_2d)
                        #     print("argument_hat_2d[0]:", argument_hat_2d)
                        #     print("=======================")
                    else:
                        batch_size = len(arguments_2d)
                        argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
                        arguments_hat_all.extend(argument_hat_2d)
                        cum_loss += trigger_loss
                        cum_trigger_loss += trigger_loss
                        trigger_count += 1

            triggers_true, triggers_pred, arguments_true, arguments_pred = [], [], [], []
            for i, (words, triggers, triggers_hat, arguments, arguments_hat) in enumerate(
                    zip(words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all)):
                triggers_hat = triggers_hat[:len(words)]
                triggers_hat = [idx2trigger[hat] for hat in triggers_hat]

                # [(ith sentence, t_start, t_end, t_type_str)]
                triggers_true.extend([(i, *item) for item in find_triggers(triggers)])
                triggers_pred.extend([(i, *item) for item in find_triggers(triggers_hat)])

                # [(ith sentence, t_start, t_end, t_type_str, a_start, a_end, a_type_idx)]
                for trigger in arguments['events']:
                    t_start, t_end, t_type_str = trigger
                    for argument in arguments['events'][trigger]:
                        a_start, a_end, a_type_idx = argument
                        arguments_true.append((i, t_start, t_end, t_type_str, a_start, a_end, a_type_idx))

                for trigger in arguments_hat['events']:
                    t_start, t_end, t_type_str = trigger
                    for argument in arguments_hat['events'][trigger]:
                        a_start, a_end, a_type_idx = argument
                        arguments_pred.append((i, t_start, t_end, t_type_str, a_start, a_end, a_type_idx))

            # print(classification_report([idx2trigger[idx] for idx in y_true], [idx2trigger[idx] for idx in y_pred]))

            trigger_p, trigger_r, trigger_f1 = calc_metric(triggers_true, triggers_pred)
            argument_p, argument_r, argument_f1 = calc_metric(arguments_true, arguments_pred)
            eval_results = OrderedDict()
            eval_results[iterator.name] = (trigger_f1, argument_f1), OrderedDict()
            return eval_results

        return validation


    def get_model_class_fn(self):
        def partial_class(*wargs, **kwargs):

            from ace2005_module.consts import NONE, PAD, CLS, SEP, TRIGGERS, ARGUMENTS, ENTITIES, POSTAGS

            def build_vocab(labels, BIO_tagging=True):
                all_labels = [PAD, NONE]
                for label in labels:
                    if BIO_tagging:
                        all_labels.append('B-{}'.format(label))
                        all_labels.append('I-{}'.format(label))
                    else:
                        all_labels.append(label)
                label2idx = {tag: idx for idx, tag in enumerate(all_labels)}
                idx2label = {idx: tag for idx, tag in enumerate(all_labels)}
                # idx2label.update({tag: idx for idx, tag in enumerate(all_labels)})

                return all_labels, label2idx, idx2label

            all_triggers, trigger2idx, idx2trigger = build_vocab(TRIGGERS)
            all_entities, entity2idx, idx2entity = build_vocab(ENTITIES)
            all_postags, postag2idx, idx2postag = build_vocab(POSTAGS, BIO_tagging=False)
            all_arguments, argument2idx, idx2argument = build_vocab(ARGUMENTS, BIO_tagging=False)
            premodel = MaskedLanguageModel.load_model(*wargs, **kwargs)
            return Net(trigger_size=len(all_triggers),PreModel=premodel.deberta, entity_size=len(all_entities),
                  all_postags=len(all_postags),
                  argument_size=len(all_arguments), idx2trigger=idx2trigger)

        return partial_class

    @classmethod
    def add_arguments(cls, parser):
        """Add task specific arguments
      e.g. parser.add_argument('--data_dir', type=str, help='The path of data directory.')
    """
        parser.add_argument('--max_ngram', type=int, default=1, help='Maxium ngram sampling span')
        parser.add_argument('--num_training_steps', type=int, default=None, help='Maxium pre-training steps')


def test_MLGM():
    from ...deberta import tokenizers, load_vocab
    import pdb
    vocab_path, vocab_type = load_vocab(vocab_path=None, vocab_type='spm', pretrained_id='xlarge-v2')
    tokenizer = tokenizers[vocab_type](vocab_path)
    mask_gen = NGramMaskGenerator(tokenizer, max_gram=1)
    mlm = MLGMTask('/mnt/penhe/data/wiki103/spm', tokenizer, None)
    train_data = mlm.train_data(mask_gen=mask_gen)
    pdb.set_trace()
