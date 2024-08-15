import json
import re
import os
import string
import bz2
import random
import platform
import sys
from collections import Counter
from typing import List
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from dataset_utils.NQ_TriviaQA_utils import has_answer

random.seed(0)


def _normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _update_sp(metrics, prediction, gold):
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    return em, prec, recall


def _exact_match_score(prediction, ground_truth):
    return _normalize_answer(prediction) == _normalize_answer(ground_truth)


def _f1_score(prediction, ground_truth):
    normalized_prediction = _normalize_answer(prediction)
    normalized_ground_truth = _normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def _update_answer(metrics, prediction, gold):
    em = _exact_match_score(prediction, gold)
    f1, prec, recall = _f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, f1, prec, recall


class HotpotQAUtils:
    def __init__(self):
        self.root = root_path
        self.qs_file = os.path.join(self.root, 'data/hotpotQA/hotpot_dev_distractor_v1.json')
        self.sample_qs_file = os.path.join(self.root, 'data/hotpotQA/sampled_data.json')

    def load_qs_list(self):
        qs_list = json.load(open(self.sample_qs_file, 'r'))
        _qs_list = []
        for qs in qs_list:
            _qs_list.append(dict(qs_id=qs['_id'], question=qs['question']))
        return _qs_list

    def load_oracle_list(self):
        qs_list = json.load(open(self.sample_qs_file, 'r', encoding='utf-8'))
        _qs_list = []
        for qs in qs_list:
            proc_sp = list(set([sp[0] for sp in qs['supporting_facts']]))
            assert len(proc_sp) == 2    # 2 supporting facts per sample in hotpotQA
            _qs_list.append(dict(qs_id=qs['_id'], oracle_docs=proc_sp, answer=qs['answer']))
        return _qs_list

    def sample_dataset(self, k=2000, sampled_data_file=None):
        """
        sample 2000 queries
        :return:
        """
        qs_list = json.load(open(self.qs_file, 'r'))
        problem_id_list = list(range(0, len(qs_list)))
        sampled_id_list = random.sample(problem_id_list, k=k)
        sampled_data = list()
        for id in sampled_id_list:
            sampled_data.append(qs_list[id])

        if sampled_data_file is not None:
            with open(sampled_data_file, 'w+') as f:
                json.dump(sampled_data, f, indent=2)
        return sampled_data

    def get_sample(self, sample_num):
        """
        for prompt generation usage, randomly sample instances in dataset
        :param sample_num:
        :return:
        """
        qs_list = json.load(open(self.qs_file, 'r'))
        problem_id_list = list(range(0, len(qs_list)))
        sampled_qs_list = json.load(open(self.sample_qs_file, 'r'))
        new_sampled_id_list = []
        while len(new_sampled_id_list) < sample_num:
            sampled_id = random.sample(problem_id_list, 1)[0]
            if sampled_id not in new_sampled_id_list and qs_list[sampled_id] not in sampled_qs_list:
                new_sampled_id_list.append(sampled_id)
        new_sampled_qs_list = []
        for id in new_sampled_id_list:
            new_sampled_qs_list.append(qs_list[id])

        return new_sampled_qs_list

    @staticmethod
    def eval_sp(preds, golds, top_k):
        """
        evaluate retrieval doc acc
        :param preds:
        :param golds:
        :param top_k:
        :return:
        """
        assert len(preds) == len(golds)
        metrics = dict()
        for k in top_k:
            metrics[k] = {'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0}
            for pred, gold in zip(preds, golds):
                _update_sp(metrics=metrics[k], prediction=pred[:k], gold=gold)
            N = len(golds)
            for key in metrics[k].keys():
                metrics[k][key] /= N
        _metrics = dict()
        for k in top_k:
            _metrics[k] = round(metrics[k]['sp_recall'], 3)
        print(_metrics)
        return metrics

    @staticmethod
    def eval_pred(pred_list, oracle_list):
        """
        evaluate final prediction acc
        :param pred_list:
        :param oracle_list:
        :return:
        """
        metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0, 'has_answer': 0}
        eval_records = dict()
        for pred, oracle in zip(pred_list, oracle_list):
            eval_records[pred['qs_id']] = dict()
            assert pred['qs_id'] == oracle['qs_id']
            if has_answer([oracle['answer']], pred['output']): eval_records[pred['qs_id']]['has_answer'] = True
            else: eval_records[pred['qs_id']]['has_answer'] = False
            em, f1, prec, recall = _update_answer(metrics, pred['output'], oracle['answer'])
            eval_records[pred['qs_id']]['em'] = em
            eval_records[pred['qs_id']]['f1'] = f1
            eval_records[pred['qs_id']]['prec'] = prec
            eval_records[pred['qs_id']]['recall'] = recall
            eval_records[pred['qs_id']]['has_answer'] += 1 if eval_records[pred['qs_id']]['has_answer'] is True else 0
        N = len(oracle_list)
        for k in metrics.keys():
            metrics[k] /= N
        # metrics = dict(em=metrics['em'], f1=metrics['f1'])
        # print(metrics)
        return metrics, eval_records

