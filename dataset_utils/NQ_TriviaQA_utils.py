import json
import os
from collections import Counter

import ijson
import random
import platform
import sys
import re, string
import unicodedata
import regex
import copy
from typing import List
system = platform.system()
if system == 'Darwin':
    root_path = '/'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/RAG_Benchmark_Code_QA'
sys.path.insert(0, root_path)
from dataset_utils.corpus_utils import WikiCorpusLoader

random.seed(0)


class Tokens(object):
    """A class to represent a list of tokenized text."""

    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i:j]
        return new_tokens

    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return "".join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        """Returns a list of the text of each token

        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if "pos" not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if "lemma" not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if "ner" not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.

        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """

        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [
            (s, e + 1)
            for s in range(len(words))
            for e in range(s, min(s + n, len(words)))
            if not _skip(words[s : e + 1])
        ]

        # Concatenate into strings
        if as_strings:
            ngrams = ["{}".format(" ".join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get("non_ent", "O")
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while idx < len(entities) and entities[idx] == ner_tag:
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups


class Tokenizer(object):
    """Base tokenizer class.
    Tokenizers implement tokenize, which should return a Tokens class.
    """

    def tokenize(self, text):
        raise NotImplementedError

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()


class SimpleTokenizer(Tokenizer):
    ALPHA_NUM = r"[\p{L}\p{N}\p{M}]+"
    NON_WS = r"[^\p{Z}\p{C}]"

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            "(%s)|(%s)" % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE,
        )
        # if len(kwargs.get("annotators", {})) > 0:
        #     logger.warning(
        #         "%s only tokenizes! Skipping annotators: %s" % (type(self).__name__, kwargs.get("annotators"))
        #     )
        self.annotators = set()

    def tokenize(self, text):
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()

            # Get whitespace
            span = matches[i].span()
            start_ws = span[0]
            if i + 1 < len(matches):
                end_ws = matches[i + 1].span()[0]
            else:
                end_ws = span[1]

            # Format data
            data.append(
                (
                    token,
                    text[start_ws:end_ws],
                    span,
                )
            )
        return Tokens(data, self.annotators)

def _normalize(text):
    return unicodedata.normalize("NFD", text)


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def has_answer(answers, doc):
    tokenizer = SimpleTokenizer()

    doc = tokenizer.tokenize(_normalize(doc)).words(uncased=True)
    for answer in answers:
        answer = tokenizer.tokenize(_normalize(answer)).words(uncased=True)
        for i in range(0, len(doc) - len(answer) + 1):
            if answer == doc[i:i + len(answer)]:
                return True


class NQTriviaQAUtils:
    def __init__(self, dataset):
        assert dataset in ["NQ", "TriviaQA"]
        self.root = root_path
        self.dataset = dataset
        if dataset == 'NQ':
            self.train_file = os.path.join(self.root, 'data/NQ/biencoder-nq-train.json')
            self.sampled_data_file = os.path.join(self.root, 'data/NQ/sampled_data.json')
        else:
            self.train_file = os.path.join(self.root, 'data/TriviaQA/biencoder-trivia-train.json')
            self.sampled_data_file = os.path.join(self.root, 'data/TriviaQA/sampled_data.json')

    def load_qs_list(self):
        data_list = json.load(open(self.sampled_data_file, 'r'))
        qs_list = [dict(qs_id=str(idx), question=item['question']) for idx, item in enumerate(data_list)]
        return qs_list

    def load_oracle_list(self):
        """
        each oracle paragraph contains the info to answer the question, so just pick one para that has answer as oracle
        :return:
        """
        data_list = json.load(open(self.sampled_data_file, 'r'))
        oracle_list = [dict(qs_id=str(idx), oracle_doc=item['oracle_doc'], answers=item['answers']) for idx, item in enumerate(data_list)]
        return oracle_list

    # def remove_no_oracle(self):
    #     qs_list = json.load(open(self.qs_file, 'r'))
    #     _qs_list = []
    #     for idx, qs in enumerate(qs_list):
    #         oracle_doc = None
    #         for doc in qs['ctxs']:
    #             if doc['has_answer']:
    #                 oracle_doc = doc['id']
    #                 break
    #         if oracle_doc is not None: _qs_list.append(qs)
    #
    #     with open(self.filtered_qs_file, 'w+') as f:
    #         json.dump(_qs_list, f, indent=2)

    # def sample_data(self, k=2000, sampled_data_file=None):
    #     """
    #     sample 2000 queries from train set
    #     :return:
    #     """
    #     has_positive_idx_list = []
    #     count = 0
    #     with open(self.train_file, 'rb') as f:
    #         for record in ijson.items(f, 'item'):
    #             if record['positive_ctxs'] != []:
    #                 has_positive_idx_list.append(count)
    #             count += 1
    #     random_idx_list = random.sample(has_positive_idx_list, k)
    #
    #     count = 0
    #     data_list = []
    #     with open(self.train_file, 'rb') as f:
    #         for record in ijson.items(f, 'item'):
    #             if count in random_idx_list:
    #                 assert record['positive_ctxs'] != []
    #                 if self.dataset == 'TriviaQA':
    #                     proc_record = dict(question=record['question'], answers=record['answers'], oracle_doc=record['positive_ctxs'][0]['psg_id'])
    #                 elif self.dataset == 'NQ':
    #                     proc_record = dict(question=record['question'], answers=record['answers'], oracle_doc=record['positive_ctxs'][0]['passage_id'])
    #                 data_list.append(proc_record)
    #             count += 1
    #     assert len(data_list) == k
    #
    #     if sampled_data_file is not None:
    #         with open(sampled_data_file, 'w+') as f:
    #             json.dump(data_list, f, indent=2)
    #
    #     return data_list

    # def if_has_answer(self, doc, qs_id):
    #     oracle_list = self.load_oracle_list()
    #     answers = None
    #     for oracle in oracle_list:
    #         if oracle['qs_id'] == qs_id:
    #             answers = oracle['answers']
    #     if answers is None:
    #         raise Exception(f'wrong qs_id: {qs_id}')
    #     # doc = _normalize(doc)
    #     # answers = [_normalize(answer) for answer in answers]
    #     return has_answer(answers, doc)

    @staticmethod
    def retrieval_eval(docs_list, answers_list, top_k):
        """
        follow DPR: if answer is in retrieval docs, then retrieval right, and randomly set one text as oracle
        :param docs_list: a list of docs, each one is a list of retrieved docs of a sample
        :return: hits_list: a list of list, each list corresponds to the retrieved docs of a sample
                hits_rate: a dict records the recall of top_k retrieval
        """
        from tqdm import tqdm
        hits_list = list()
        for docs, answers in tqdm(zip(docs_list, answers_list), total=len(docs_list)):
            # docs = wiki_loader.get_docs(doc_keys)
            hits = [has_answer(answers, doc) for doc in docs]
            hits_list.append(hits)

        hits_rate = dict()
        for k in top_k:
            hits_rate[k] = 0
            for hits in hits_list:
                if True in hits[:k]:
                    hits_rate[k] += 1
            hits_rate[k] = hits_rate[k] / len(hits_list)
        print(hits_rate)
        return hits_rate

    @staticmethod
    def pred_eval(preds, answers_list):
        """
        follow DPR and In-context RALM, if pred matches to any of the answers, then exact match
        :param preds:
        :return:
        """
        metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0, 'has_answer': 0}
        eval_records = dict()
        for idx, (pred, answers) in enumerate(zip(preds, answers_list)):
            eval_records[idx] = dict()
            if has_answer(answers, pred): eval_records[idx]['has_answer'] = True
            else: eval_records[idx]['has_answer'] = False
            em = 0
            max_precision, max_recall, max_f1 = 0, 0, 0
            for answer in answers:
                nor_pred, nor_answer = normalize_answer(pred), normalize_answer(answer)
                if nor_pred == nor_answer: em = 1
                pred_tokens, answer_tokens = nor_pred.split(), nor_answer.split()
                if len(answer_tokens) == 0 or len(pred_tokens) == 0: continue
                common = Counter(pred_tokens) & Counter(answer_tokens)
                num_same = sum(common.values())
                precision = num_same / len(pred_tokens)
                recall = num_same / len(answer_tokens)
                if recall > max_recall: # use answer with max recall to calc f1, prec, recall
                    max_recall = recall
                    max_precision = precision
                    max_f1 = 2 * precision * recall / (precision + recall)
            metrics['em'] += em
            metrics['f1'] += max_f1
            metrics['prec'] += max_precision
            metrics['recall'] += max_recall
            metrics['has_answer'] += 1 if eval_records[idx]['has_answer'] is True else 0
            eval_records[idx]['em'] = em
            eval_records[idx]['f1'] = max_f1
            eval_records[idx]['prec'] = max_precision
            eval_records[idx]['recall'] = max_recall
        for key in metrics.keys():
            metrics[key] /= len(preds)
        # print(metrics)
        return metrics, eval_records


if __name__ == '__main__':
    nq_loader = NQTriviaQAUtils('TriviaQA')
    nq_loader.sample_data()
    # data_list = []
    # with open(nq_loader.train_file, 'r') as f:
    #     for record in ijson.items(f, 'item'):
    #         data_list.append(proc_record)
    #         break
    # with open(nq_loader.sampled_data_file, 'w+') as f:
    #     json.dump(data_list, f, indent=2)
