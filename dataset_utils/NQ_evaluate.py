import platform
import sys
import re, string
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from dataset_utils.dataset_configs import NQLoader, WikiCorpusLoader


def _normalize_answer(s):
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


# top-1
def retrieval_eval(doc_key_list, answer_list):
    """
    follow DPR: if answer is in retrieval docs, then retrieval right, and randomly set one text as oracle
    :param doc_key_list: a list of doc keys, each one is a list of retrieved doc keys of a sample
    :return:
    """
    # get_docs get tons of doc key is much faster
    wiki_loader = WikiCorpusLoader()
    for doc_keys, answers in zip(doc_key_list, answer_list):
        docs = wiki_loader.get_docs(doc_keys)
        for item in docs:
            if has_answer(answers, item['doc']):





def pred_eval(pred):
    """
    follow DPR and In-context RALM, if pred matches to any of the answers, then exact match
    :param pred:
    :return:
    """