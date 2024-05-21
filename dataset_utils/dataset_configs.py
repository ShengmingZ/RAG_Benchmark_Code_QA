import json
import gzip
import csv
import unicodedata
from collections import Counter
import os
import bz2
import random
import platform
import sys
from typing import List
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from data.DS1000.ds1000 import DS1000Dataset

random.seed(0)




class TldrLoader:
    def __init__(self):
        self.root = root_path
        self.doc_whole_file = os.path.join(self.root, "data/tldr/manual_all_raw.json")
        self.doc_line_file = os.path.join(self.root, "data/tldr/manual_section.json")
        self.dev_qs_file = os.path.join(self.root, "data/tldr/cmd_dev.seed.json")
        self.dev_oracle_file = os.path.join(self.root, "data/tldr/cmd_dev.oracle_man.full.json")
        self.test_qs_file = os.path.join(self.root, "data/tldr/cmd_test.seed.json")
        self.test_oracle_file = os.path.join(self.root, "data/tldr/cmd_test.oracle_man.full.json")
        self.train_qs_file = os.path.join(self.root, "data/tldr/cmd_train.seed.json")
        self.train_oracle_file = os.path.join(self.root, "data/tldr/cmd_train.oracle_man.full.json")

    def load_doc_list_whole(self):
        """
        all docs should be in the format of {f'{doc_key}': content}
        """
        return json.load(open(self.doc_whole_file, 'r'))

    def load_doc_list_line(self):
        return json.load(open(self.doc_line_file, 'r'))

    def load_qs_list(self, dataset):
        """
        all elements in qs list should be in the format of {'nl': nl, 'qs_id': qs_id}
        """
        assert dataset in ['train', 'dev', 'test']
        if dataset == 'train':
            qs_list = json.load(open(self.train_qs_file, 'r'))
        elif dataset == 'dev':
            qs_list = json.load(open(self.dev_qs_file, 'r'))
        else:
            qs_list = json.load(open(self.test_qs_file, 'r'))
        qs_list = [dict(nl=qs['nl'], qs_id=qs['question_id']) for qs in qs_list]
        return qs_list

    def load_oracle_list(self, dataset):
        """
        {'qs_id': str, 'doc_keys': a list of keys of docs, 'output': output, 'line_keys': line level libs}
        """
        assert dataset in ['train', 'dev', 'test']
        if dataset == 'train':
            oracle_list = json.load(open(self.train_oracle_file, 'r'))
        elif dataset == 'dev':
            oracle_list = json.load(open(self.dev_oracle_file, 'r'))
        else:
            oracle_list = json.load(open(self.test_oracle_file, 'r'))
        oracle_list = [dict(qs_id=oracle['question_id'],
                            output=oracle['cmd'],
                            doc_keys=[oracle['cmd_name']],
                            line_keys=oracle['oracle_man']) for oracle in oracle_list]
        return oracle_list

    def remove_repeat(self):

        def remove(qs_list, oracle_list):
            _qs_list = list()
            _oracle_list = list()
            for qs, oracle in zip(qs_list, oracle_list):
                assert qs['question_id'] == oracle['question_id']
                is_repeat = False
                for _qs, _oracle in zip(_qs_list, _oracle_list):
                    if qs['question_id'] == _qs['question_id']:
                        is_repeat = True
                        break
                if is_repeat is False:
                    _qs_list.append(qs)
                    _oracle_list.append(oracle)
            return _qs_list, _oracle_list

        # train
        qs_list = json.load(open(self.train_qs_file, 'r'))
        oracle_list = json.load(open(self.train_oracle_file, 'r'))
        qs_list, oracle_list = remove(qs_list, oracle_list)
        with open(self.train_qs_file, 'w+') as f:
            json.dump(qs_list, f, indent=2)
        with open(self.train_oracle_file, 'w+') as f:
            json.dump(oracle_list, f, indent=2)
        print('train: ', len(qs_list))

        # test
        qs_list = json.load(open(self.test_qs_file, 'r'))
        oracle_list = json.load(open(self.test_oracle_file, 'r'))
        qs_list, oracle_list = remove(qs_list, oracle_list)
        with open(self.test_qs_file, 'w+') as f:
            json.dump(qs_list, f, indent=2)
        with open(self.test_oracle_file, 'w+') as f:
            json.dump(oracle_list, f, indent=2)
        print('test: ', len(qs_list))

        # dev
        qs_list = json.load(open(self.dev_qs_file, 'r'))
        oracle_list = json.load(open(self.dev_oracle_file, 'r'))
        qs_list, oracle_list = remove(qs_list, oracle_list)
        with open(self.dev_qs_file, 'w+') as f:
            json.dump(qs_list, f, indent=2)
        with open(self.dev_oracle_file, 'w+') as f:
            json.dump(oracle_list, f, indent=2)
        print('dev: ', len(qs_list))


# class HumanEvalLoader:
#     def __init__(self):
#         self.root = root_path
#         self.problems = read_problems()
#         # self.doc_file = os.path.join(self.root, "data/conala/conala_docs.json")
#
#     def load_qs_list(self, sampled=False):
#         """
#         all elements in qs list should be in the format of {'nl': nl, 'qs_id': qs_id}
#         """
#         qs_list = list()
#         for task_id in self.problems.keys():
#             qs_list.append(dict(nl=self.problems[task_id]['prompt'], qs_id=task_id))
#         return qs_list
#
#     # def load_doc_list(self, sampled=False):
#     #     """
#     #     all docs should be in the format of {f'{doc_key}': content}
#     #     """
#     #     return json.load(open(self.doc_file, 'r'))
#
#     def load_oracle_list(self, sampled=False):
#         """
#         {'qs_id': str, 'doc_keys': a list of libs, 'output': output}
#         """
#         oracle_list = []
#         for task_id in self.problems.keys():
#             oracle_list.append(dict(qs_id=task_id, output=self.problems[task_id]['canonical_solution']))
#         return oracle_list
#
#
# class MBPPLoader:
#     def __init__(self):
#         self.root = root_path
#         self.dataset = load_dataset('mbpp')
#         # self.doc_file = os.path.join(self.root, "data/conala/conala_docs.json")
#
#     def load_qs_list(self, dataset='test', sampled=False):
#         """
#         all elements in qs list should be in the format of {'nl': nl, 'qs_id': qs_id}
#         """
#         qs_list = list()
#         self.dataset['test']
#         for task_id in self.problems.keys():
#             qs_list.append(dict(nl=self.problems[task_id]['prompt'], qs_id=task_id))
#         return qs_list
#
#     # def load_doc_list(self, sampled=False):
#     #     """
#     #     all docs should be in the format of {f'{doc_key}': content}
#     #     """
#     #     return json.load(open(self.doc_file, 'r'))
#
#     def load_oracle_list(self, dataset='test', sampled=False):
#         """
#         {'qs_id': str, 'doc_keys': a list of libs, 'output': output}
#         """
#         oracle_list = []
#         for task_id in self.problems.keys():
#             oracle_list.append(dict(qs_id=task_id, output=self.problems[task_id]['canonical_solution']))
#         return oracle_list






if __name__ == '__main__':
    ...

    # tldr_dataloader = TldrLoader()
    # tldr_dataloader.remove_repeat()
    # conala_dataloader = ConalaLoader()
    # print(len(conala_dataloader.load_qs_list('dev')))
    # print(len(conala_dataloader.load_qs_list('test')))
    # ds1000_loader = DS1000Loader()
    # ds1000_loader.sample_dataset()
    # print(len(ds1000_loader.load_qs_list(sampled=True)))
    # humaneval_loader = HumanEvalLoader()
    # print(len(humaneval_loader.load_qs_list()))

    # wikicorpus_loader = WikiCorpusLoader()
    # wikicorpus_loader.process_wiki_corpus()

    # hotpotqa_loader = HotpotQALoader()
    # sampled_qs_list = hotpotqa_loader.get_sample(3)
    #
    # for qs in sampled_qs_list:
    #     print(qs['question'])
    #     print(qs['answer'])
    #     print(qs['supporting_facts'])
    #     print(qs['context'])

    # nq_loader = NQLoader()
    # nq_loader.remove_no_oracle()
    # oracle_list = nq_loader.load_oracle_list()


    """
    test wiki get_docs()
    """
    # hotpot_loader = HotpotQALoader()
    # oracle_list = hotpot_loader.load_oracle_list()
    # doc_key_list = [oracle['oracle_docs'] for oracle in oracle_list]
    #
    # wiki_loader = WikiCorpusLoader()
    # docs = wiki_loader.get_docs([oracle_list[1]['oracle_docs']])
    # print(docs[0][0])
    # print(docs[0][1])
    # print(docs[0][2])

    # hotpot_loader = HotpotQALoader()
    # hotpot_loader.sample_dataset()

    # test_doc_key_list = []
    # wiki_corpus_file = os.path.join(root_path, 'data/wikipedia/psgs_w100.tsv')
    # with open(wiki_corpus_file, 'r', newline='') as tsvfile:
    #     reader = csv.reader(tsvfile, delimiter='\t')
    #     for row in reader:
    #         if 'Medell' in row[1] and 'Cartel' in row[1]:
    #             print(row[1])
    #             test_doc_key_list.append(row[1])
    #             break

    # with open('temp_file.txt', 'r', encoding='utf-8') as f:
    #     temp_doc_key = f.readlines()[0].strip()
    # print(temp_doc_key)

    # for temp_char, char in zip(temp_doc_key, oracle_list[1]['oracle_docs'][0]):
    #     if temp_char != char:
    #         print(temp_char, char)
    # if unicodedata.normalize('NFD', temp_doc_key) == unicodedata.normalize('NFD', oracle_list[1]['oracle_docs'][0]):
    #     print('ok')