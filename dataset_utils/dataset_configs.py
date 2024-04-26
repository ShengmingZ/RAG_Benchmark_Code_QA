import json
import gzip
import csv
from collections import Counter
import os
import random
import platform
import sys
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from data.DS1000.ds1000 import DS1000Dataset

random.seed(0)


class WikiCorpusLoader:
    def __init__(self):
        if system == 'Darwin':
            self.wiki_corpus_file = os.path.join(root_path, 'data/wikipedia/psgs_w100.tsv')
        elif system == 'Linux':
            self.wiki_corpus_file = '/data/zhaoshengming/wikipedia/psgs_w100.tsv'
    def load_wiki_corpus_iter(self):
        with open(self.wiki_corpus_file, 'r', newline='') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                data = dict(id=row[1], text=row[2])
                yield data

    def load_wiki_corpus(self):
        data_list = list()
        with open(self.wiki_corpus_file, 'r', newline='') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                data_list.append(dict(id=row[1], text=row[2]))
        return data_list

    def load_wiki_id(self):
        data_list = list()
        with open(self.wiki_corpus_file, 'r', newline='') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                data_list.append(row[1])
        return data_list

    def get_docs(self, doc_key_list):
        docs = list()
        with open(self.wiki_corpus_file, 'r', newline='') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                if row[1] in doc_key_list:
                    docs.append(dict(doc_key=row[1], doc=row[2]))
        return docs

    def process_wiki_corpus(self):
        wiki_rec_file = self.wiki_corpus_file.replace('.tsv', '_rec.tsv')
        with open(wiki_rec_file, 'r', newline='') as infile, open(self.wiki_corpus_file, 'w', newline='') as outfile:
            reader = csv.reader(infile, delimiter='\t')
            writer = csv.writer(outfile, delimiter='\t')

            key = None
            for row in reader:
                if key is None or key != row[2]:
                    key = row[2]
                    key_count = 0
                else:
                    key_count += 1
                processed_doc_key = key + '_' + str(key_count)
                writer.writerow([row[0], processed_doc_key, row[1]])


class NQLoader:
    def __init__(self):
        self.root = root_path
        self.qs_file = os.path.join(self.root, 'data/NQ/nq-test.json')
        self.filtered_qs_file = os.path.join(self.root, 'data/NQ/nq-test-filtered.json')

    def load_qs_list(self):
        qs_list = json.load(open(self.filtered_qs_file, 'r'))
        _qs_list = [dict(qs_id=idx, question=item['question']) for idx, item in enumerate(qs_list)]
        return _qs_list

    def load_oracle_list(self):
        """
        each oracle paragraph contains the info to answer the question, so just pick one para that has answer as oracle
        :return:
        """
        qs_list = json.load(open(self.filtered_qs_file, 'r'))
        oracle_list = []
        for idx, qs in enumerate(qs_list):
            oracle_doc = None
            for doc in qs['ctxs']:
                if doc['has_answer']:
                    oracle_doc = doc['id']
                    break
            if oracle_doc is None: raise Exception(f'no oracle found in qs {qs["question"]}')
            oracle_list.append(dict(qs_id=idx, answers=qs['answers'], oracle_doc=oracle_doc))
        return oracle_list

    def remove_no_oracle(self):
        qs_list = json.load(open(self.qs_file, 'r'))
        _qs_list = []
        for idx, qs in enumerate(qs_list):
            oracle_doc = None
            for doc in qs['ctxs']:
                if doc['has_answer']:
                    oracle_doc = doc['id']
                    break
            if oracle_doc is not None: _qs_list.append(qs)

        with open(self.filtered_qs_file, 'w+') as f:
            json.dump(_qs_list, f, indent=2)


class HotpotQALoader:
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
        qs_list = json.load(open(self.sample_qs_file, 'r'))
        _qs_list = []
        for qs in qs_list:
            _qs_list.append(dict(qs_id=qs['_id'], oracle_docs=[sp[0]+'_'+str(sp[1]) for sp in qs['supporting_facts']], answer=qs['answer']))

        return _qs_list

    def sample_dataset(self):
        """
        sample 20%
        :return:
        """
        qs_list = json.load(open(self.qs_file, 'r'))
        num_sampled = int(len(qs_list)*0.2)
        problem_id_list = list(range(0, len(qs_list)))
        sampled_id_list = random.sample(problem_id_list, num_sampled)
        sampled_data = list()
        for id in sampled_id_list:
            sampled_data.append(qs_list[id])

        with open(os.path.join(self.root, 'data/hotpotQA/sampled_data.json'), 'w+') as f:
            json.dump(sampled_data, f, indent=2)

    def get_sample(self, sample_num):
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



class ConalaLoader:
    def __init__(self):
        self.root = root_path
        # self.qs_file = os.path.join(self.root, "data/conala/conala_nl.txt")
        # self.qs_idx_file = self.qs_file.replace(".txt", ".id")
        # self.train_qs_file = os.path.join(self.root, "data/conala/train_qs.json")
        # self.test_qs_file = os.path.join(self.root, "data/conala/test_qs.json")
        # self.dev_qs_file = os.path.join(self.root, "data/conala/dev_qs.json")
        # self.doc_firstpara_file = os.path.join(self.root, "data/conala/python_manual_firstpara.tok.txt")
        # self.doc_firstpara_idx_file = self.doc_firstpara_file.replace(".txt", ".id")
        # self.doc_file = os.path.join(self.root, "data/conala/conala_docs.json")
        # self.test_oracle_file = os.path.join(self.root, "data/conala/cmd_test.oracle_man.full.json")
        # self.train_oracle_file = os.path.join(self.root, "data/conala/cmd_train.oracle_man.full.json")
        # self.dev_oracle_file = os.path.join(self.root, "data/conala/cmd_dev.oracle_man.full.json")
        self.unittest_file = os.path.join(self.root, "data/conala/unittest_docprompting_conala.json")

    # def load_doc_list(self):
    #     """
    #     all docs should be in the format of {f'{doc_key}': content}
    #     """
    #     return json.load(open(self.doc_file, 'r'))

    # def load_doc_list_firstpara(self):
    #     doc_firstpara_content_list = list()
    #     with open(self.doc_firstpara_file, 'r') as f:
    #         for line in f:
    #             doc_firstpara_content_list.append(line.strip())
    #     doc_firstpara_id_list = list()
    #     with open(self.doc_firstpara_idx_file, 'r') as f:
    #         for line in f:
    #             doc_firstpara_id_list.append(line.strip())
    #     assert len(doc_firstpara_content_list) == len(doc_firstpara_id_list)
    #     doc_list_firstpara = dict()
    #     for doc_firstpara_id, doc_firstpara_content in zip(doc_firstpara_id_list, doc_firstpara_content_list):
    #         doc_list_firstpara[doc_firstpara_id] = doc_firstpara_content
    #     return doc_list_firstpara

    def load_qs_list(self):
        """
        all elements in qs list should be in the format of {'nl': nl, 'qs_id': qs_id}
        """
        dataset = json.load(open(self.unittest_file, 'r'))
        qs_list = []
        for data in dataset.items():
            qs_list.append(dict(qs_id=data['task_id'], nl=data['intent']))
        return qs_list

    def load_oracle_list(self):
        """
        {'qs_id': str, 'doc_keys': a list of libs, 'output': output}
        """
        dataset = json.load(open(self.unittest_file, 'r'))
        oracle_list = []
        for data in dataset.values():
            oracle_list.append(dict(qs_id=data['task_id'], output=data['canonical_solution']))
        return oracle_list

    def split_qs(self):
        train_oracle = json.load(open(self.train_oracle_file, 'r'))
        dev_oracle = json.load(open(self.dev_oracle_file, 'r'))
        test_oracle = json.load(open(self.test_oracle_file, 'r'))
        qs_content_list = []
        with open(self.qs_file, 'r') as f:
            for line in f:
                qs_content_list.append(line.strip())
        qs_id_list = []
        with open(self.qs_idx_file, 'r') as f:
            for line in f:
                qs_id_list.append(line.strip())
        assert len(qs_content_list) == len(qs_id_list)
        assert len(train_oracle)+len(dev_oracle)+len(test_oracle) == len(qs_id_list)
        train_idx, dev_idx, test_idx = 0, 0, 0
        train_qs_list, dev_qs_list, test_qs_list = [], [], []
        for qs_content, qs_id in zip(qs_content_list, qs_id_list):
            if train_idx < len(train_oracle) and qs_id == train_oracle[train_idx]['question_id']:
                assert qs_content == train_oracle[train_idx]['nl'].replace('\r','').replace('\n','')
                train_qs_list.append(dict(nl=qs_content, qs_id=qs_id))
                train_idx += 1
            elif dev_idx < len(dev_oracle) and qs_id == dev_oracle[dev_idx]['question_id']:
                assert qs_content == dev_oracle[dev_idx]['nl'].replace('\r','').replace('\n','')
                dev_qs_list.append(dict(nl=qs_content, qs_id=qs_id))
                dev_idx += 1
            elif test_idx < len(test_oracle) and qs_id == test_oracle[test_idx]['question_id']:
                assert qs_content == test_oracle[test_idx]['nl'].replace('\r','').replace('\n','')
                test_qs_list.append(dict(nl=qs_content, qs_id=qs_id))
                test_idx += 1
            else:
                raise Exception(f'Unexpected question id: {qs_id}')

        assert len(train_qs_list) == len(train_oracle)
        assert len(test_qs_list) == len(test_oracle)
        assert len(dev_qs_list) == len(dev_oracle)
        with open(self.train_qs_file, 'w+') as f:
            json.dump(train_qs_list, f, indent=2)
        with open(self.dev_qs_file, 'w+') as f:
            json.dump(dev_qs_list, f, indent=2)
        with open(self.test_qs_file, 'w') as f:
            json.dump(test_qs_list, f, indent=2)

    def remove_repeat(self):
        def remove(qs_list, oracle_list):
            _qs_list = list()
            _oracle_list = list()
            for qs, oracle in zip(qs_list, oracle_list):
                assert qs['qs_id'] == oracle['question_id']
                is_repeat = False
                for _qs, _oracle in zip(_qs_list, _oracle_list):
                    if qs['qs_id'] == _qs['qs_id']:
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


class DS1000Loader:
    def __init__(self):
        self.root = root_path
        # self.ds1000 = DS1000Dataset(source_dir=os.path.join(self.root, 'data/DS1000/ds1000_data'), libs='all', mode='Completion')
        # self.sampled_idx_file = os.path.join(self.root, 'data/DS1000/sampled_idx.json')
        self.sampled_data_file = os.path.join(self.root, 'data/DS1000/sampled_data.json')
        self.oracle_doc_file = os.path.join(self.root, 'data/DS1000/oracle_docs_matched.json')
        self.doc_file = os.path.join(self.root, "data/conala/conala_docs.json")

    def load_qs_list(self):
        """
        all elements in qs list should be in the format of {'nl': nl, 'qs_id': qs_id}
        """
        sampled_data = json.load(open(self.sampled_data_file, 'r'))
        qs_list = []
        for data in sampled_data:
            qs_list.append(dict(nl=data['prompt'], qs_id=data['qs_id']))
        return qs_list

    # def load_doc_list(self):
    #     """
    #     all docs should be in the format of {f'{doc_key}': content}
    #     """
    #     return json.load(open(self.doc_file, 'r'))

    def load_oracle_list(self):
        """
        {'qs_id': str, 'doc_keys': a list of libs, 'output': output}
        """
        return json.load(open(self.oracle_doc_file))

    # sample 20%
    def sample_dataset(self):
        import data.DS1000.ds1000
        import random

        random.seed(0)

        ds1000 = DS1000Dataset(source_dir='../data/DS1000/ds1000_data', libs='all', mode='Completion')

        sampled_idx_dict = dict()
        sampled_data_list = list()
        for lib in ds1000.libs:
            num_sampled = int(len(ds1000[lib]) * 0.2)
            problem_id_list = list(range(0, len(ds1000[lib])))
            sampled_idx = random.sample(problem_id_list, num_sampled)
            sampled_idx_dict[lib] = sampled_idx
            print(len(sampled_idx))
            for idx in sampled_idx:
                sampled_data = ds1000[lib][idx]
                sampled_data_list.append(dict(qs_id=f'{lib}_{idx}',
                                              reference_code=sampled_data['reference_code'],
                                              code_context=sampled_data['code_context'],
                                              prompt=sampled_data['prompt']))

        with open('../data/DS1000/sampled_data.json', 'w+') as f:
            json.dump(sampled_data_list, f, indent=2)


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


class PandasNumpyEvalLoader:
    def __init__(self):
        self.root = root_path
        self.pandas_eval_file = os.path.join(self.root, 'data/pandas-numpy-eval/data/PandasEval.jsonl.gz')
        self.numpy_eval_file = self.pandas_eval_file.replace('PandasEval', 'NumpyEval')
        self.oracle_docs_matched_file = os.path.join(self.root, 'data/pandas-numpy-eval/data/oracle_docs_matched.json')
        self.pandas_eval_data = list()
        with gzip.open(self.pandas_eval_file, 'rt') as f:
            for line in f:
                self.pandas_eval_data.append(json.loads(line))

        self.numpy_eval_data = list()
        with gzip.open(self.numpy_eval_file, 'rt') as f:
            for line in f:
                self.numpy_eval_data.append(json.loads(line))

    def load_qs_list(self, sampled=False):
        """
        all elements in qs list should be in the format of {'nl': nl, 'qs_id': qs_id}
        """
        qs_list = list()
        for data in self.pandas_eval_data:
            qs_list.append(dict(qs_id=data['task_id'], nl=data['prompt']))
        for data in self.numpy_eval_data:
            qs_list.append(dict(qs_id=data['task_id'], nl=data['prompt']))
        return qs_list

    def load_oracle_list(self, sampled=False):
        """
        {'qs_id': str, 'doc_keys': a list of libs, 'output': output}
        """
        # oracle_list = []
        # for data in self.pandas_eval_data:
        #     oracle_list.append(dict(qs_id=data['task_id'], outputs=data['canonical_solution']))
        # for data in self.numpy_eval_data:
        #     oracle_list.append(dict(qs_id=data['task_id'], outputs=data['canonical_solution']))
        # return oracle_list
        return json.load(open(self.oracle_docs_matched_file, 'r'))



if __name__ == '__main__':
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

    nq_loader = NQLoader()
    nq_loader.remove_no_oracle()
    oracle_list = nq_loader.load_oracle_list()