import json
import gzip
import csv
import re
import unicodedata
import os
import bz2
import random
import platform
import sys
from multiprocessing import Pool
from typing import List
from tqdm import tqdm
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)

random.seed(0)


class PythonDocsLoader:
    def __init__(self):
        self.root = root_path
        self.api_doc_builtin = os.path.join(root_path, 'data/python_docs/api_doc_builtin.json')
        self.api_sign_builtin = os.path.join(root_path, 'data/python_docs/api_sign_builtin.txt')
        self.api_doc_third_party = self.api_doc_builtin.replace('builtin', 'third_party')
        self.api_sign_third_party = self.api_sign_builtin.replace('builtin', 'third_party')
        self.proc_corpus_file = os.path.join(self.root, 'data/python_docs/proc_python_docs.json')

    def load_api_signs(self):
        python_docs = json.load(open(self.proc_corpus_file, 'r'))
        python_doc_id_list = []
        for item in python_docs:
            python_doc_id_list.append(item['api_sign'])
        return python_doc_id_list

    def load_api_docs(self):
        python_docs = json.load(open(self.proc_corpus_file, 'r'))
        return python_docs

    def get_docs(self, api_signs):
        python_docs = json.load(open(self.proc_corpus_file, 'r'))
        docs = ['']*len(api_signs)
        for item in python_docs:
            for idx, api_sign in enumerate(api_signs):
                if api_sign in item['api_sign']:
                    docs[idx] = item['doc']
        for idx, doc in enumerate(docs):
            if doc == '':
                print(f'not find doc for api sign {api_signs[idx]}')
        return docs


    def get_random_docs(self, k):
        corpus_ids = self.load_api_signs()
        random_doc_keys = random.sample(corpus_ids, k)
        docs = self.get_docs(random_doc_keys)
        return docs


    def process_docs(self):
        python_docs = dict()
        python_docs_third = json.load(open(self.api_doc_third_party, 'r'))
        python_docs_builtin = json.load(open(self.api_doc_builtin, 'r'))
        python_docs.update(python_docs_third)
        python_docs.update(python_docs_builtin)
        print(len(python_docs))

        proc_python_docs = list()
        for api_sign, doc in python_docs.items():
            lines = doc.split('\n')
            prefix = lines[0]
            function_head = re.sub(r'self[^,]*?:[^,]+?, ', '', lines[2])
            function_head = (function_head.replace('self, ', '').replace('self', '').
                             replace('...', '').replace(', /', '').replace('/, ', ''))
            try:
                function_head = function_head[:re.search(r'\(.*\)', function_head).end()]
            except:
                ...
            main_content = function_head + '\n' + '\n'.join(lines[3:])
            # main_content = doc
            is_match = False
            for item in proc_python_docs:
                if item['doc'] == main_content:
                    item['api_sign'].append(api_sign)
                    is_match = True
            if is_match is False:
                proc_python_docs.append(dict(api_sign=[api_sign], doc=main_content))
        print(len(proc_python_docs))
        with open(self.proc_corpus_file, 'w+') as f:
            json.dump(proc_python_docs, f, indent=2)


# if __name__ == '__main__':
#     python_docs_loader = PythonDocsLoader()
#     python_docs_loader.process_docs()

class WikiCorpusLoader:
    def __init__(self):
        if system == 'Darwin':
            self.wiki_corpus_file_NQ = os.path.join(root_path, 'data/wikipedia/psgs_w100.tsv')
            self.wiki_corpus_file_hotpot = os.path.join(root_path, 'data/wikipedia/enwiki-20171001-pages-meta-current-withlinks-abstracts')
            self.index_offsets_file = os.path.join(root_path, 'data/wikipedia/index_offsets.txt')
        elif system == 'Linux':
            self.wiki_corpus_file_NQ = '/data/zhaoshengming/wikipedia/psgs_w100.tsv'
            self.wiki_corpus_file_hotpot = '/data/zhaoshengming/wikipedia/enwiki-20171001-pages-meta-current-withlinks-abstracts'
            self.index_offsets_file = '/data/zhaoshengming/wikipedia/index_offsets.txt'

    def _get_hotpot_corpus_file_paths(self):
        file_paths = []
        dir_names = sorted(os.listdir(self.wiki_corpus_file_hotpot))
        for dir_name in dir_names:
            file_names = sorted(os.listdir(os.path.join(self.wiki_corpus_file_hotpot, dir_name)))
            file_paths.extend([os.path.join(self.wiki_corpus_file_hotpot, dir_name, file_name) for file_name in file_names])
        return file_paths

    @staticmethod
    def _load_data_from_bz2(file_path):
        with bz2.open(file_path, 'rt') as f:
            contents = f.read().split('\n')
            for content in contents:
                if content != '':
                    data = json.loads(content)
                    return data

    def load_wiki_corpus_iter(self, dataset):
        assert dataset in ['hotpotQA', 'TriviaQA', 'NQ']
        if dataset == 'hotpotQA':
            file_paths = self._get_hotpot_corpus_file_paths()
            for file_path in file_paths:
                data = self._load_data_from_bz2(file_path)
                yield dict(id=data['title'], text=''.join(data['text']))
        elif dataset in ['TriviaQA', 'NQ']:
            with open(self.wiki_corpus_file_NQ, 'r', newline='') as tsvfile:
                reader = csv.reader(tsvfile, delimiter='\t')
                for row in reader:
                    data = dict(id=row[0], text=row[1])
                    yield data

    def load_wiki_corpus(self, dataset):
        assert dataset in ['hotpotQA', 'TriviaQA', 'NQ']
        data_list = list()
        if dataset == 'hotpotQA':
            file_paths = self._get_hotpot_corpus_file_paths()
            for file_path in file_paths:
                data = self._load_data_from_bz2(file_path)
                data_list.append(dict(id=data['title'], text=''.join(data['text'])))
        elif dataset in ['TriviaQA', 'NQ']:
            with open(self.wiki_corpus_file_NQ, 'r', newline='') as tsvfile:
                reader = csv.reader(tsvfile, delimiter='\t')
                for row in reader:
                    data_list.append(dict(id=row[0], text=row[1]))
        print('done load WikiCorpus')
        return data_list

    def load_wiki_id(self, dataset):
        assert dataset in ['hotpotQA', 'TriviaQA', 'NQ']
        id_list = list()
        if dataset == 'hotpotQA':
            file_paths = self._get_hotpot_corpus_file_paths()
            for file_path in file_paths:
                data = self._load_data_from_bz2(file_path)
                id_list.append(data['title'])
        elif dataset in ['TriviaQA', 'NQ']:
            with open(self.wiki_corpus_file_NQ, 'r', newline='') as tsvfile:
                reader = csv.reader(tsvfile, delimiter='\t')
                for row in reader:
                    id_list.append(row[0])
        return id_list

    def _get_docs_hotpot(self, doc_keys):
        doc_keys_placeholder = [[idx, key] for idx, key in enumerate(doc_keys)]
        docs = [None]*len(doc_keys)
        file_paths = self._get_hotpot_corpus_file_paths()
        for file_path in tqdm(file_paths, total=len(file_paths)):
            with bz2.open(file_path, 'rt') as f:
                contents = f.read().split('\n')
                for content in contents:
                    if content != '':
                        data = json.loads(content)
                        target_key = unicodedata.normalize('NFD', data['title'])
                        for item in doc_keys_placeholder:
                            if item[1] == target_key:
                                docs[item[0]] = data['text']
                                doc_keys_placeholder.remove(item)
                                if len(doc_keys_placeholder) == 0: return docs
                                break
        raise Exception(f'not fully match for doc_keys: {doc_keys}')

    def _get_docs_NQ(self, doc_keys):
        # build index
        if not os.path.exists(self.index_offsets_file):
            index_offsets = []
            current_offset = 0
            with open(self.wiki_corpus_file_NQ, 'rb') as index_file:
                for line in index_file:
                    index_offsets.append(current_offset)
                    current_offset = current_offset + len(line) + 1
            with open(self.index_offsets_file, 'w+') as f:
                for item in index_offsets:
                    f.write(f"{item}\n")
        else:
            index_offsets = []
            with open(self.index_offsets_file, 'r') as f:
                for line in f:
                    index_offsets.append(int(line.strip()))

        # get docs by bytes offset
        doc_keys_placeholder = [[idx, key] for idx, key in enumerate(doc_keys)]
        doc_keys_placeholder.sort(key=lambda x:x[1], reverse=False)
        docs = [None]*len(doc_keys)
        with open(self.wiki_corpus_file_NQ, 'rb') as f:
            for item in doc_keys_placeholder:
                f.seek(index_offsets[item[1]])
                row = f.read(index_offsets[item[1]+1] - index_offsets[item[1]])
                row = row.decode('utf-8')
                docs[item[0]] = row.split('\t')[1][1:-1]
        return docs

        # start_time = time.time()
        # with open(self.wiki_corpus_file_NQ, 'r', newline='') as tsvfile:
        #     reader = csv.reader(tsvfile, delimiter='\t')
        #
        #     current_place = 0
        #     for item in doc_keys_placeholder:
        #         skip = item[1] - current_place
        #         for _ in range(skip):
        #             next(reader)
        #         for row in reader:
        #             docs[item[0]] = row[1]
        #             break
        #         current_place = current_place + skip + 1
        # print('traverse duration: ', time.time() - start_time)
        # return docs

        #     next(reader)
        #     for row in reader:
        #         if int(row[0]) == doc_keys_placeholder[0][1]:
        #             docs[doc_keys_placeholder[0][0]] = row[1]
        #             doc_keys_placeholder.pop(0)
        #             if len(doc_keys_placeholder) == 0: return docs
        # raise Exception(f'not fully match for doc_keys: {doc_keys_placeholder}')

    def get_docs(self, doc_keys_list: List[List[str]], dataset, num_procs=16):
        """
        given doc key, return corresponding doc, each element in doc_key_list is a list of doc keys of a sample
        :param doc_keys_list:
        :return:
        """
        assert dataset in ['hotpotQA', 'TriviaQA', 'NQ']
        docs_list = [None]*len(doc_keys_list)
        if dataset == 'hotpotQA':
            # normalize doc key
            _doc_keys_list = list()
            for doc_keys in doc_keys_list:
                _doc_keys_list.append([unicodedata.normalize('NFD', key) for key in doc_keys])
            doc_keys_list = _doc_keys_list
            with Pool(num_procs) as pool:
                for sample_idx, docs in tqdm(enumerate(pool.imap(self._get_docs_hotpot, doc_keys_list)), total=len(doc_keys_list)):
                    docs = [unicodedata.normalize('NFD', doc) for doc in docs]
                    docs_list[sample_idx] = docs
        elif dataset in ['TriviaQA', 'NQ']:
            _doc_keys_list = list()
            for doc_keys in doc_keys_list:
                _doc_keys_list.append([int(key) for key in doc_keys])
            doc_keys_list = _doc_keys_list
            with Pool(num_procs) as pool:
                for sample_idx, docs in tqdm(enumerate(pool.imap(self._get_docs_NQ, doc_keys_list)), total=len(doc_keys_list)):
                    # print(f'{sample_idx} completed')
                    docs = [unicodedata.normalize('NFD', doc) for doc in docs]
                    docs_list[sample_idx] = docs

        return docs_list

    def get_random_docs(self, k):
        data_count = 21015325
        doc_idxes = random.sample(range(data_count), k)
        doc_keys = [str(idx) for idx in doc_idxes]
        docs = self.get_docs([doc_keys], 'NQ')[0]
        return docs

    # def process_wiki_corpus(self):
    #     wiki_rec_file = self.wiki_corpus_file.replace('.tsv', '_rec.tsv')
    #     with open(wiki_rec_file, 'r', newline='') as infile, open(self.wiki_corpus_file, 'w', newline='') as outfile:
    #         reader = csv.reader(infile, delimiter='\t')
    #         writer = csv.writer(outfile, delimiter='\t')
    #         key = None
    #         for row in reader:
    #             if key is None or key != row[2]:
    #                 key = row[2]
    #                 key_count = 0
    #             else:
    #                 key_count += 1
    #             processed_doc_key = key + '_' + str(key_count)
    #             writer.writerow([row[0], processed_doc_key, row[1]])


if __name__ == '__main__':
    loader = WikiCorpusLoader()
    doc_keys = [["Thus Spoke Zarathustra"]]
    docs = loader.get_docs(doc_keys, 'hotpotQA', num_procs=8)
    print(docs)
