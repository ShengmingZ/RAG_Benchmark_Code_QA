import json
from collections import Counter
import os
import platform
import sys
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
sys.path.insert(0, root_path + '/DS-1000')
from abc import ABC, abstractmethod
from ds1000 import DS1000Dataset




class TldrLoader:
    def __init__(self):
        self.root = root_path
        self.doc_whole_file = os.path.join(self.root, "docprompting_data/tldr/manual_all_raw.json")
        self.doc_line_file = os.path.join(self.root, "docprompting_data/tldr/manual_section.json")
        self.dev_qs_file = os.path.join(self.root, "docprompting_data/tldr/cmd_dev.seed.json")
        self.dev_oracle_file = os.path.join(self.root, "docprompting_data/tldr/cmd_dev.oracle_man.full.json")
        self.test_qs_file = os.path.join(self.root, "docprompting_data/tldr/cmd_test.seed.json")
        self.test_oracle_file = os.path.join(self.root, "docprompting_data/tldr/cmd_test.oracle_man.full.json")
        self.train_qs_file = os.path.join(self.root, "docprompting_data/tldr/cmd_train.seed.json")
        self.train_oracle_file = os.path.join(self.root, "docprompting_data/tldr/cmd_train.oracle_man.full.json")

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
        self.qs_file = os.path.join(self.root, "docprompting_data/conala/conala_nl.txt")
        self.qs_idx_file = self.qs_file.replace(".txt", ".id")
        self.train_qs_file = os.path.join(self.root, "docprompting_data/conala/train_qs.json")
        self.test_qs_file = os.path.join(self.root, "docprompting_data/conala/test_qs.json")
        self.dev_qs_file = os.path.join(self.root, "docprompting_data/conala/dev_qs.json")
        self.doc_firstpara_file = os.path.join(self.root, "docprompting_data/conala/python_manual_firstpara.tok.txt")
        self.doc_firstpara_idx_file = self.doc_firstpara_file.replace(".txt", ".id")
        self.doc_file = os.path.join(self.root, "docprompting_data/conala/conala_docs.json")
        self.test_oracle_file = os.path.join(self.root, "docprompting_data/conala/cmd_test.oracle_man.full.json")
        self.train_oracle_file = os.path.join(self.root, "docprompting_data/conala/cmd_train.oracle_man.full.json")
        self.dev_oracle_file = os.path.join(self.root, "docprompting_data/conala/cmd_dev.oracle_man.full.json")
        self.unittest_file = os.path.join(self.root, "docprompting_data/conala/unittest_docprompting_conala.json")

    def load_doc_list(self):
        """
        all docs should be in the format of {f'{doc_key}': content}
        """
        return json.load(open(self.doc_file, 'r'))

    def load_doc_list_firstpara(self):
        doc_firstpara_content_list = list()
        with open(self.doc_firstpara_file, 'r') as f:
            for line in f:
                doc_firstpara_content_list.append(line.strip())
        doc_firstpara_id_list = list()
        with open(self.doc_firstpara_idx_file, 'r') as f:
            for line in f:
                doc_firstpara_id_list.append(line.strip())
        assert len(doc_firstpara_content_list) == len(doc_firstpara_id_list)
        doc_list_firstpara = dict()
        for doc_firstpara_id, doc_firstpara_content in zip(doc_firstpara_id_list, doc_firstpara_content_list):
            doc_list_firstpara[doc_firstpara_id] = doc_firstpara_content
        return doc_list_firstpara

    def load_qs_list(self, dataset):
        """
        all elements in qs list should be in the format of {'nl': nl, 'qs_id': qs_id}
        """
        assert dataset in ['train', 'dev', 'test']
        # # load target qs id list for train, dev, test
        # if dataset == 'train':
        #     oracle_list = json.load(open(self.train_oracle_file, 'r'))
        # elif dataset == 'dev':
        #     oracle_list = json.load(open(self.dev_oracle_file, 'r'))
        # else:
        #     oracle_list = json.load(open(self.test_oracle_file, 'r'))
        # target_qs_id_list = [item['question_id'] for item in oracle_list]
        # # assemble qs list
        # qs_content_list = []
        # with open(self.qs_file, 'r') as f:
        #     for line in f:
        #         qs_content_list.append(line.strip())
        # qs_id_list = []
        # with open(self.qs_idx_file, 'r') as f:
        #     for line in f:
        #         qs_id_list.append(line.strip())
        # qs_list = []
        # for id in target_qs_id_list:
        #     for idx in range(len(qs_id_list)):
        #         if qs_id_list[idx] == id:
        #             qs_list.append(dict(nl=qs_content_list[idx], qs_id=qs_id_list[idx]))
        # return qs_list
        if dataset == 'train':
            qs_list = json.load(open(self.train_qs_file, 'r'))
        elif dataset == 'dev':
            qs_list = json.load(open(self.dev_qs_file, 'r'))
        else:
            qs_list = json.load(open(self.test_qs_file, 'r'))
        return qs_list

    def load_oracle_list(self, dataset):
        """
        {'qs_id': str, 'doc_keys': a list of libs, 'output': output}
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
                            doc_keys=oracle['oracle_man']) for oracle in oracle_list]
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
        self.ds1000 = DS1000Dataset(source_dir=os.path.join(self.root, 'DS-1000/ds1000_data'), libs='all', mode='completion')
        self.doc_file = os.path.join(self.root, "docprompting_data/conala/conala_docs.json")

    def load_qs_list(self):
        """
        all elements in qs list should be in the format of {'nl': nl, 'qs_id': qs_id}
        """
        qs_list = []
        for lib in self.ds1000.libs:
            for idx in range(len(self.ds1000[lib])):
                qs_id = lib + '_' + str(idx)
                prompt = self.ds1000[lib][idx]['prompt']
                qs_list.append(dict(qs_id=qs_id, nl=prompt))
        return qs_list

    def load_doc_list(self):
        """
        all docs should be in the format of {f'{doc_key}': content}
        """
        return json.load(open(self.doc_file, 'r'))

    def load_oracle_list(self):
        """
        {'qs_id': str, 'doc_keys': a list of libs, 'output': output}
        """
        oracle_list = []
        for lib in self.ds1000.libs:
            for idx in range(len(self.ds1000[lib])):
                qs_id = lib + '_' + str(idx)
                output = self.ds1000[lib][idx]['reference_code']
                oracle_list.append(dict(qs_id=qs_id, output=output))
        return oracle_list


if __name__ == '__main__':
    # tldr_dataloader = TldrLoader()
    # tldr_dataloader.remove_repeat()
    conala_dataloader = ConalaLoader()
    print(len(conala_dataloader.load_qs_list('dev')))
    print(len(conala_dataloader.load_qs_list('test')))
