import json
import os
import bz2
import random
import platform
import sys
import evaluate
from tqdm import tqdm
import random
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

random.seed(0)


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
        self.oracle_doc_file = os.path.join(self.root, "data/conala/oracle_docs_matched_processed.json")

    def load_qs_list(self):
        """
        all elements in qs list should be in the format of {'nl': nl, 'qs_id': qs_id}
        """
        dataset = json.load(open(self.unittest_file, 'r'))
        qs_list = []
        for data in dataset.items():
            qs_list.append(dict(qs_id=data['task_id'], question=data['intent']))
        return qs_list

    def load_oracle_list(self):
        """
        {'qs_id': str, 'doc_keys': a list of libs, 'output': output}
        """
        dataset = json.load(open(self.oracle_doc_file, 'r'))
        oracle_list = []
        for data in dataset.values():
            oracle_list.append(dict(qs_id=data['qs_id'], output=data['output'], oracle_docs=data['oracle_docs']))
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


    def eval_passk(self, results, top_k, num_proc=16):
        unittests = json.load(open(self.unittest_file, 'r'))
        code_eval_metric = evaluate.load("code_eval")
        pass_k_list = []
        for result in results:
            for idx, qs_id in tqdm(enumerate(unittests.keys())):
                unittest = unittests[qs_id]
                if result['qs_id'] == qs_id:
                    break
            assert result['qs_id'] == qs_id
            suffix = unittest['suffix']
            entry_point = unittest["entry_point"]
            test_func = f"\n{unittest['test']}\ncheck({entry_point})"
            runnable_func = [f"{unittest['prompt']}{x}{suffix}" for x in result['output']]
            # runnable_func = [f"{unittest['prompt']}{unittest['canonical_solution']}{suffix}"] # oracle

            pass_k, _ = code_eval_metric.compute(
                predictions=[runnable_func],
                references=[test_func],
                k=top_k,
                num_workers=num_proc,
            )
            # print(idx, pass_k)
            pass_k_list.append(pass_k)
        _pass_k = {}
        pass_keys = list(pass_k_list[0].keys())
        for key in pass_keys: _pass_k[key] = 0
        for pass_k in pass_k_list:
            for key in pass_keys: _pass_k[key] += pass_k[key]
        for key in pass_keys: _pass_k[key] = _pass_k[key] / len(unittests)
        print(_pass_k)
        return _pass_k


if __name__ == '__main__':
    loader = ConalaLoader()
    dataset = json.load(open(loader.unittest_file, 'r'))
    oracle_list = json.load(open(loader.oracle_doc_file, 'r'))
    # for idx, oracle in enumerate(oracle_list):
    #     count = 0
    #     for qs_id, data in dataset.items():
    #         if oracle['output'] == data['canonical_solution']:
    #             oracle_list[idx]['qs_id'] = qs_id
    #             count += 1
    #     if count != 1: print(f'error for {oracle}')
    #
    # with open(loader.oracle_doc_file, 'w+') as f:
    #     json.dump(oracle_list, f, indent=2)

    preds = [dict(qs_id=oracle['qs_id'], output=oracle['output']) for oracle in oracle_list]
    loader.eval_passk(preds, [1])
