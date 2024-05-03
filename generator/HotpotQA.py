import json
import random
from tqdm import tqdm
import platform
import sys
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from generator.run_model import chatgpt
from prompt import conala_prompt
from dataset_utils.dataset_configs import HotpotQALoader, WikiCorpusLoader
from retriever.sparse_retriever import sparse_retriever_config
from retriever.dense_retriever import dense_retriever_config
from generator.generate_utils import (approximate_token, generate_config, save_results_to_files,
                                      control_ret_acc, perturb_ret_doc_type,
                                      process_retrieval_doc, apply_prompt_method)
from retriever.retriever_utils import get_ret_results
from dataset_utils.hotpot_evaluate_v1 import eval_pred

class GeneHotpotQA:
    def __init__(self, args):
        # load parameters
        self.dataset = args.dataset
        self.save_file = args.save_file
        self.analysis_type = args.analysis_type
        self.dataset_type = 'nlp'

        self.model = args.model
        if 'gpt' in self.model:
            self.model_type = 'gpt'
        elif 'llama' in self.model:
            self.model_type = 'llama'
        self.temperature = args.temperature
        self.n = args.n
        self.max_tokens = args.max_tokens

        self.retriever = args.retriever
        self.ret_info_type = args.ret_info_type
        self.ret_acc = args.ret_acc

        self.top_k = args.top_k
        self.doc_max_length = args.doc_max_length
        self.prompt_type = args.prompt_type

        self.hotpotqa_loader = HotpotQALoader()
        self.qs_list = self.hotpotqa_loader.load_qs_list()
        self.oracle_list = self.hotpotqa_loader.load_oracle_list()
        self.wiki_loader = WikiCorpusLoader()
        self.ret_results = get_ret_results(self.dataset, self.retriever)

        print('qs_num:', len(self.qs_list))
        print('save_to:', self.save_file)

    def gene_response(self):
        # perturb retrieval files
        if self.ret_info_type == 'oracle':
            ret_doc_key_list, ret_docs = control_ret_acc(self.ret_acc, [oracle["oracle_docs"] for oracle in self.oracle_list], self.dataset_type)
        elif self.ret_acc == 1 and self.ret_info_type != 'oracle':
            ret_doc_key_list = []
            for idx, (qs_id, result) in enumerate(self.ret_results.items()):
                assert qs_id == self.oracle_list[idx]['qs_id']
                ret_doc_key_list.append([item['doc_key'] for item in result])
            ret_doc_key_list, ret_docs = perturb_ret_doc_type(self.ret_info_type, ret_doc_key_list, [oracle["oracle_docs"] for oracle in self.oracle_list], self.dataset_type, self.model_type)
        else:
            raise Exception('You cannot perturb both retrieval acc and ret doc type')

        # generate prompt
        prompts = apply_prompt_method(questions=[qs['question'] for qs in self.qs_list], ret_docs=ret_docs, prompt_type=self.prompt_type, dataset=self.dataset)
        print(prompts[0])
        approximate_token(prompts)

        # gene response
        gene_results = list()
        for idx, prompt in tqdm(prompts):
            if self.model_type == 'gpt':
                outputs, logprobs = chatgpt(prompt=prompt, model=self.model, temperature=self.temperature, max_tokens=self.max_tokens, n=self.n)
            elif self.model_type == 'llama':
                ...
            gene_results.append(dict(qs_id=self.qs_list[idx]['qs_id'], output=outputs[0], logprob=logprobs[0], ret_libs=ret_doc_key_list[idx]))
        save_results_to_files(save_file=self.save_file, gene_results=gene_results)

    def eval(self):
        pred_list = json.load(open(self.save_file, 'r'))
        eval_pred(pred_list, self.oracle_list)



if __name__ == '__main__':
    # in_program_call = '--dataset hotpotQA --analysis_type retrieval_quality --retriever bm25 --retrieval_acc 1'
    # args = generate_config(in_program_call)
    args = generate_config()
    generator = GeneHotpotQA(args)
    generator.gene_response()
    generator.eval()


