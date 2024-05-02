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
from generator.generate_utils import (truncate_doc, approximate_token, get_dummy_text, generate_config, save_results_to_files, control_ret_acc,
                                      perturb_ret_doc_type, process_retrieval_doc, apply_prompt_method)


class GeneHotpotQA:
    def __init__(self, args):
        # load parameters
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

        print('qs_num:', len(self.qs_list))
        print('save_to:', self.save_file)

    def gene_response(self):
        gene_results = list()
        prompts = list()
        if self.ret_acc != 1 and self.ret_info_type == 'oracle':
            ret_doc_key_list, ret_docs = control_ret_acc(self.ret_acc, [oracle["oracle_docs"] for oracle in self.oracle_list], self.wiki_loader.load_wiki_id())
        elif self.ret_acc == 1 and self.ret_info_type != 'oracle':
            ret_results = ...
            ret_doc_key_list, ret_docs = perturb_ret_doc_type(self.ret_info_type, ..., [oracle["oracle_docs"] for oracle in self.oracle_list], self.dataset_type, self.model_type)
        else:
            raise Exception('You cannot perturb both retrieval acc and ret doc type')
        for idx, (qs, ret_doc) in tqdm(enumerate(zip(self.qs_list, ret_docs))):
            prompt = self.gene_prompt(qs['question'], ret_docs)
            prompts.append(prompt)
            if idx == 0: print(prompt)

            # gene response
            if self.model_type == 'gpt':
                outputs, logprobs = chatgpt(prompt=prompt, model=self.model, temperature=self.temperature, max_tokens=self.max_tokens, n=self.n)
            elif self.model_type == 'llama':
                ...
            gene_results.append(dict(question=qs, output=outputs[0], logprob=logprobs[0], ret_libs=ret_doc_key_list[idx]))

        approximate_token(prompts)
        save_results_to_files(save_file=self.save_file, gene_results=gene_results)


if __name__ == '__main__':
    in_program_call = '--dataset hotpotqa --analysis_type retrieval_quality --retriever bm25 --retrieval_acc 1'
    args = generate_config(in_program_call)
    args.retrieval_file = ...

    gene_conala = GeneHotpotQA(args)
    gene_conala.gene_response()