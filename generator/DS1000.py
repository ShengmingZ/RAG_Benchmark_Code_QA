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
from dataset_utils.dataset_configs import DS1000Loader
from retriever.sparse_retriever import sparse_retriever_config
from retriever.dense_retriever import dense_retriever_config
from generator.generate_utils import truncate_too_long_doc, approximate_token, get_dummy_text, generate_config, save_results_to_files


class GeneDS1000:
    def __init__(self, args, retriever_args):
        # load parameters
        self.save_file = args.save_file
        self.top_k = args.top_k
        self.ret_doc_type = args.ret_doc_type
        self.prompt_type = args.prompt_type
        self.max_doc_tokens = args.max_doc_tokens
        self.model = args.model
        self.temperature = args.temperature
        self.max_tokens = args.max_tokens
        self.n = args.n
        # load docs
        self.ds1000_loader = DS1000Loader()
        self.doc_list = self.ds1000_loader.load_doc_list()
        self.qs_list = self.ds1000_loader.load_qs_list(sampled=args.sampled)
        self.oracle_list = self.ds1000_loader.load_oracle_list(sampled=args.sampled)
        # if args.retriever == 'bm25':
        #     self.ret_result = json.load(open(retriever_args.conala_ret_result, 'r'))
        # elif 'codeT5' in args.retriever:
        #     self.ret_result = json.load(open(retriever_args.save_file, 'r'))
        # else:
        #     raise Exception('retriever type not supported')

        print('qs_num:', len(self.qs_list))
        print('save_to:', self.save_file)

    def get_ret_docs(self):
        # todo: complete retrieval
        if self.ret_doc_type == 'none':
            ret_libs = []
        ret_docs = []
        return ret_libs, ret_docs

    def prepare_prompt(self, nl):
        # todo: complete prompt generation
        if self.ret_doc_type == 'none':
            prompt = nl
        return prompt

    def gene_response(self):
        gene_results = []
        prompts = []
        for idx, (qs, oracle) in tqdm(enumerate(zip(self.qs_list, self.oracle_list))):
            assert qs['qs_id'] == oracle['qs_id']
            ret_libs, ret_docs = self.get_ret_docs()
            prompt = self.prepare_prompt(qs['nl'])

            prompts.append(prompt)
            outputs = chatgpt(prompt=prompt, model=self.model, temperature=self.temperature, max_tokens=self.max_tokens, stop=["</code>", "# SOLUTION END"], n=self.n)
            # gene_results.append(dict(nl=qs, output=output, ret_libs=ret_libs, oracle_libs=oracle['doc_keys'], oracle_output=oracle['output']))
            gene_results.append(dict(nl=qs, outputs=outputs, oracle_output=oracle['output']))

        approximate_token(prompts)
        save_results_to_files(save_file=self.save_file, gene_results=gene_results)


if __name__ == '__main__':
    in_program_call = '--dataset ds1000 --sampled --n 100 --top_k 1 --retriever bm25 --ret_doc_type none --prompt_type original'
    args = generate_config(in_program_call)
    retriever_args = None

    gene_ds1000 = GeneDS1000(args, retriever_args)
    gene_ds1000.gene_response()
