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
from dataset_utils.dataset_configs import HotpotQALoader
from retriever.sparse_retriever import sparse_retriever_config
from retriever.dense_retriever import dense_retriever_config
from generator.generate_utils import truncate_doc, approximate_token, get_dummy_text, generate_config, save_results_to_files


class GeneHotpotQA:
    def __init__(self, args):
        # load parameters
        self.save_file = args.save_file
        self.top_k = args.top_k
        self.ret_doc_type = args.ret_doc_type
        self.prompt_type = args.prompt_type
        self.max_doc_tokens = args.max_doc_tokens
        self.model = args.model
        self.temperature = args.temperature
        self.max_tokens = args.max_tokens
        self.hotpotqa_loader = HotpotQALoader()
        self.qs_list = self.hotpotqa_loader.load_qs_list()
        self.oracle_list = self.hotpotqa_loader.load_oracle_list()

        print('qs_num:', len(self.qs_list))
        print('save_to:', self.save_file)


    def get_ret_docs(self, qs_id, oracle, doc_type):
        # prepare retrieved docs
        if doc_type == 'oracle':
            ret_libs = oracle['doc_keys']
        elif doc_type == 'retrieved':
            ret_libs = [result['doc_key'] for result in self.ret_result[qs_id][0:self.top_k]]
        elif self.ret_doc_type == 'related':
            ret_libs = [result['doc_key'] for result in self.ret_result[qs_id] if
                        result['doc_key'] not in oracle['doc_keys']][:self.top_k]
        elif self.ret_doc_type == 'random':
            doc_key_list = list(self.doc_list.keys())
            # doc_key_list = [item for item in doc_key_list if item not in oracle_libs]
            ret_libs = random.sample(doc_key_list, self.top_k)
        elif self.ret_doc_type == 'unrelated':
            ret_libs = []
        elif self.ret_doc_type == 'none':
            ret_libs = []
        else:
            raise Exception('no such ret doc type')
        ret_docs = list()
        for line_idx, ret_lib in enumerate(ret_libs):
            ret_docs.append(f"potential document {line_idx}: {ret_lib}: {self.doc_list[ret_lib]}")
            ret_docs[line_idx] = ret_docs[line_idx].replace('\n', ' ')
        if self.ret_doc_type == 'unrelated':
            ret_docs = get_dummy_text(prompt_length=4000, dataset='conala')

        return ret_libs, ret_docs



    def gene_prompt(self, nl, ret_docs):
        if self.ret_doc_type == 'none':
            if self.prompt_type == 'original':
                prompt = conala_prompt.conala_original_no_retrieval_prompt
            elif self.prompt_type == 'instruct':
                prompt = conala_prompt.tldr_no_retrieval_prompt_with_instruction
            else:
                raise Exception('no such prompt type for non-retrieval')
        else:
            if self.prompt_type == '0shot':
                prompt = conala_prompt.conala_0shot_prompt
            elif self.prompt_type == 'original':
                prompt = conala_prompt.conala_original_3shots_prompt
            elif self.prompt_type == 'instruct':
                prompt = conala_prompt.conala_3shots_prompt_with_instruction
            else:
                raise Exception('no such prompt type')
        prompt += '\n'
        for doc in ret_docs:
            doc = truncate_too_long_doc(doc, max_length=self.max_doc_tokens)
            prompt += doc
            prompt += '\n'
        prompt += f'# {nl}'
        return prompt

    def gene_response(self):
        gene_results = list()
        prompts = list()
        for idx, (qs, oracle) in tqdm(enumerate(zip(self.qs_list, self.oracle_list))):

            ret_libs, ret_docs = self.get_ret_docs(qs['qs_id'], oracle)
            prompt = self.gene_prompt(qs['question'], ret_docs)

            # gene response
            prompts.append(prompt)
            output = chatgpt(prompt=prompt, model=self.model, temperature=self.temperature, max_tokens=self.max_tokens)[0].replace('\n', ' ').replace('#END', '')
            output = (output.replace("{{", " {{").replace("\n", ' ').replace("\r", "").replace("<pad>", "").replace("<s>", "").replace("</s>", "").strip())
            output = " ".join(output.split())
            gene_results.append(dict(nl=qs, output=output, ret_libs=ret_libs, oracle_libs=oracle['doc_keys'], oracle_output=oracle['output']))
            if idx == 0: print(prompt)

        approximate_token(prompts)
        save_results_to_files(save_file=self.save_file, gene_results=gene_results)


if __name__ == '__main__':
    in_program_call = '--dataset hotpotqa --sampled --analysis_type retrieval_quality --retriever bm25 --retrieval_acc 1'
    args = generate_config(in_program_call)
    args.retrieval_file = ...

    gene_conala = GeneHotpotQA(args)
    gene_conala.gene_response()