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
from prompt import tldr_prompt
from dataset_utils.dataset_configs import TldrLoader
from retriever.sparse_retriever import sparse_retriever_config
from retriever.dense_retriever import dense_retriever_config
from generator.generate_utils import truncate_too_long_doc, approximate_token, get_dummy_text, generate_config, save_results_to_files


class GeneTldr:
    def __init__(self, args, retriever_args):
        # load parameters
        self.save_file = args.save_file
        self.top_k = args.top_k
        self.k_line = args.k_line
        self.ret_doc_type = args.ret_doc_type
        self.prompt_type = args.prompt_type
        self.max_doc_tokens = args.max_doc_tokens
        self.model = args.model
        self.temperature = args.temperature
        self.max_tokens = args.max_tokens
        # load docs
        self.tldr_loader    = TldrLoader()
        self.doc_list_whole = self.tldr_loader.load_doc_list_whole()
        self.doc_list_line  = self.tldr_loader.load_doc_list_line()
        self.qs_list        = self.tldr_loader.load_qs_list(retriever_args.dataset_type)
        self.oracle_list    = self.tldr_loader.load_oracle_list(retriever_args.dataset_type)
        if args.retriever == 'bm25':
            self.ret_result_whole = json.load(open(retriever_args.tldr_ret_result_whole, 'r'))
            self.ret_result_line = json.load(open(retriever_args.tldr_ret_result_line, 'r'))
        else:
            raise Exception('unfinished retriever type')

        print('qs_num:', len(self.qs_list))
        print('save_to:', args.save_file)

    def get_ret_docs(self, qs_id, oracle):
        # todo: now only attach retrieved docs in line level
        # res whole: {'qs_id': [{'key1', 'score1'}, ...]} res line {'qs_id': [[{'key1_1', 'score1_1'}, ...], ...]
        if self.ret_doc_type == 'oracle':
            ret_cmd_line = oracle['line_keys'][:self.k_line]
        elif self.ret_doc_type == 'retrieved':
            ret_cmd_line = list()
            for k in range(0, self.top_k):
                ret_cmd_line.extend([ret_result['doc_key'] for ret_result in self.ret_result_line[qs_id][k][0:self.k_line]])
        elif self.ret_doc_type == 'related':
            ret_cmd_idx = [idx for idx, ret_result in enumerate(self.ret_result_whole[qs_id]) if
                           ret_result['doc_key'] not in oracle['doc_keys']][:self.top_k]  # get top k cmd expect oracle
            ret_cmd_line = list()
            for cmd_idx in ret_cmd_idx:
                assert self.ret_result_line[qs_id][cmd_idx][0]['doc_key'].rsplit('_', 1)[0] == self.ret_result_whole[qs_id][cmd_idx]['doc_key']
                ret_cmd_line.extend([ret_result['doc_key'] for ret_result in self.ret_result_line[qs_id][cmd_idx][0:self.k_line]])
        elif self.ret_doc_type == 'random':
            doc_key_whole_list = list(self.doc_list_whole.keys())
            # doc_key_whole_list.remove(oracle_cmd)
            random_cmds = random.sample(doc_key_whole_list, self.top_k)  # get top k random cmds
            ret_cmd_line = list()
            for random_cmd in random_cmds:
                cmd_key_list = [key for key in self.doc_list_line.keys() if key.startswith(random_cmd)]
                ret_cmd_line.extend(random.sample(cmd_key_list, self.k_line if len(cmd_key_list) > self.k_line else len(cmd_key_list)))  # get random k lines for each cmd
        elif self.ret_doc_type == 'unrelated':
            ret_cmd_line = []
        elif self.ret_doc_type == 'none':
            ret_cmd_line = []
        else:
            raise Exception('no such ret doc type')
        ret_docs = list()
        for line_idx, cmd_line in enumerate(ret_cmd_line):
            ret_docs.append(f"potential document {line_idx}: {cmd_line}: {self.doc_list_line[cmd_line]}")
        if self.ret_doc_type == 'unrelated':
            ret_docs = get_dummy_text(prompt_length=1000, dataset='tldr')

        return ret_cmd_line, ret_docs

    def prepare_prompt(self, nl, ret_docs):
        if self.ret_doc_type == 'none':
            if self.prompt_type == 'original':
                prompt = tldr_prompt.tldr_original_no_retrieval_prompt
            elif self.prompt_type == 'instruct':
                prompt = tldr_prompt.tldr_no_retrieval_prompt_with_instruction
            else:
                raise Exception('no such prompt type for non-retrieval')
        else:
            if self.prompt_type == '0shot':
                prompt = tldr_prompt.tldr_0shot_prompt
            elif self.prompt_type == 'original':
                prompt = tldr_prompt.tldr_original_3shots_prompt
            elif self.prompt_type == 'instruct':
                prompt = tldr_prompt.tldr_3shots_prompt_with_instruction
            else:
                raise Exception('no such prompt type')
        prompt += '\n\n'
        for doc in ret_docs:
            doc = truncate_too_long_doc(doc=doc, max_length=self.max_doc_tokens)
            prompt += doc
            prompt += '\n'
        prompt += f'# {nl}'
        return prompt


    def gene_response(self):
        gene_results = list()
        prompts = list()
        for idx, (qs, oracle) in tqdm(enumerate(zip(self.qs_list, self.oracle_list))):
            if oracle['doc_keys'][0] == 'xkcdpass': continue  # cmd not exists in docs
            qs_id = qs['qs_id']
            assert qs_id == oracle['qs_id']

            ret_cmd_line, ret_docs = self.get_ret_docs(qs_id=qs_id, oracle=oracle)
            prompt = self.prepare_prompt(nl=qs['nl'], ret_docs=ret_docs)

            prompts.append(prompt)
            output = chatgpt(prompt=prompt, model=self.model, temperature=self.temperature, max_tokens=self.max_tokens)[0].replace('\n',' ').replace('#END','')
            gene_results.append(dict(nl=qs, output=output, ret_cmd=ret_cmd_line, oracle_cmd=oracle['doc_keys'][0], oracle_output=oracle['output'], oracle_ret=oracle['line_keys']))
            if idx == 0: print(prompt)
            # print(output)

        # count tokens
        approximate_token(prompts)
        save_results_to_files(save_file=self.save_file, gene_results=gene_results)


if __name__ == '__main__':
    in_program_call = '--dataset tldr --top_k 1 --k_line 10 --retriever bm25 --ret_doc_type retrieved --prompt_type original'
    args = generate_config(in_program_call)
    retriever_args = sparse_retriever_config('--dataset tldr --dataset_type dev')

    gene_tldr = GeneTldr(args, retriever_args)
    gene_tldr.gene_response()