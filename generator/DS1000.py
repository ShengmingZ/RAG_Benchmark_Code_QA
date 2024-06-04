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
from prompt import DS1000_prompt
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

    def get_ret_docs(self, oracle):
        # todo: complete retrieval
        if self.ret_doc_type == 'none':
            ret_libs = []
        elif self.ret_doc_type == 'oracle':
            ret_libs = oracle['doc_keys']
        else:
            raise Exception('no such ret doc type')
        ret_docs = list()
        for line_idx, ret_lib in enumerate(ret_libs):
            ret_docs.append(f"potential document {line_idx}: {ret_lib}: {self.doc_list[ret_lib]}")
            ret_docs[line_idx] = ret_docs[line_idx].replace('\n', ' ')
        return ret_libs, ret_docs

    def prepare_prompt(self, nl, ret_docs):
        if self.ret_doc_type == 'none':
            return nl

        if '\nA:\n' in nl:
            prompt_with_problem = True
        else:
            prompt_with_problem = False

        if prompt_with_problem:
            [problem, code_snippet] = nl.split('\nA:\n')
            problem = '### ' + problem + '\n\n'
            code_snippet = '### Uncompleted Code Snippet:\n' + code_snippet
        else:
            code_snippet = '### Uncompleted Code Snippet:\n' + nl

        if prompt_with_problem:
            prompt = ds1000_prompt.retrieval_3shots_prompt_1 + '\n\n' + problem + '\n' + '### Potential Document:\n'
        else:
            prompt = ds1000_prompt.retrieval_3shots_prompt_2 + '\n\n' + '### Potential Document:\n'
        for doc in ret_docs:
            doc = truncate_too_long_doc(doc, max_length=self.max_doc_tokens)
            prompt += doc
            prompt += '\n'
        prompt += f'\n\n{code_snippet}'
        prompt += f'\n\n### Answer:\n'

        return prompt

    def gene_response(self):
        gene_results = []
        prompts = []
        for idx, (qs, oracle) in tqdm(enumerate(zip(self.qs_list, self.oracle_list))):
            if qs['qs_id'].split('_')[0].lower() == 'scipy': continue     # now skip scipy
            assert qs['qs_id'] == oracle['qs_id']
            ret_libs, ret_docs = self.get_ret_docs(oracle=oracle)
            prompt = self.prepare_prompt(nl=qs['nl'], ret_docs=ret_docs)

            prompts.append(prompt)
            outputs = chatgpt(prompt=prompt, model=self.model, temperature=self.temperature, max_tokens=self.max_tokens, stop=["</code>", "###", "# SOLUTION END", "END"], n=self.n)
            gene_results.append(dict(nl=qs, outputs=outputs, ret_libs=ret_libs, oracle_libs=oracle['doc_keys'], oracle_output=oracle['output']))
            # gene_results.append(dict(nl=qs, outputs=outputs, oracle_output=oracle['output']))
            if idx == 0:
                print(prompt)
                print(outputs[0])

        approximate_token(prompts)
        save_results_to_files(save_file=self.save_file, gene_results=gene_results)


if __name__ == '__main__':
    in_program_call = '--dataset ds1000 --n 100 --sampled --top_k 1 --retriever bm25 --ret_doc_type oracle --prompt_type original'
    args = generate_config(in_program_call)
    retriever_args = None

    gene_ds1000 = GeneDS1000(args, retriever_args)
    gene_ds1000.gene_response()

    # nl = "Problem:\nI have a set of data and I want to compare which line describes it best (polynomials of different orders, exponential or logarithmic).\nI use Python and Numpy and for polynomial fitting there is a function polyfit(). \nHow do I fit y = A + Blogx using polyfit()? The result should be an np.array of [A, B]\nA:\n<code>\nimport numpy as np\nimport scipy\nx = np.array([1, 7, 20, 50, 79])\ny = np.array([10, 19, 30, 35, 51])\n\n</code>\nresult = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\n"
    # ret_libs = ["matplotlib._as_gen.matplotlib.axes.axes.add_patch"]
    # print(gene_ds1000.prepare_prompt(nl, ret_libs))

    # ds1000_loader = DS1000Loader()
    # qs_list = ds1000_loader.load_qs_list()
    # nl_with_problem_num, nl_wo_problem_num = 0, 0
    # exception_num = 0
    # for idx, qs in enumerate(qs_list):
    #     nl = qs['nl']
    #     try:
    #         [problem, answer] = nl.split('\nA:\n')
    #     except:
    #         exception_num += 1
    #         print('nl:\n', nl)
    #     if 'Problem:' in nl:
    #         nl_with_problem_num += 1
    #     else:
    #         nl_wo_problem_num += 1
    #
    # print(exception_num)
    # print(nl_with_problem_num)
    # print(nl_wo_problem_num)