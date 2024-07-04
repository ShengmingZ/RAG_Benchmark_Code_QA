import json
import os
import random
import time

from tqdm import tqdm
import platform
import sys
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from generator.run_model import chatgpt, llama, chatgpt_batch
from prompt import conala_prompt
from retriever.retriever_utils import retriever_config, get_ret_results
from dataset_utils.conala_utils import ConalaLoader
from dataset_utils.NQ_TriviaQA_utils import NQTriviaQAUtils
from dataset_utils.corpus_utils import PythonDocsLoader, WikiCorpusLoader
from generator.generate_utils import control_ret_acc, save_results_to_files, generate_prompts, generate_config, truncate_docs, approximate_token, perturb_ret_doc_type, select_retrieval_docs, get_docs_for_pl_analysis
from dataset_utils.DS1000_utils import DS1000Loader
from dataset_utils.pandas_numpy_eval_utils import PandasNumpyEvalLoader
from dataset_utils.hotpotQA_utils import HotpotQAUtils
from prompt import NQ_TriviaQA_prompt


class Generator:
    def __init__(self, args):
        # load parameters
        self.dataset = args.dataset
        self.save_file = args.save_file
        self.model = args.model
        self.n = args.n
        self.max_tokens = args.max_tokens
        self.temperature = args.temperature
        self.batch = args.batch

        self.retriever = args.retriever

        self.analysis_type = args.analysis_type
        self.ret_acc = args.ret_acc
        self.ret_doc_type = args.ret_doc_type
        self.doc_selection_type = args.doc_selection_type
        self.prompt_type = args.prompt_type
        self.pl_analysis = args.pl_analysis
        self.doc_max_length = args.doc_max_length
        # load docs
        if self.dataset == 'conala':
            self.dataset_loader = ConalaLoader()
        elif self.dataset == 'DS1000':
            self.dataset_loader = DS1000Loader()
        elif self.dataset == 'pandas_numpy_eval':
            self.dataset_loader = PandasNumpyEvalLoader()
        elif self.dataset == 'hotpotQA':
            self.dataset_loader = HotpotQAUtils()
        else:
            self.dataset_loader = NQTriviaQAUtils(dataset=self.dataset)
        self.qs_list = self.dataset_loader.load_qs_list()
        self.oracle_list = self.dataset_loader.load_oracle_list()
        if self.dataset in ['conala', 'DS1000', 'pandas_numpy_eval']:
            self.corpus_loader = PythonDocsLoader()
            # self.stop = '</code>'
        else:
            self.corpus_loader = WikiCorpusLoader()
            # self.stop = '</answer>'
        self.stop = None
        self.ret_results = get_ret_results(dataset=args.dataset, retriever=args.retriever)

        # test
        # self.qs_list = self.qs_list[:1]
        # self.oracle_list = self.oracle_list[:1]

        print('qs_num:', len(self.qs_list))
        print('save_to:', self.save_file)

    def gene_prompts(self):
        if self.analysis_type == 'retrieval_recall':
            ret_doc_keys_list, docs_list = control_ret_acc(ret_acc=args.ret_acc,
                                                           oracle_list=self.oracle_list,
                                                           ret_results=self.ret_results,
                                                           dataset=self.dataset)
        elif self.analysis_type == 'retrieval_doc_type':
            ret_doc_keys_list, docs_list = perturb_ret_doc_type(perturb_doc_type=self.ret_doc_type,
                                                                oracle_list=self.oracle_list,
                                                                ret_results=self.ret_results,
                                                                model=self.model,
                                                                dataset=self.dataset)
        elif self.analysis_type == 'retrieval_doc_selection':
            ret_doc_keys_list, docs_list = select_retrieval_docs(ret_results=self.ret_results,
                                                                 oracle_list=self.oracle_list,
                                                                 doc_selection_type=self.doc_selection_type,
                                                                 dataset=self.dataset)
        elif self.analysis_type == 'prompt_length':
            ret_doc_keys_list, docs_list = get_docs_for_pl_analysis(pl_analysis=self.pl_analysis,
                                                                    oracle_list=self.oracle_list,
                                                                    ret_results=self.ret_results,
                                                                    model=self.model,
                                                                    dataset=self.dataset)

        else:
            raise NotImplementedError(f'unknown analysis type: {self.analysis_type}')


        prompts = generate_prompts(questions=[qs['question'] for qs in self.qs_list],
                                   ret_docs_list=docs_list,
                                   prompt_type=self.prompt_type,
                                   dataset=self.dataset,
                                   model_name=self.model,
                                   doc_max_length=self.doc_max_length)
        return ret_doc_keys_list, prompts

    def test_prompt(self):
        random.seed()
        self.oracle_list = random.sample(self.oracle_list, 1)
        for qs in self.qs_list:
            if qs['qs_id'] == self.oracle_list[0]['qs_id']:
                self.qs_list = [qs]
                break
        assert len(self.qs_list) == len(self.oracle_list)
        _, prompts = self.gene_prompts()
        if self.model.startswith('gpt'):
            print(prompts[0][0])
            print(prompts[0][1])
        else:
            print(prompts[0])

    def calc_prompt_tokens(self):
        self.gene_prompts()

    def gene_response(self):
        # if self.batch: self.save_file = self.save_file.replace('.json', '_test_batch.json')
        if os.path.exists(self.save_file):
            print(f'generation results exists for {self.save_file}')
            return

        ret_doc_keys_list, prompts = self.gene_prompts()

        if self.model.startswith('llama') or self.model.startswith('codellama'):
            outputs_list, logprobs_list = llama(prompts=prompts, model_name=self.model, max_new_tokens=self.max_tokens, temperature=self.temperature, n=self.n, stop=self.stop)
        elif self.model.startswith('gpt'):
            if self.batch is False:
                outputs_list, logprobs_list = chatgpt(prompts=prompts, model=self.model, max_tokens=self.max_tokens, temperature=self.temperature, n=self.n, stop=self.stop)
            else:
                prompt_save_file = self.save_file.replace('.json', '_prompts.json')
                outputs_list, logprobs_list = chatgpt_batch(prompt_save_file=prompt_save_file, prompts=prompts, model=self.model, max_tokens=self.max_tokens, temperature=self.temperature, n=self.n, stop=self.stop)
        else:
            raise NotImplementedError(f'unknown model {self.model}')
        gene_results = list()
        for idx, (outputs, logprobs) in enumerate(zip(outputs_list, logprobs_list)):
            if 'irrelevant' in self.ret_doc_type or self.ret_doc_type == 'none' or 'irrelevant' in self.pl_analysis: ret_docs = None
            else: ret_docs = ret_doc_keys_list[idx]
            gene_results.append(dict(qs_id=self.qs_list[idx]['qs_id'],
                                     question=self.qs_list[idx]['question'],
                                     ret_docs=ret_docs,
                                     outputs=outputs,
                                     logprobs=logprobs
                                     ))
        save_results_to_files(self.save_file, gene_results, overwrite=True)
        return gene_results


        # for idx, (qs, oracle) in tqdm(enumerate(zip(self.qs_list, self.oracle_list))):
        #
        #     ret_libs, ret_docs = self.get_ret_docs(qs['qs_id'], oracle)
        #     prompt = self.prepare_prompt(qs['nl'], ret_docs)
        #
        #     # gene response
        #     prompts.append(prompt)
        #     output = chatgpt(prompt=prompt, model=self.model, temperature=self.temperature, max_tokens=self.max_tokens)[0].replace('\n', ' ').replace('#END', '')
        #     output = (output.replace("{{", " {{").replace("\n", ' ').replace("\r", "").replace("<pad>", "").replace("<s>", "").replace("</s>", "").strip())
        #     output = " ".join(output.split())
        #     gene_results.append(dict(nl=qs, output=output, ret_libs=ret_libs, oracle_libs=oracle['doc_keys'], oracle_output=oracle['output']))
        #     if idx == 0: print(prompt)
        #
        # approximate_token(prompts)
        # save_results_to_files(save_file=self.save_file, gene_results=gene_results)


if __name__ == '__main__':
    # in_program_call = '--dataset conala --top_k 10 --retriever codet5-FT --ret_doc_type retrieved --prompt_type original'
    # args = generate_config(in_program_call)
    # if args.retriever == 'codeT5-FT':
    #     retriever_args = dense_retriever_config(f"--dataset conala --dataset_type test \
    #                         --model_name neulab/docprompting-codet5-python-doc-retriever")
    # elif args.retriever == 'codeT5-OTS':
    #     retriever_args = dense_retriever_config(f"--dataset conala --dataset_type test \
    #                         --model_name Salesforce/codet5-base")
    #
    # gene_conala = GeneConala(args, retriever_args)
    # gene_conala.gene_response()

    in_program_call = None
    # in_program_call = '--model gpt-3.5-turbo-0125 --temperature 0 --n 1 --batch --dataset NQ --retriever openai-embedding --analysis_type prompt_length --pl_analysis irrelevant_dummy_top10'
    # in_program_call = '--model gpt-3.5-turbo-0125 --dataset conala --retriever openai-embedding --analysis_type retrieval_doc_type --ret_doc_type none'
    # in_program_call = '--model gpt-3.5-turbo-0125 --dataset conala --retriever openai-embedding --analysis_type retrieval_doc_selection --doc_selection_type top_5'
    args = generate_config(in_program_call)
    generator = Generator(args)
    # generator.test_prompt()
    # generator.calc_prompt_tokens()

    gene_results = generator.gene_response()

