import json
import os
from datetime import datetime
from typing import List, Dict, Tuple
import argparse
import sys
sys.path.append('..')
from prompt import conala_prompt, DS1000_prompt, pandas_numpy_eval_prompt, hotpotQA_prompt, NQ_TriviaQA_prompt
from retriever.RetrievalProvider import RetrievalProvider
from dataset_utils.conala_utils import ConalaLoader
from dataset_utils.DS1000_utils import DS1000Loader
from dataset_utils.NQ_TriviaQA_utils import NQTriviaQAUtils
from dataset_utils.hotpotQA_utils import HotpotQAUtils
from dataset_utils.pandas_numpy_eval_utils import PandasNumpyEvalLoader
from llms.LLMConfig import LLMConfig, LLMSettings
from llms.OpenAIProvider import OpenAIProvider
from llms.LLAMAProvider import LlamaProvider
from generator_deprecated.pred_eval import pred_eval_new
from generator_deprecated.generate_utils import truncate_docs


class LLMOracleEvaluator:
    def __init__(self, model, dataset):
        if model == 'openai-old': self.model_config = LLMSettings().OpenAIConfigs().openai_old
        elif model == 'openai-new': self.model_config = LLMSettings().OpenAIConfigs().openai_new
        elif model == 'llama-new': self.model_config = LLMSettings().LLAMAConfigs().llama_new
        elif model == 'llama-old-code': self.model_config = LLMSettings().LLAMAConfigs().llama_old_code
        elif model == 'llama-old-qa': self.model_config = LLMSettings().LLAMAConfigs().llama_old_qa
        else: raise Exception('Unknown model')
        if dataset in ['NQ', 'TriviaQA', 'HotpotQA']: self.max_tokens = 500     # todo: do not consider prompting method!
        elif dataset == 'DS1000' or dataset == 'pandas_numpy_eval': self.max_tokens = 1000
        else: self.max_tokens = 500
        if self.model_config.organization == 'openai':
            self.llm_provider = OpenAIProvider(organization=self.model_config.organization,
                                               model=self.model_config.model,
                                               temperature=self.model_config.temperature,
                                               max_tokens=self.max_tokens,
                                               is_async=self.model_config.is_async,
                                               stop=['</answer>'])
        elif self.model_config.organization == 'llama':
            self.llm_provider = LlamaProvider(organization=self.model_config.organization,
                                              model=self.model_config.model,
                                              temperature=self.model_config.temperature,
                                              max_tokens=self.max_tokens,
                                              is_async=self.model_config.is_async,
                                              stop=['package com', '</answer>'])

        self.dataset = dataset

        self.doc_loader = RetrievalProvider(self.dataset)

        self.oracle_docs = self.doc_loader.get_oracle_docs()

        self.model_names_for_path = {"gpt-4o-mini": "gpt-4o-mini",
                                     "gpt-3.5-turbo-0125": "gpt-3-5-turbo",
                                     "codellama/CodeLlama-13b-Instruct-hf": "codellama-13b",
                                     "meta-llama/Llama-2-13b-chat-hf": 'llama2-13b'}

        if self.dataset == 'conala':
            self.problems = ConalaLoader().load_qs_list()
            self.prompt_generator = conala_prompt.prompt_0shot
            self.prompt_generator_no_ret = conala_prompt.prompt_0shot_no_ret
            self.prompt_utils = conala_prompt
        elif self.dataset == 'DS1000':
            self.problems = DS1000Loader().load_qs_list()
            self.prompt_generator = DS1000_prompt.prompt_0shot
            self.prompt_generator_no_ret = DS1000_prompt.prompt_0shot_no_ret
            self.prompt_utils = DS1000_prompt
        elif self.dataset == 'pandas_numpy_eval':
            self.problems = PandasNumpyEvalLoader().load_qs_list()
            self.prompt_generator = pandas_numpy_eval_prompt.prompt_0shot
            self.prompt_generator_no_ret = pandas_numpy_eval_prompt.prompt_0shot_no_ret
            self.prompt_utils = pandas_numpy_eval_prompt
        elif self.dataset == 'NQ' or self.dataset == 'TriviaQA':
            self.problems = NQTriviaQAUtils(dataset=self.dataset).load_qs_list()
            self.prompt_generator = NQ_TriviaQA_prompt.prompt_0shot
            self.prompt_generator_no_ret = NQ_TriviaQA_prompt.prompt_0shot_no_ret
            self.prompt_utils = NQ_TriviaQA_prompt
        elif self.dataset == 'hotpotQA':
            self.problems = HotpotQAUtils().load_qs_list()
            self.prompt_generator = hotpotQA_prompt.prompt_0shot
            self.prompt_generator_no_ret = hotpotQA_prompt.prompt_0shot_no_ret
            self.prompt_utils = hotpotQA_prompt
        else:
            raise Exception('unsupported dataset')

    def generate_single_llm(self, result_path=None, test_prompt=False):
        # default result path for SINGLE llm
        if result_path is None:
            model_name_for_path = self.model_names_for_path[self.model_config.model]
            result_path = f'../data/{self.dataset}/new_results/single_{model_name_for_path}.json'

        # Prepare messages
        prompts = []
        problem_ids = []

        for problem in self.problems:
            # if '\nA:' not in problem['question']: print(problem['question'])
            prompt = self.prompt_generator_no_ret(question=problem['question'], model=self.model_config.model)
            prompts.append(prompt)
            problem_ids.append(problem['qs_id'])

        if test_prompt:
            if 'gpt' in self.model_config.model:
                print(prompts[0][0]['content'])
                print(prompts[0][1]['content'])
            elif 'llama' in self.model_config.model.lower():
                print(prompts[0])
            return

        if os.path.exists(result_path):
            print('result already exists in path {}, if want to overwrite, please delete it first'.format(result_path))
            # pred_eval_new(self.dataset, result_path=result_path)
            return

        """Generate responses using single LLM only"""
        print(f"🤖 Generating Single LLM responses for {len(self.problems)} questions...")

        # Batch API call
        if 'gpt' in self.model_config.model:
            llm_responses = self.llm_provider.batch_generate(
                prompts=prompts,
                return_type="text",
                include_logits=True,
                custom_id_prefix=f"single_{self.dataset}_{self.model_config.model}"
            )
        elif 'llama' in self.model_config.model.lower():
            llm_responses = self.llm_provider.generate_batch(
                prompts=prompts,
                return_type="text",
                include_logits=True
            )
        else:
            raise Exception('unsupported model')

        # Process results
        results = []
        for problem_id, response in zip(problem_ids , llm_responses):
            results.append({
                'qs_id': problem_id,
                'method': 'single_llm',
                'response': response.get('text', ''),
                'logprobs': response.get('logprobs', []),
            })

        print(f"✅ Generated {len(results)} Single LLM responses")

        os.makedirs(result_path.rsplit('/', 1)[0], exist_ok=True)
        with open(result_path, 'w+') as f:
            json.dump(results, f, indent=2)


    def generate_with_oracle(self, result_path=None, test_prompt=False):
        # default result path for SINGLE llm
        if result_path is None:
            model_name_for_path = self.model_names_for_path[self.model_config.model]
            result_path = f'../data/{self.dataset}/new_results/oracle_{model_name_for_path}.json'

        """Generate responses using single LLM only"""
        print(f"🤖 Generating Oracle LLM responses for {len(self.problems)} questions...")

        # Prepare messages
        prompts = []
        problem_ids = []

        oracle_docs = self.doc_loader.get_oracle_docs()

        for problem, qs_id in zip(self.problems, oracle_docs):
            assert qs_id == problem['qs_id']
            truncated_docs = truncate_docs(oracle_docs[qs_id], model='gpt-3.5-turbo-0125', max_length=500)
            prompt = self.prompt_generator(question=problem['question'], model=self.model_config.model, ret_docs=truncated_docs)
            prompts.append(prompt)
            problem_ids.append(problem['qs_id'])

        if test_prompt:
            if 'gpt' in self.model_config.model:
                print(prompts[0][0]['content'])
                print(prompts[0][1]['content'])
            elif 'llama' in self.model_config.model.lower():
                print(prompts[0])
            return

        if os.path.exists(result_path):
            print('result already exists in path {}, if want to overwrite, please delete it first'.format(result_path))
            # pred_eval_new(self.dataset, result_path=result_path)
            return

        # Batch API call
        if 'gpt' in self.model_config.model:
            llm_responses = self.llm_provider.batch_generate(
                prompts=prompts,
                return_type="text",
                include_logits=True,
                custom_id_prefix=f"single_{self.dataset}_{self.model_config.model}"
            )
        elif 'llama' in self.model_config.model.lower():
            llm_responses = self.llm_provider.generate_batch(
                prompts=prompts,
                return_type="text",
                include_logits=True
            )
        else:
            raise Exception('unsupported model')

        # Process results
        results = []
        for problem_id, response in zip(problem_ids , llm_responses):
            results.append({
                'qs_id': problem_id,
                'method': 'oracle_llm',
                'response': response.get('text', ''),
                'logprobs': response.get('logprobs', []),
            })

        print(f"✅ Generated {len(results)} Oracle LLM responses")

        os.makedirs(result_path.rsplit('/', 1)[0], exist_ok=True)
        with open(result_path, 'w+') as f:
            json.dump(results, f, indent=2)




    def generate_with_recall(self, recall, result_path=None, test_prompt=False):
        self.recall_range = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        assert recall in self.recall_range

        if recall == 0: recall = int(recall)    # turn 0.0 to 0
        # default result path for SINGLE llm
        if result_path is None and recall != 1.0:
            model_name_for_path = self.model_names_for_path[self.model_config.model]
            result_path = f'../data/{self.dataset}/new_results/recall-{recall}_{model_name_for_path}.json'
        elif result_path is None and recall == 1.0:
            model_name_for_path = self.model_names_for_path[self.model_config.model]
            result_path = f'../data/{self.dataset}/new_results/oracle_{model_name_for_path}.json'
        # if result_path is None and recall != 1.0: result_path = f'../data/{self.dataset}/new_results/recall-{recall}_{self.model_config.model}.json'
        # elif result_path is None and recall == 1.0: result_path = f'../data/{self.dataset}/new_results/oracle_{self.model_config.model}.json'
        if os.path.exists(result_path):
            print('result already exists in path {}, if want to overwrite, please delete it first'.format(result_path))
            # pred_eval_new(self.dataset, result_path=result_path)
            return

        """Generate responses using single LLM only"""
        print(f"🤖 Generating Oracle LLM responses for {len(self.problems)} questions...")

        # Prepare messages
        prompts = []
        problem_ids = []

        controlled_docs = self.doc_loader.get_recall_controlled_docs(recall=recall)

        for problem, qs_id in zip(self.problems, controlled_docs):
            assert qs_id == problem['qs_id']
            truncated_docs = truncate_docs(controlled_docs[qs_id], model='gpt-3.5-turbo-0125', max_length=500)
            prompt = self.prompt_generator(question=problem['question'], model=self.model_config.model, ret_docs=truncated_docs)
            prompts.append(prompt)
            problem_ids.append(problem['qs_id'])

        if test_prompt:
            if 'gpt' in self.model_config.model:
                print(prompts[0][0]['content'])
                print(prompts[0][1]['content'])
            elif 'llama' in self.model_config.model.lower():
                print(prompts[0])
            return

        # Batch API call
        if 'gpt' in self.model_config.model:
            llm_responses = self.llm_provider.batch_generate(
                prompts=prompts,
                return_type="text",
                include_logits=True,
                custom_id_prefix=f"single_{self.dataset}_{self.model_config.model}"
            )
        elif 'llama' in self.model_config.model.lower():
            llm_responses = self.llm_provider.generate_batch(
                prompts=prompts,
                return_type="text",
                include_logits=True
            )
        else:
            raise Exception('unsupported model')

        # Process results
        results = []
        for problem_id, response in zip(problem_ids , llm_responses):
            results.append({
                'qs_id': problem_id,
                'method': 'recall_llm',
                'response': response.get('text', ''),
                'logprobs': response.get('logprobs', []),
            })

        print(f"✅ Generated {len(results)} Recall Analysis LLM responses")

        os.makedirs(result_path.rsplit('/', 1)[0], exist_ok=True)
        with open(result_path, 'w+') as f:
            json.dump(results, f, indent=2)




    def generate_with_k(self, k, result_path=None, test_prompt=False):
        if self.dataset in ['conala', 'DS1000', 'pandas_numpy_eval']:
            self.k_range = [1, 3, 5, 7, 10, 13, 16, 20]
        else:
            if self.model_config == LLMSettings().LLAMAConfigs().llama_old_qa:
                self.k_range = [1, 3, 5, 10, 15, 20]
            else:
                self.k_range = [1, 3, 5, 10, 15, 20, 30, 40]
        assert k in self.k_range

        # Prepare messages
        prompts = []
        problem_ids = []

        ret_docs = self.doc_loader.get_ret_docs()

        for problem in self.problems:
            ret_docs_exist = False
            # ret docs' qs_id may be different from oracle
            for qs_id in ret_docs:
                if qs_id == problem['qs_id']:
                    ret_docs_exist = True
                    break
            if not ret_docs_exist: raise Exception(f'no ret docs for problem: {qs_id}')
            # use top-k docs as retrieved docs
            if self.dataset in ['conala', 'DS1000', 'pandas_numpy_eval']:
                truncated_docs = truncate_docs(ret_docs[qs_id][:k], model='gpt-3.5-turbo-0125', max_length=500)
            else:
                truncated_docs = [item['doc'] for item in ret_docs[qs_id][:k]]
            prompt = self.prompt_generator(question=problem['question'], model=self.model_config.model, ret_docs=truncated_docs)
            prompts.append(prompt)
            problem_ids.append(problem['qs_id'])



        if test_prompt:
            # if self.model_config == LLMSettings().LLAMAConfigs().llama_old_qa:
            #     prompt_lengths = get_docs_tokens(prompts, model='llama2-13b')
            #     for length in prompt_lengths:
            #         if length > 4096: print(length)
            if 'gpt' in self.model_config.model:
                print(prompts[0][0]['content'])
                print(prompts[0][1]['content'])
            elif 'llama' in self.model_config.model.lower():
                print(prompts[0])
            return

        # default result path for different k
        if result_path is None:
            model_name_for_path = self.model_names_for_path[self.model_config.model]
            result_path = f'../data/{self.dataset}/new_results/DocNum/{k}_{model_name_for_path}.json'
            os.makedirs(result_path.rsplit('/', 1)[0], exist_ok=True)
        if os.path.exists(result_path):
            print('result already exists in path {}, if want to overwrite, please delete it first'.format(
                result_path))
            # pred_eval_new(self.dataset, result_path=result_path)
            return

        f"""Generate responses using RAG with {k} only"""
        print(f"🤖 Generating RAG wth {k} responses for {len(self.problems)} questions...")


        # Batch API call
        if 'gpt' in self.model_config.model:
            llm_responses = self.llm_provider.batch_generate(
                prompts=prompts,
                return_type="text",
                include_logits=True,
                custom_id_prefix=f"single_{self.dataset}_{self.model_config.model}"
            )
        elif 'llama' in self.model_config.model.lower():
            llm_responses = self.llm_provider.generate_batch(
                prompts=prompts,
                return_type="text",
                include_logits=True
            )
        else:
            raise Exception('unsupported model')

        # Process results
        results = []
        for problem_id, response in zip(problem_ids , llm_responses):
            results.append({
                'qs_id': problem_id,
                'method': 'recall_llm',
                'response': response.get('text', ''),
                'logprobs': response.get('logprobs', []),
            })

        print(f"✅ Generated {len(results)} Recall Analysis LLM responses")

        os.makedirs(result_path.rsplit('/', 1)[0], exist_ok=True)
        with open(result_path, 'w+') as f:
            json.dump(results, f, indent=2)



    def generate_with_prompt_method(self, prompt_method, result_path=None, test_prompt=False):
        if self.dataset in ['conala', 'DS1000', 'pandas_numpy_eval']:
            k = 5
        else:
            k = 10
        self.prompt_methods = ['few-shot', 'emotion', 'CoT', 'zero-shot-CoT', 'Least-to-Most', 'Plan-and-Solve', 'self-refine', 'CoN']
        assert prompt_method in self.prompt_methods

        if prompt_method == 'few-shot':
            self.prompt_generator = self.prompt_utils.prompt_3shot
        elif prompt_method == 'emotion':
            self.prompt_generator = self.prompt_utils.prompt_emotion
        elif prompt_method == 'CoT':
            self.prompt_generator = self.prompt_utils.prompt_cot
        elif prompt_method == 'zero-shot-CoT':
            self.prompt_generator = self.prompt_utils.prompt_zero_shot_cot
        elif prompt_method == 'Least-to-Most':
            self.prompt_generator = self.prompt_utils.prompt_least_to_most
        elif prompt_method == 'Plan-and-Solve':
            self.prompt_generator = self.prompt_utils.prompt_plan_and_solve
        elif prompt_method == 'self-refine':
            self.prompt_generator = self.prompt_utils.prompt_self_refine
        elif prompt_method == 'CoN':
            self.prompt_generator = self.prompt_utils.prompt_con
        else:
            raise Exception(f'Unsupported Prompt Method {prompt_method}')

        # Prepare messages
        prompts = []
        problem_ids = []

        ret_docs = self.doc_loader.get_ret_docs()

        # if prompt method is self-refine, load initial results from doc num results
        if prompt_method == 'self-refine':
            if self.dataset == 'conala':
                from generator_deprecated.pred_eval import parsing_for_conala_new as result_parser
            elif self.dataset == 'DS1000':
                from generator_deprecated.pred_eval import parsing_for_ds1000_new as result_parser
            elif self.dataset == 'pandas_numpy_eval':
                from generator_deprecated.pred_eval import parsing_for_pne_new as result_parser
            elif self.dataset in ['NQ', 'TriviaQA', 'hotpotQA']:
                from generator_deprecated.pred_eval import parsing_for_qa_new as result_parser
            else:
                raise Exception('Unsupported Dataset')
            initial_results = json.load(open(f'../data/{self.dataset}/new_results/DocNum/{k}_{self.model_names_for_path[self.model_config.model]}.json', 'r'))
            initial_results = result_parser(qs_list=self.problems, model=self.model_config.model, prompt_method='initial_output', results=initial_results)

        for idx, problem in enumerate(self.problems):
            ret_docs_exist = False
            # ret docs' qs_id may be different from oracle
            for qs_id in ret_docs:
                if qs_id == problem['qs_id']:
                    ret_docs_exist = True
                    break
            if not ret_docs_exist: raise Exception(f'no ret docs for problem: {qs_id}')
            # use top-k docs as retrieved docs
            if self.dataset in ['conala', 'DS1000', 'pandas_numpy_eval']:
                truncated_docs = truncate_docs(ret_docs[qs_id][:k], model='gpt-3.5-turbo-0125', max_length=500)
            else:
                truncated_docs = [item['doc'] for item in ret_docs[qs_id][:k]]
            if prompt_method == 'self-refine':
                assert problem['qs_id'] == initial_results[idx]['qs_id']
                prompt = self.prompt_generator(question=problem['question'], model=self.model_config.model, ret_docs=truncated_docs, initial_output=initial_results[idx]['outputs'][0])
            else:
                prompt = self.prompt_generator(question=problem['question'], model=self.model_config.model, ret_docs=truncated_docs)
            prompts.append(prompt)
            problem_ids.append(problem['qs_id'])

        if test_prompt:
            if 'gpt' in self.model_config.model:
                print(prompts[0][0]['content'])
                print(prompts[0][1]['content'])
            elif 'llama' in self.model_config.model.lower():
                print(prompts[0])
            return

        if result_path is None:
            model_name_for_path = self.model_names_for_path[self.model_config.model]
            result_path = f'../data/{self.dataset}/new_results/Prompt/{prompt_method}_{model_name_for_path}.json'
            os.makedirs(result_path.rsplit('/', 1)[0], exist_ok=True)
        if os.path.exists(result_path):
            print('result already exists in path {}, if want to overwrite, please delete it first'.format(result_path))
            # pred_eval_new(self.dataset, result_path=result_path)
            return

        f"""Generate responses using RAG with {prompt_method} only"""
        print(f"🤖 Generating RAG wth {prompt_method} responses for {len(self.problems)} questions...")


        # Batch API call
        if 'gpt' in self.model_config.model:
            llm_responses = self.llm_provider.batch_generate(
                prompts=prompts,
                return_type="text",
                include_logits=True,
                custom_id_prefix=f"single_{self.dataset}_{self.model_config.model}"
            )
        elif 'llama' in self.model_config.model.lower():
            llm_responses = self.llm_provider.generate_batch(
                prompts=prompts,
                return_type="text",
                include_logits=True
            )
        else:
            raise Exception('unsupported model')

        # Process results
        results = []
        for problem_id, response in zip(problem_ids , llm_responses):
            results.append({
                'qs_id': problem_id,
                'method': 'recall_llm',
                'response': response.get('text', ''),
                'logprobs': response.get('logprobs', []),
            })
            
        print(f"✅ Generated {len(results)} Recall Analysis LLM responses")

        os.makedirs(result_path.rsplit('/', 1)[0], exist_ok=True)
        with open(result_path, 'w+') as f:
            json.dump(results, f, indent=2)


# Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Dataset (conala, DS1000)')
    parser.add_argument('--model', required=True, help='Model (openai-new, claude)')
    parser.add_argument('--mode', required=True, choices=['single', 'oracle', 'recall', 'DocNum', 'prompt'])
    parser.add_argument('--recall', type=float, default=1, help='Recall, only effective if mode is "recall"')
    parser.add_argument('--k', type=int, default=1, help='Doc Num, only effective if mode is "DocNum"')
    parser.add_argument('--prompt', type=str, default='zero-shot', choices=['few-shot', 'emotion', 'CoT', 'zero-shot-CoT', 'Least-to-Most', 'Plan-and-Solve', 'self-refine', 'CoN'])
    parser.add_argument('--test-prompt', action='store_true')

    args = parser.parse_args()

    evaluator = LLMOracleEvaluator(dataset=args.dataset, model=args.model)

    if args.mode == 'single':
        evaluator.generate_single_llm(test_prompt=args.test_prompt)
    elif args.mode == 'oracle':
        evaluator.generate_with_oracle(test_prompt=args.test_prompt)
    elif args.mode == 'recall':
        evaluator.generate_with_recall(test_prompt=args.test_prompt, recall=args.recall)
    elif args.mode == 'DocNum':
        evaluator.generate_with_k(k=args.k, test_prompt=args.test_prompt)
    elif args.mode == 'prompt':
        evaluator.generate_with_prompt_method(prompt_method=args.prompt, test_prompt=args.test_prompt)

