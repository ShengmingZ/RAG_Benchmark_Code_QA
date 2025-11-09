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
        self.llm_provider = OpenAIProvider(organization=self.model_config.organization,
                                           model=self.model_config.model,
                                           temperature=self.model_config.temperature,
                                           max_tokens=self.model_config.max_tokens,
                                           is_async=self.model_config.is_async,
                                           stop=None)

        self.dataset = dataset

        self.doc_loader = RetrievalProvider(self.dataset)

        self.oracle_docs = self.doc_loader.get_oracle_docs()

        if self.dataset == 'conala':
            self.problems = ConalaLoader().load_qs_list()
            self.prompt_generator = conala_prompt.prompt_0shot
            self.prompt_generator_no_ret = conala_prompt.prompt_0shot_no_ret
        elif self.dataset == 'DS1000':
            self.problems = DS1000Loader().load_qs_list()
            self.prompt_generator = DS1000_prompt.prompt_0shot
            self.prompt_generator_no_ret = DS1000_prompt.prompt_0shot_no_ret
        elif self.dataset == 'pandas_numpy_eval':
            self.problems = PandasNumpyEvalLoader().load_qs_list()
            self.prompt_generator = pandas_numpy_eval_prompt.prompt_0shot
            self.prompt_generator_no_ret = pandas_numpy_eval_prompt.prompt_0shot_no_ret
        elif self.dataset == 'NQ' or self.dataset == 'TriviaQA':
            self.problems = NQTriviaQAUtils(dataset=self.dataset).load_qs_list()
            self.prompt_generator = NQ_TriviaQA_prompt.prompt_0shot
            self.prompt_generator_no_ret = NQ_TriviaQA_prompt.prompt_0shot_no_ret
        elif self.dataset == 'hotpotQA':
            self.problems = HotpotQAUtils().load_qs_list()
            self.prompt_generator = hotpotQA_prompt.prompt_0shot
            self.prompt_generator_no_ret = hotpotQA_prompt.prompt_0shot_no_ret
        else:
            raise Exception('unsupported dataset')

        self.recall_range = [0, 0.2, 0.4, 0.6, 0.8, 1.0]


    def generate_with_retrieval_recall(self, recall, result_path=None, test_prompt=False):
        assert recall in self.recall_range
        # default result path
        if result_path is None and recall != 1.0: result_path = f'../data/{self.dataset}/new_results/recall-{recall}_{self.model_config.model}.json'
        elif result_path is None and recall == 1.0: result_path = f'../data/{self.dataset}/new_results/oracle_{self.model_config.model}.json'
        if os.path.exists(result_path):
            print('result already exists in path {}, if want to overwrite, please delete it first'.format(result_path))
            pred_eval_new(self.dataset, result_path=result_path)
            return

        print(f"🤖 Generating Retrieval Recall LLM responses for {len(self.problems)} questions...")

        # Prepare messages
        prompts = []
        problem_ids = []

        oracle_docs = self.doc_loader.get_oracle_docs()
        ret_docs = self.doc_loader.get_ret_docs()

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
            return

        # Batch API call
        if 'gpt' in self.model_config.model:
            llm_responses = self.llm_provider.batch_generate(
                prompts=prompts,
                return_type="text",
                include_logits=True,
                custom_id_prefix=f"single_{self.dataset}_{self.model_config.model}"
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




# Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Dataset (conala, DS1000)')
    parser.add_argument('--model', required=True, help='Model (openai-new, claude)')
    parser.add_argument('--mode', required=True, choices=['single', 'oracle', 'both'])
    parser.add_argument('--test-prompt', action='store_true')

    args = parser.parse_args()

    evaluator = LLMOracleEvaluator(dataset=args.dataset, model=args.model)

    if args.mode == 'single':
        evaluator.generate_single_llm(test_prompt=args.test_prompt)
    elif args.mode == 'oracle':
        evaluator.generate_with_oracle(test_prompt=args.test_prompt)
