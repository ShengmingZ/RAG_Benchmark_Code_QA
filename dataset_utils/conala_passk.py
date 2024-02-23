import platform
import sys
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
import evaluate
from tqdm import tqdm
from retriever.dense_retriever import dense_retriever_config
from generator.generate import gene_conala, generate_config
import json
from dataset_utils.dataset_configs import ConalaLoader
import random
from generator.run_model import chatgpt
from generator.generate import get_dummy_text, truncate_too_long_doc
from prompt import conala_prompt
from datasets import load_metric
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

unittest_file = os.path.join(root_path, 'docprompting_data/conala/unittest_docprompting_conala.json')

def get_unittest_qs_oracle():
    unittest = json.load(open(unittest_file, 'r'))
    conala_loader = ConalaLoader()
    qs_list = conala_loader.load_qs_list('test')
    oracle_list = conala_loader.load_oracle_list('test')
    _qs_list, _oracle_list = list(), list()
    for qs_id in unittest.keys():
        for (qs, oracle) in zip(qs_list, oracle_list):
            if qs['qs_id'] == qs_id:
                assert oracle['qs_id'] == qs_id
                _qs_list.append(qs)
                _oracle_list.append(oracle)
                continue
        assert _qs_list[-1]['qs_id'] == qs_id
    return _qs_list, _oracle_list

def gene_k(args, retriever_args):
    # load corresponding 100 docs
    qs_list, oracle_list = get_unittest_qs_oracle()
    conala_loader = ConalaLoader()
    doc_list = conala_loader.load_doc_list()
    ret_result = json.load(open(retriever_args.save_file, 'r'))

    gene_results = list()
    for idx, (qs, oracle) in tqdm(enumerate(zip(qs_list, oracle_list))):
        qs_id = qs['qs_id']
        # prepare retrieved docs
        if args.ret_doc_type == 'oracle':
            ret_libs = oracle['doc_keys']
        elif args.ret_doc_type == 'retrieved':
            ret_libs = [result['doc_key'] for result in ret_result[qs_id][0:args.top_k]]
        elif args.ret_doc_type == 'related':
            ret_libs = [result['doc_key'] for result in ret_result[qs_id] if result['doc_key'] not in oracle['doc_keys']][:args.top_k]
        elif args.ret_doc_type == 'random':
            doc_key_list = list(doc_list.keys())
            # doc_key_list = [item for item in doc_key_list if item not in oracle_libs]
            ret_libs = random.sample(doc_key_list, args.top_k)
        elif args.ret_doc_type == 'unrelated':
            ret_libs = []
        elif args.ret_doc_type == 'none':
            ret_libs = []
        else:
            raise Exception('no such ret doc type')
        ret_docs = list()
        for line_idx, ret_lib in enumerate(ret_libs):
            ret_docs.append(f"potential document {line_idx}: {ret_lib}: {doc_list[ret_lib]}")
            ret_docs[line_idx] = ret_docs[line_idx].replace('\n', ' ')
        if args.ret_doc_type == 'unrelated':
            ret_docs = get_dummy_text()

        def prepare_prompt(args):
            if args.ret_doc_type == 'none':
                if args.prompt_type == 'original':
                    prompt = conala_prompt.conala_original_no_retrieval_prompt
                elif args.prompt_type == 'instruct':
                    prompt = conala_prompt.tldr_no_retrieval_prompt_with_instruction
                else:
                    raise Exception('no such prompt type for non-retrieval')
            else:
                if args.prompt_type == '0shot':
                    prompt = conala_prompt.conala_0shot_prompt
                elif args.prompt_type == 'original':
                    prompt = conala_prompt.conala_original_3shots_prompt
                elif args.prompt_type == 'instruct':
                    prompt = conala_prompt.conala_3shots_prompt_with_instruction
                else:
                    raise Exception('no such prompt type')
            prompt += '\n'
            for doc in ret_docs:
                doc = truncate_too_long_doc(doc, max_length=args.max_doc_tokens)
                prompt += doc
                prompt += '\n'
            prompt += f'# {qs["nl"]}'
            return prompt

        prompt = prepare_prompt(args)

        # gene response
        outputs = chatgpt(prompt=prompt, model=args.model, temperature=args.temperature, max_tokens=args.max_tokens, n=args.pass_k)
        _outputs = list()
        for output in outputs:
            output = (output.replace("{{", " {{").replace("\n", ' ').replace("\r", "").replace("<pad>", "").
                   replace("<s>","").replace("</s>", "").replace('#END', '').strip())
            output = " ".join(output.split())
            _outputs.append(output)
        gene_results.append(dict(nl=qs, outputs=outputs, ret_libs=ret_libs, oracle_libs=oracle['doc_keys'], oracle_output=oracle['output']))
        if idx == 0: print(prompt)


    # save to files
    if os.path.exists(args.save_file):
        user_input = input(f'The file {args.save_file} already exists. Overwrite? (y/n): ').lower()
        if user_input == 'y':
            with open(args.save_file, 'w+') as f:
                json.dump(gene_results, f, indent=2)
            print('overwrite file done')
        else:
            print('save file not overwrite')
    else:
        with open(args.save_file, 'w+') as f:
            json.dump(gene_results, f, indent=2)


def pass_at_k(result_file):
    unittests = json.load(open(unittest_file, 'r'))
    results = json.load(open(result_file, 'r'))

    # run the test
    # load the metric from huggingface
    code_eval_metric = evaluate.load("code_eval")
    pass_k_list = []
    for idx, qs_id in tqdm(enumerate(unittests.keys())):
        unittest = unittests[qs_id]
        for result in results:
            if result['nl']['qs_id'] == qs_id:
                break
        assert result['nl']['qs_id'] == qs_id
        suffix = unittest['suffix']
        entry_point = unittest["entry_point"]
        test_func = f"\n{unittest['test']}\ncheck({entry_point})"
        runnable_func = [f"{unittest['prompt']}{x}{suffix}" for x in result['outputs']]
        # runnable_func = [f"{unittest['prompt']}{unittest['canonical_solution']}{suffix}"] # oracle

        pass_k, _ = code_eval_metric.compute(
            predictions=[runnable_func],
            references=[test_func],
            k=[1, 5, 10, 50, 100],
            # k=[1],
            num_workers=1,
        )
        print(idx, pass_k)
        pass_k_list.append(pass_k)
    _pass_k = {}
    pass_keys = list(pass_k_list[0].keys())
    for key in pass_keys: _pass_k[key] = 0
    for pass_k in pass_k_list:
        for key in pass_keys: _pass_k[key] += pass_k[key]
    for key in pass_keys: _pass_k[key] = _pass_k[key]/len(unittests)
    print(_pass_k)


if __name__ == '__main__':
    args = generate_config('--dataset conala --top_k 10 --retriever codeT5-FT --ret_doc_type oracle --prompt_type original --max_doc_tokens 4000')
    retrieval_args = dense_retriever_config(f"--dataset conala --dataset_type test \
                        --model_name neulab/docprompting-codet5-python-doc-retriever")
    args.pass_k = 100
    args.save_file = args.save_file.replace('.json', '_unittest.json')
    print('begin unittest generate, save_to:', args.save_file)
    gene_k(args, retrieval_args)

    print('begin eval')
    pass_at_k(args.save_file)
