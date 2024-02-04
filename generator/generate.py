import os
import json
import shlex
import tiktoken
import argparse
import random
from tqdm import tqdm
from run_model import chatgpt
from prompt import conala_prompt, tldr_prompt
from dataset.dataset_configs import TldrLoader, ConalaLoader
from retriever.sparse_retriever import sparse_retriever_config
from retriever.dense_retriever import dense_retriever_config



def approximate_token(prompts, model='gpt-3.5-turbo'):
    max_tokens, avg_tokens = 0, 0
    encoding = tiktoken.encoding_for_model(model)
    for prompt in prompts:
        tokens = len(encoding.encode(prompt))
        avg_tokens += tokens
        if tokens > max_tokens: max_tokens = tokens
    avg_tokens = avg_tokens / len(prompts)
    print(f"Average tokens: {avg_tokens:.3f}, Max tokens: {max_tokens}")


def truncate_too_long_doc(doc, model='gpt-3.5-turbo', max_length=1000):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    encoded_doc = encoding.encode(doc)
    if len(encoded_doc) > max_length:
        encoded_doc = encoded_doc[:max_length]
        doc = encoding.decode(encoded_doc)
    return doc


def get_dummy_text():   # todo: could be docs from another corpus like wiki or just random words
    return ''


def gene_tldr(args, retriever_args):
    # load docs
    tldr_loader     = TldrLoader()
    doc_list_whole  = tldr_loader.load_doc_list_whole()
    doc_list_line   = tldr_loader.load_doc_list_line()
    qs_list         = tldr_loader.load_qs_list(args.dataset_type)
    oracle_list     = tldr_loader.load_oracle_list(args.dataset_type)
    if args.retriever == 'bm25':
        ret_result_whole = json.load(open(retriever_args.tldr_ret_result_whole, 'r'))
        ret_result_line = json.load(open(retriever_args.tldr_ret_result_line, 'r'))
    else:
        raise Exception('unfinished retriever type')

    print('qs_num:', len(qs_list))
    print('save_to:', args.save_file)

    gene_results = list()
    prompts = list()
    for idx, qs, oracle in tqdm(enumerate(zip(qs_list, oracle_list))):
        if oracle['doc_key'] == 'xkcdpass': continue  # cmd not exists in docs
        qs_id = qs['qs_id']
        assert qs_id == oracle['qs_id']

        # prepare retrieved docs
        # todo: now attach retrieved docs in line level
        # res whole: {'qs_id': [{'key1', 'score1'}, ...]} res line {'qs_id': [[{'key1_1', 'score1_1'}, ...], ...]
        if args.ret_doc_type == 'oracle':
            ret_cmd_line = oracle['line_keys'][:args.k_line]
        elif args.ret_doc_type == 'retrieved':
            ret_cmd_line = list()
            for k in range(0, args.top_k):
                ret_cmd_line.extend([ret_result['doc_key'] for ret_result in ret_result_line[qs_id][k][0:args.k_line]])
        elif args.ret_doc_type == 'related':
            ret_cmd_idx = [idx for idx, ret_result in enumerate(ret_result_whole[qs_id]) if ret_result['doc_key'] not in oracle['doc_keys']][:args.top_k] # get top k cmd expect oracle
            ret_cmd_line = list()
            for cmd_idx in enumerate(ret_cmd_idx):
                assert ret_result_line[qs_id][cmd_idx][0]['doc_key'].split('_')[0] == ret_result_whole[qs_id][cmd_idx]['doc_key']
                ret_cmd_line.extend([ret_result['doc_key'] for ret_result in ret_result_line[qs_id][cmd_idx][0:args.k_line]])
        elif args.ret_doc_type == 'random':
            doc_key_whole_list = list(doc_list_whole.keys())
            # doc_key_whole_list.remove(oracle_cmd)
            random_cmds = random.sample(doc_key_whole_list, args.top_k)  # get top k random cmds
            ret_cmd_line = list()
            for random_cmd in random_cmds:
                cmd_key_list = [key for key in doc_list_line.keys() if key.startswith(random_cmd)]
                ret_cmd_line.extend(random.sample(cmd_key_list, args.k_line if len(cmd_key_list) > args.k_line else len(cmd_key_list))) # get random k lines for each cmd
        elif args.ret_doc_type == 'unrelated':
            ret_cmd_line = []
        elif args.ret_doc_type == 'none':
            ret_cmd_line = []
        else:
            raise Exception('no such ret doc type')
        ret_docs = list()
        for line_idx, cmd_line in enumerate(ret_cmd_line):
            ret_docs.append(f"potential document {line_idx}: {cmd_line}: {doc_list_line[cmd_line]}")
        if args.retriever == 'unrelated':
            ret_docs = get_dummy_text()

        def prepare_prompt(args):
            if args.retriever == 'none':
                if args.prompt_type == 'original':
                    prompt = tldr_prompt.tldr_original_no_retrieval_prompt
                elif args.prompt_type == 'instruct':
                    prompt = tldr_prompt.tldr_no_retrieval_prompt_with_instruction
                else:
                    raise Exception('no such prompt type for non-retrieval')
            else:
                if args.prompt_type == '0shots':
                    prompt = tldr_prompt.tldr_0shot_prompt
                elif args.prompt_type == 'original':
                    prompt = tldr_prompt.tldr_original_3shots_prompt
                elif args.prompt_type == 'instruct':
                    prompt = tldr_prompt.tldr_3shots_prompt_with_instruction
                else:
                    raise Exception('no such prompt type')
            prompt += '\n'
            for doc in ret_docs:
                prompt += doc
                prompt += '\n'
            prompt += f'# {qs}'
            return prompt
        prompt = prepare_prompt(args)

        # gene response
        prompts.append(prompt)
        output = chatgpt(prompt=prompt, model=args.model, temperature=args.temperature, max_tokens=args.max_tokens)[0].replace('\n',' ').replace('#END','')
        gene_results.append(dict(nl=qs, output=output, ret_cmd=ret_cmd_line, oracle_cmd=oracle['doc_keys'][0], oracle_output=oracle['output'], oracle_ret=oracle['line_keys']))
        # print(prompt)
        # print(output)

    # count tokens
    approximate_token(prompts)

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



def gene_conala(args, retriever_args):
    # load docs
    conala_loader = ConalaLoader()
    doc_list = conala_loader.load_doc_list()
    qs_list = conala_loader.load_qs_list(args.dataset_type)
    oracle_list = conala_loader.load_oracle_list(args.dataset_type)
    if args.retriever == 'bm25':
        ret_result = json.load(open(retriever_args.conala_ret_result, 'r'))
    elif 'codeT5' in args.retriever:
        ret_result = json.load(open(retriever_args.save_file, 'r'))
    else:
        raise Exception('retriever type not supported')

    print('qs_num:', len(qs_list))
    print('save_to:', args.save_file)

    gene_results = list()
    prompts = list()
    for idx, qs, oracle in tqdm(enumerate(zip(qs_list, oracle_list))):
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
            ret_docs[line_idx] = ret_docs[line_idx].replace('\n',' ')
        if args.ret_doc_type == 'unrelated':
            ret_docs = get_dummy_text()

        def prepare_prompt(args):
            if args.retriever == 'none':
                if args.prompt_type == 'original':
                    prompt = conala_prompt.conala_original_no_retrieval_prompt
                else:
                    raise Exception('no such prompt type for non-retrieval')
            else:
                if args.prompt_type == '0shots':
                    prompt = conala_prompt.conala_0shots_prompt
                elif args.prompt_type == 'original':
                    prompt = conala_prompt.conala_original_3shots_prompt
                else:
                    raise Exception('no such prompt type')
            prompt += '\n'
            for doc in ret_docs:
                doc = truncate_too_long_doc(doc)
                prompt += doc
                prompt += '\n'
            prompt += f'# {qs}'
            return prompt
        prompt = prepare_prompt(args)

        # gene response
        prompts.append(prompt)
        output = chatgpt(prompt=prompt, model=args.model, temperature=args.temperature, max_tokens=args.max_tokens)[0].replace('\n', ' ').replace('#END', '')
        gene_results.append(dict(nl=qs, output=output, ret_libs=ret_libs, oracle_libs=oracle['doc_keys'], oracle_output=oracle['output']))

    # count tokens
    approximate_token(prompts)

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


def generate_config(in_program_call=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tldr')
    parser.add_argument('--top_k', type=int, default=1)     # k docs
    parser.add_argument('--k_line', type=int, default=5)    # for tldr
    parser.add_argument('--retriever', type=str, default='bm25',
                        choices=['bm25', 'codeT5-FT', 'codeT5-OTS'])
    parser.add_argument('--ret_doc_type', type=str, default='retrieved',
                        choices=['oracle', 'retrieved', 'related', 'random', 'unrelated', 'none'])
    parser.add_argument('--prompt_type', type=str, default='original',
                        choices=['0shots', 'instruct', 'CoT'])
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-1106')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--save_file', type=str, default=None)
    parser.add_argument('--max_tokens', type=int, default=1000)

    args = parser.parse_args() if in_program_call is None else parser.parse_args(shlex.split(in_program_call))
    if args.save_file is None and args.dataset == 'tldr':
        args.save_file = (f'docprompting_data/conala/gene_result_'
                          f'model_{args.model}_'
                          f'retriever_{args.retriever}_{args.ret_doc_type}_'
                          f'prompt_type_{args.prompt_type}_'
                          f'top_k_{args.top_k}_k_line_{args.k_line}.json')
    elif args.save_file is None and args.dataset == 'conala':
        args.save_file = (f'docprompting_data/conala/gene_result_'
                          f'model_{args.model}_'
                          f'retriever_{args.retriever}_{args.ret_doc_type}_'
                          f'prompt_type_{args.prompt_type}_'
                          f'top_k_{args.top_k}.json')

    print(json.dumps(vars(args), indent=2))
    return args


if __name__ == '__main__':
    in_program_call = '--dataset tldr --top_k 1 --k_line 5 --retriever bm25'
    # in_program_call = '--dataset conala --top_k 5 --retriever codeT5-OTS'
    args = config(in_program_call)
    if args.retriever == 'bm25':
        retriever_args = sparse_retriever_config('')
    elif args.retriever == 'codeT5-FT':
        retriever_args = dense_retriever_config(f"--dataset conala \
                        --model_name neulab/docprompting-codet5-python-doc-retriever")
    elif args.retriever == 'codeT5-OTS':
        retriever_args = dense_retriever_config(f"--dataset conala \
                        --model_name Salesforce/codet5-base")
    else:
        retriever_args = None

    if args.dataset == 'tldr':
        gene_tldr(args, retriever_args)
    elif args.dataset == 'conala':
        gene_conala(args, retriever_args)