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


def get_dummy_text():
    pass


def gene_tldr(args, retriever_args):
    # load docs
    tldr_loader = TldrLoader()
    doc_list_whole = tldr_loader.load_doc_list_whole()
    doc_list_line = tldr_loader.load_doc_list_line()
    qs_list = tldr_loader.load_qs_list(args.dataset_type)
    oracle_list = tldr_loader.load_oracle_list(args.dataset_type)

    def load_retrieve_results(retriever_args):
        ret_result_whole, ret_result_line = None, None
        if args.retriever == 'bm25':
            ret_result_whole = json.load(open(retriever_args.tldr_ret_result_whole, 'r'))
            ret_result_line = json.load(open(retriever_args.tldr_ret_result_line, 'r'))
        #elif args.retriever == 'dense':
        return ret_result_whole, ret_result_line
    ret_result_whole, ret_result_line = load_retrieve_results(retriever_args)

    print('qs_num:', len(qs_list))
    print('save_to:', args.save_file)

    gene_results = list()
    prompts = list()
    for idx, qs, oracle in tqdm(enumerate(zip(qs_list, oracle_list))):
        assert qs['qs_id'] == oracle['qs_id']
        # cmd not exists in docs
        if oracle['doc_key'] == 'xkcdpass': continue
        oracle_output = oracle['output']
        oracle_cmd = oracle['doc_keys'][0]
        oracle_ret = oracle['line_keys']

        # prepare retrieved docs, todo: now attach retrieved docs in line level
        if args.retriever == 'oracle':
            oracle_ret = oracle_ret[:args.k_line]
            ret_cmd_line = oracle_ret
        elif args.retriever == 'bm25':
            ret_cmd_line = list()
            for k in range(0, args.top_k):
                ret_cmd_line.extend([ret_result['doc_key'] for ret_result in ret_result_line[qs['qs_id']][k][0:args.k_line]])
        elif args.retriever == 'related':
            # res whole: {'qs_id': [{'key1', 'score1'}, ...]} res line {'qs_id': [[{'key1_1', 'score1_1'}, ...], ...]
            ret_cmd_idx = [idx for idx, ret_result in enumerate(ret_result_whole[qs['qs_id']]) if ret_result['doc_key'] != oracle_cmd][:args.top_k]
            ret_cmd_line = list()
            for cmd_idx in enumerate(ret_cmd_idx):
                assert ret_result_line[qs['qs_id']][cmd_idx][0]['doc_key'].split('_')[0] == ret_result_whole[qs['qs_id']][cmd_idx]['doc_key']
                ret_cmd_line.extend([ret_result['doc_key'] for ret_result in ret_result_line[qs['qs_id']][cmd_idx][0:args.k_line]])
        elif args.retriever == 'random':
            # get top k random cmds
            doc_key_whole_list = list(doc_list_whole.keys())
            # doc_key_whole_list.remove(oracle_cmd)
            random_cmds = random.sample(doc_key_whole_list, args.top_k)
            # get random k lines for each cmd
            ret_cmd_line = list()
            for random_cmd in random_cmds:
                cmd_key_list = [key for key in doc_list_line.keys() if key.startswith(random_cmd)]
                ret_cmd_line.extend(random.sample(cmd_key_list, args.k_line if len(cmd_key_list) > args.k_line else len(cmd_key_list)))
        elif args.retriever == 'unrelated':
            ret_cmd_line = []
        elif args.retriever == 'none':
            ret_cmd_line = []
        else:
            raise Exception('no such retriever')
        ret_docs = list()
        for line_idx, cmd_line in enumerate(ret_cmd_line):
            ret_docs.append(f"potential document {line_idx}: {cmd_line}: {doc_list_line[cmd_line]}")
        if args.retriever == 'unrelated':
            ret_cmd_line = get_dummy_text()

        def prepare_prompt(args):
            if args.zero_shot is True:
                prompt = tldr_prompt.tldr_0shot_prompt
            elif args.retriever == 'none':
                prompt = tldr_prompt.tldr_original_no_retrieval_prompt
            else:
                prompt = tldr_prompt.tldr_original_3shots_prompt
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
        gene_results.append(dict(nl=qs, output=output, ret_cmd=ret_cmd_line, oracle_cmd=oracle_cmd, oracle_output=oracle_output, oracle_ret=oracle_ret))
        # print(prompt)
        # print(output)
    # approximate_token(prompts)

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
    # load retrieve results and docs
    conala_args = conala_config()
    conala_qs = []
    with open(conala_args.qs_file, 'r') as f:
        for line in f:
            conala_qs.append(line.strip())
    conala_qs_id = []
    with open(conala_args.qs_idx_file, 'r') as f:
        for line in f:
            conala_qs_id.append(line.strip())
    oracle_results = json.load(open(conala_args.oracle, 'r'))
    if args.passk is not None:

    conala_docs = json.load(open(conala_args.doc_file, 'r'))
    if args.retriever == 'bm25':
        ret_result_bm25 = json.load(open(retriever_args.conala_ret_result, 'r'))
    elif 'codeT5' in args.retriever:
        ret_result_dense = json.load(open(retriever_args.save_file, 'r'))
    conala_prompt = ConalaPrompt()

    print('qs_num:', len(oracle_results))
    print('save_to:', args.save_file)

    gene_results = list()
    prompts = list()
    for item in tqdm(oracle_results):
        qs_id = item['question_id']
        qs = item['nl']
        oracle_output = item['cmd']
        oracle_libs = item['oracle_man']

        # prepare retrieved docs
        if args.retriever == 'bm25':
            ret_libs = [result['lib_key'] for result in ret_result_bm25[qs_id][0:args.top_k]]
        elif 'codeT5' in args.retriever:
            ret_libs = [result['lib_key'] for result in ret_result_dense[qs_id][0:args.top_k]]
        elif args.retriever == 'oracle':
            ret_libs = item['oracle_man']
        elif args.retriever == 'none':
            ret_libs = []
        elif args.retriever == 'unrelated':
            doc_key_list = list(conala_docs.keys())
            doc_key_list = [item for item in doc_key_list if item not in oracle_libs]
            ret_libs = random.sample(doc_key_list, args.top_k)
        ret_docs = list()
        for line_idx, ret_lib in enumerate(ret_libs):
            ret_docs.append(f"potential document {line_idx}: {ret_lib}: {conala_docs[ret_lib]}")
            ret_docs[line_idx] = ret_docs[line_idx].replace('\n',' ')

        # prepare prompt
        if args.zero_shot is True:
            prompt = conala_prompt.conala_0shots_prompt
        elif args.retriever == 'none':
            prompt = conala_prompt.conala_original_no_retrieval_prompt
        else:
            prompt = conala_prompt.conala_original_3shots_prompt
        prompt += '\n'
        for doc in ret_docs:
            doc = truncate_too_long_doc(doc)
            prompt += doc
            prompt += '\n'
        prompt += f'# {qs}'

        # gene response
        prompts.append(prompt)
        output = chatgpt(prompt=prompt, model=args.model, temperature=args.temperature, max_tokens=args.max_tokens)[0].replace('\n', ' ').replace('#END', '')
        gene_results.append(dict(nl=qs, output=output, ret_libs=ret_libs, oracle_libs=oracle_libs, oracle_output=oracle_output))
    # approximate_token(prompts)

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


def config(in_program_call=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tldr')
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--k_line', type=int, default=5)
    parser.add_argument('--retriever', type=str, default='bm25',
                        choices=['bm25', 'codeT5-FT', 'codeT5-OTS'])
    parser.add_argument('--ret_doc_type', type=str, default='retrieved',
                        choices=['oracle', 'retrieved', 'related', 'random', 'unrelated', 'none'])
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-1106')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--save_file', type=str, default=None)
    parser.add_argument('--passk', type=int, default=None)
    parser.add_argument('--max_tokens', type=int, default=1000)
    parser.add_argument('--zero_shot', action='store_true')

    args = parser.parse_args() if in_program_call is None else parser.parse_args(shlex.split(in_program_call))
    if args.save_file is None and args.dataset == 'tldr':
        args.save_file = (f'docprompting_data/conala/gene_result_model_{args.model}_'
                          f'retriever_{args.retriever}_top_k_{args.top_k}_k_line_{args.k_line}_'
                          f'zero_shot_{args.zero_shot}.json')
    elif args.save_file is None and args.dataset == 'conala':
        args.save_file = (f'docprompting_data/conala/gene_result_model_{args.model}_'
                          f'retriever_{args.retriever}_top_k_{args.top_k}_zero_shot_{args.zero_shot}.json')

    print(json.dumps(vars(args), indent=2))
    return args


if __name__ == '__main__':
    """
    none: no retrieve docs
    unrelated: docs not related to the NL
    relevant: docs somehow related to the NL but not correct
    """
    in_program_call = '--dataset conala --top_k 5 --retriever codeT5-OTS'
    args = config(in_program_call)
    if args.retriever == 'bm25':
        retriever_args = sparse_retriever_config('')
    elif args.retriever in ['none', 'unrelated', 'oracle', 'dummy']:
        retriever_args = None
    elif args.retriever == 'codeT5-FT':
        retriever_args = dense_retriever_config(f"--dataset conala \
                        --model_name neulab/docprompting-codet5-python-doc-retriever \
                        --sim_func cls_distance.cosine")
    elif args.retriever == 'codeT5-OTS':
        retriever_args = dense_retriever_config(f"--dataset conala \
                        --model_name Salesforce/codet5-base \
                        --sim_func cls_distance.cosine")

    if args.dataset == 'tldr':
        gene_tldr(args, retriever_args)
    elif args.dataset == 'conala':
        gene_conala(args, retriever_args)