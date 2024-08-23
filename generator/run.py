import argparse
import subprocess


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['llama3-8b', 'llama2-13b-chat', 'codellama-13b-instruct', 'gpt-3.5-turbo-0125', 'gpt-4o'])
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--dataset', type=str, default=None, choices=['conala', 'DS1000', 'pandas_numpy_eval', 'NQ', 'TriviaQA', 'hotpotQA'])
    parser.add_argument('--retriever', type=str, choices=['best', 'BM25', 'contriever', 'miniLM', 'openai-embedding'])
    parser.add_argument('--analysis_type', type=str, choices=['retrieval_recall', 'retrieval_doc_type', 'retrieval_doc_selection_topk', 'retrieval_doc_selection_pl', 'prompt_method',
                                                              'prompt_length_oracle', 'prompt_length_distracting', 'prompt_length_retrieved_top', 'prompt_length_none', 'prompt_length_irrelevant'])
    parser.add_argument('--prompt_type', type=str, default=None, choices=['3shot', 'cot', 'self-consistency', 'least_to_most', 'plan_and_solve'])
    parser.add_argument('--action', type=str, choices=['gene_prompts', 'gene_responses', 'eval_pred'])
    parser.add_argument('--n', type=int)
    parser.add_argument('--batch', action='store_true')
    args = parser.parse_args()
    return args


args = config()

# ret acc
if args.analysis_type == "retrieval_recall":
    ret_acc_list = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    cmds = []
    if args.action == 'gene_responses' and args.batch is True and 'gpt' in args.model:  # run in the same time
        for ret_acc in ret_acc_list:
            cmd = (f'python generator/generate.py --action {args.action} --model {args.model} --temperature {args.temperature} --batch '
                   f'--dataset {args.dataset} --retriever {args.retriever} --analysis_type {args.analysis_type} --n {args.n} --ret_acc {ret_acc}')
            cmds.append(cmd)
        batch_cmd = ''
        for cmd in cmds:
            batch_cmd = batch_cmd + cmd + ' & '
    else:
        for ret_acc in ret_acc_list:
            cmd = (f'python generator/generate.py --action {args.action} --model {args.model} --temperature {args.temperature} '
                   f'--dataset {args.dataset} --retriever {args.retriever} --analysis_type {args.analysis_type} --n {args.n} --ret_acc {ret_acc}')
            cmds.append(cmd)
        batch_cmd = ''
        for cmd in cmds:
            batch_cmd = batch_cmd + cmd + ' ; '

    if args.action == 'eval_pred':
        for cmd in cmds:
            proc = subprocess.Popen(cmd.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (output, error) = proc.communicate()
            output_lines = output.decode().split('\n')
            if output_lines[-1] == '': print(cmd, '\n', output_lines[-2])
            else: print(cmd, '\n', output_lines[-1])
    else:
        subprocess.check_output(batch_cmd, shell=True)
        print(f'done {args.action} for retrieval recall analysis, {args.model} {args.dataset}')


elif args.analysis_type == "retrieval_doc_type":
    ret_doc_type_list = ['oracle', 'retrieved', 'distracting', 'random', 'irrelevant_dummy', 'irrelevant_diff', 'none']
    cmds = []
    if args.action == 'gene_responses' and args.batch is True and 'gpt' in args.model:  # run in the same time
        for ret_doc_type in ret_doc_type_list:
            cmd = (f'python generator/generate.py --action {args.action} --model {args.model} --temperature {args.temperature} --batch '
                   f'--dataset {args.dataset} --retriever {args.retriever} --analysis_type {args.analysis_type} --n {args.n} --ret_doc_type {ret_doc_type}')
            cmds.append(cmd)
        batch_cmd = ''
        for cmd in cmds:
            batch_cmd = batch_cmd + cmd + ' & '
    else:       # run in a sequence
        for ret_doc_type in ret_doc_type_list:
            cmd = (f'python generator/generate.py --action {args.action} --model {args.model} --temperature {args.temperature} '
                   f'--dataset {args.dataset} --retriever {args.retriever} --analysis_type {args.analysis_type} --n {args.n} --ret_doc_type {ret_doc_type}')
            cmds.append(cmd)
        batch_cmd = ''
        for cmd in cmds:
            batch_cmd = batch_cmd + cmd + ' ; '

    if args.action == 'eval_pred':
        for cmd in cmds:
            proc = subprocess.Popen(cmd.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (output, error) = proc.communicate()
            output_lines = output.decode().split('\n')
            if output_lines[-1] == '':
                print(cmd, '\n', output_lines[-2])
            else:
                print(cmd, '\n', output_lines[-1])
    else:
        subprocess.check_output(batch_cmd, shell=True)
        print(f'done {args.action} for ret_doc_type analysis, {args.model} {args.dataset}')


elif args.analysis_type.startswith("retrieval_doc_selection"):
    if args.analysis_type.rsplit('_',1)[-1] == 'topk':
        if args.dataset in ['NQ', 'TriviaQA', 'hotpotQA']:
            if 'gpt' in args.model:
                doc_selection_type_list = ['top_1', 'top_5', 'top_10', 'top_15', 'top_20', 'top_40', 'top_60', 'top_80']
            else:
                doc_selection_type_list = ['top_1', 'top_5', 'top_10', 'top_15', 'top_20']
        else:
            doc_selection_type_list = ['top_1', 'top_5', 'top_10', 'top_15', 'top_20']
    elif args.analysis_type.rsplit('_',1)[-1] == 'pl':
        if args.dataset in ['NQ', 'TriviaQA', 'hotpotQA'] and 'llama' in args.model:
            doc_selection_type_list = ['pl_500', 'pl_1000', 'pl_1500', 'pl_2000']
        else:
            doc_selection_type_list = ['pl_500', 'pl_2000', 'pl_4000', 'pl_6000', 'pl_8000']
    else:
        raise ValueError('invalid analysis type {}'.format(args.analysis_type))
    cmds = []
    args.analysis_type = args.analysis_type.rsplit('_', 1)[0]
    if args.action == 'gene_responses' and args.batch is True and 'gpt' in args.model:
        for doc_selection_type in doc_selection_type_list:
            cmd = (f'python generator/generate.py --action {args.action} --model {args.model} --temperature {args.temperature} --batch '
                   f'--dataset {args.dataset} --retriever {args.retriever} --analysis_type {args.analysis_type} --n {args.n} --doc_selection_type {doc_selection_type}')
            cmds.append(cmd)
        batch_cmd = ''
        for cmd in cmds:
            batch_cmd = batch_cmd + cmd + ' & '
    else:
        for doc_selection_type in doc_selection_type_list:
            cmd = (f'python generator/generate.py --action {args.action} --model {args.model} --temperature {args.temperature} '
                   f'--dataset {args.dataset} --retriever {args.retriever} --analysis_type {args.analysis_type} --n {args.n} --doc_selection_type {doc_selection_type}')
            cmds.append(cmd)
        batch_cmd = ''
        for cmd in cmds:
            batch_cmd = batch_cmd + cmd + ' ; '
    if args.action == 'eval_pred':
        for cmd in cmds:
            proc = subprocess.Popen(cmd.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (output, error) = proc.communicate()
            output_lines = output.decode().split('\n')
            if output_lines[-1] == '':
                print(cmd, '\n', output_lines[-2])
            else:
                print(cmd, '\n', output_lines[-1])
    else:
        subprocess.check_output(batch_cmd, shell=True)
        print(f'done {args.action} for retrieval doc selection analysis, {args.model} {args.dataset}')


elif args.analysis_type.startswith('prompt_length'):
    target_pl = 4000
    target_type = args.analysis_type.replace('prompt_length_','')
    if target_type == 'oracle':
        pl_analysis_list = [f'oracle', f'oracle_repeat_{target_pl}', f'oracle_pad_random_{target_pl}', f'oracle_pad_repeat_random_{target_pl}', f'oracle_pad_diff_{target_pl}', f'oracle_pad_repeat_diff_{target_pl}', f'oracle_pad_dummy_{target_pl}', f'oracle_pad_ellipsis_{target_pl}',]
    elif target_type == 'distracting':
        pl_analysis_list = [f'distracting', f'distracting_repeat_{target_pl}', f'distracting_pad_random_{target_pl}', f'distracting_pad_repeat_random_{target_pl}', f'distracting_pad_diff_{target_pl}', f'distracting_pad_repeat_diff_{target_pl}', f'distracting_pad_dummy_{target_pl}', f'distracting_pad_ellipsis_{target_pl}',]
    elif target_type == 'retrieved_top':
        pl_analysis_list = [f'retrieved_top', f'retrieved_top_repeat_{target_pl}', f'retrieved_top_pad_random_{target_pl}', f'retrieved_top_pad_repeat_random_{target_pl}', f'retrieved_top_pad_diff_{target_pl}', f'retrieved_top_pad_repeat_diff_{target_pl}', f'retrieved_top_pad_dummy_{target_pl}', f'retrieved_top_pad_ellipsis_{target_pl}',]
    elif target_type == 'none':
        pl_analysis_list = [f'none', f'none_pad_random_{target_pl}', f'none_pad_repeat_random_{target_pl}', f'none_pad_diff_{target_pl}', f'none_pad_repeat_diff_{target_pl}', f'none_pad_dummy_{target_pl}', f'none_pad_ellipsis_{target_pl}']
    elif target_type == 'irrelevant':
        pl_analysis_list = [f'random', f'random_{target_pl}', f'random_repeat_{target_pl}', f'diff', f'diff_{target_pl}', f'diff_repeat_{target_pl}', f'dummy', f'dummy_{target_pl}', f'ellipsis', f'ellipsis_{target_pl}']
    # if target_type == 'oracle':
    #     pl_analysis_list = [f'oracle_repeat_{target_pl}', f'oracle_pad_diff_{target_pl}', f'oracle_pad_dummy_{target_pl}', f'oracle_pad_ellipsis_{target_pl}', ]
    # elif target_type == 'distracting':
    #     pl_analysis_list = [f'distracting_repeat_{target_pl}', f'distracting_pad_diff_{target_pl}',
    #                         f'distracting_pad_dummy_{target_pl}', f'distracting_pad_ellipsis_{target_pl}', ]
    # elif target_type == 'retrieved_top':
    #     pl_analysis_list = [f'retrieved_top_repeat_{target_pl}',f'retrieved_top_pad_diff_{target_pl}',
    #                         f'retrieved_top_pad_dummy_{target_pl}', f'retrieved_top_pad_ellipsis_{target_pl}', ]
    # elif target_type == 'none':
    #     pl_analysis_list = [f'none_pad_diff_{target_pl}',
    #                         f'none_pad_dummy_{target_pl}', f'none_pad_ellipsis_{target_pl}']
    # elif target_type == 'irrelevant':
    #     pl_analysis_list = [f'random_{target_pl}', f'random_repeat_{target_pl}', f'diff_{target_pl}',
    #                         f'diff_repeat_{target_pl}', f'dummy_{target_pl}', f'ellipsis_{target_pl}']
    args.analysis_type = 'prompt_length'
    cmds = []
    if args.action == 'gene_responses' and args.batch is True and 'gpt' in args.model:  # use batch, run simo
        for pl_analysis in pl_analysis_list:
            if str(target_pl) not in pl_analysis:
                if pl_analysis == 'diff' or pl_analysis == 'dummy': pl_analysis = 'irrelevant_' + pl_analysis
                cmd = (f'python generator/generate.py --action {args.action} --model {args.model} --temperature {args.temperature} --batch '
                       f'--dataset {args.dataset} --retriever {args.retriever} --analysis_type retrieval_doc_type --n {args.n} --ret_doc_type {pl_analysis}')
            else:
                cmd = (f'python generator/generate.py --action {args.action} --model {args.model} --temperature {args.temperature} --batch '
                       f'--dataset {args.dataset} --retriever {args.retriever} --analysis_type {args.analysis_type} --n {args.n} --pl_analysis {pl_analysis}')
                if 'ellipsis' in pl_analysis: cmd = cmd.replace('--batch', '')  # for ellipsis, batch file too large
            cmds.append(cmd)
        batch_cmd = ''
        for cmd in cmds:
            batch_cmd = batch_cmd + cmd + ' & '
    else:
        for pl_analysis in pl_analysis_list:
            if target_pl == 0:
                pl_analysis = pl_analysis.rsplit('_', 1)[0]
                cmd = (f'python generator/generate.py --action {args.action} --model {args.model} --temperature {args.temperature} '
                       f'--dataset {args.dataset} --retriever {args.retriever} --analysis_type retrieval_doc_type --n {args.n} --ret_doc_type {pl_analysis}')
            else:
                cmd = (f'python generator/generate.py --action {args.action} --model {args.model} --temperature {args.temperature} '
                       f'--dataset {args.dataset} --retriever {args.retriever} --analysis_type {args.analysis_type} --n {args.n} --pl_analysis {pl_analysis}')
            cmds.append(cmd)
        batch_cmd = ''
        for cmd in cmds:
            batch_cmd = batch_cmd + cmd + ' ; '
    if args.action == 'eval_pred':
        for cmd in cmds:
            proc = subprocess.Popen(cmd.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (output, error) = proc.communicate()
            output_lines = output.decode().split('\n')
            if output_lines[-1] == '':
                print(cmd, '\n', output_lines[-2])
            else:
                print(cmd, '\n', output_lines[-1])
    else:
        subprocess.check_output(batch_cmd, shell=True)
        print(f'done {args.action} for prompt length analysis, {args.model} {args.dataset}')


elif args.analysis_type == 'prompt_method':     # run by dataset
    dataset_names = ['NQ', 'TriviaQA', 'hotpotQA', 'conala', 'DS1000', 'pandas_numpy_eval']
    cmds = []
    if args.action == 'gene_responses' and args.batch is True and 'gpt' in args.model:
        for dataset_name in dataset_names:
            cmd = (f'python generator/generate.py --action {args.action} --model {args.model} --temperature {args.temperature} --batch '
                   f'--dataset {dataset_name} --retriever {args.retriever} --analysis_type {args.analysis_type} --n {args.n} --prompt_type {args.prompt_type}')
            cmds.append(cmd)
        batch_cmd = ''
        for cmd in cmds:
            batch_cmd = batch_cmd + cmd + ' & '
    else:
        for dataset_name in dataset_names:
            if 'llama' in args.model:
                if dataset_name in ['NQ', 'TriviaQA', 'hotpotQA']: args.model = 'llama2-13b-chat'
                else: args.model = 'codellama-13b-instruct'
            cmd = (f'python generator/generate.py --action {args.action} --model {args.model} --temperature {args.temperature} '
                   f'--dataset {dataset_name} --retriever {args.retriever} --analysis_type {args.analysis_type} --n {args.n} --prompt_type {args.prompt_type}')
            cmds.append(cmd)
        batch_cmd = ''
        for cmd in cmds:
            batch_cmd = batch_cmd + cmd + ' ; '
    if args.action == 'eval_pred':
        for cmd in cmds:
            proc = subprocess.Popen(cmd.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (output, error) = proc.communicate()
            output_lines = output.decode().split('\n')
            if output_lines[-1] == '':
                print(cmd, '\n', output_lines[-2])
            else:
                print(cmd, '\n', output_lines[-1])
    else:
        subprocess.check_output(batch_cmd, shell=True)
        print(f'done {args.action} for prompt method analysis, {args.model} {args.prompt_type}')
