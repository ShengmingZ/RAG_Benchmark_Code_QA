import os
import re
import json
import ast
import random
from io import StringIO
import sys, platform
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from dataset_utils.dataset_configs import DS1000Loader, PandasNumpyEvalLoader
from generator.run_model import chatgpt
from data.DS1000.ds1000 import DS1000Dataset

random.seed(0)

common_func_list = ['copy', 'format', 'nan']
# common_func_list = ['sum', 'replace', 'lower', 'search', 'sort', 'pow', 'append', 'sub', 'sqrt', 'split', 'add', 'keys', 'items', 'join', 'extend', 'copy', 'remove', 'index', 'pop']



def extract_func_name(code_string):
    """
    given a code string, extract all function names by parse AST
    """
    # clean code
    # if 'BEGIN SOLUTION' in code_string: code_string = code_string.split('BEGIN SOLUTION')[1].split('END SOLUTION')[0].replace('\n    ','\n')
    # if code_string.startswith('    '): code_string = code_string[4:].replace('\n    ', '\n')
    # if code_string.startswith('\n    '): code_string = code_string[5:].replace('\n    ', '\n')
    # if code_string.startswith('\n   '): code_string = code_string[4:].replace('\n    ', '\n')
    # if code_string.startswith(' \n'): code_string = code_string[1:].replace('\n    ', '\n')
    # if code_string.startswith('# def '): code_string = code_string.split('\n', 1)[1].replace('    ', '')

    tree = ast.parse(code_string)

    # add parent
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node
    # functions
    func_names = [
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and hasattr(node, 'parent') and isinstance(node.parent, ast.Call)
    ]
    _func_names = list()
    for func_name in func_names:
        if re.search(f'{func_name}\\(', code_string):
            _func_names.append(func_name)
    # callable attributions
    attr_names = [
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    ]
    _attr_names = list()
    for attr_name in attr_names:
        if re.search(f'{attr_name}\\(', code_string) or re.search(f'{attr_name}\\[', code_string):
            _attr_names.append(attr_name)

    names = list(set(_func_names + _attr_names))
    _names = list()
    for name in names:
        if name not in common_func_list: _names.append(name)
    return _names


def remove_unbalanced_and_special_characters(text):
    """
    remove outer function (e.g. func1(func2())), and split special characters (e.g. x+x.mean())
    :param text:
    :return:
    """
    # detect and remove the content before unbalanced parenthesis
    stack = []
    unbalanced_indexes = []
    for i, char in enumerate(text):
        if char == '(' or char == '[':
            stack.append((char, i))
        elif char == ')' or char == ']':
            if not stack:
                unbalanced_indexes.append(i)
            else:
                opening, _ = stack.pop()
                if (opening == '[' and char != ']') or (opening == '(' and char != ')'):
                    unbalanced_indexes.append(i)
    for _, index in stack:
        unbalanced_indexes.append(index)
    if unbalanced_indexes:
        text = text[max(unbalanced_indexes) + 1:]

    # detect special chars and remove the content before them
    special_char = ['+', '-', '*', '/', '<', '>', ';', '?', '%', '^', '=']
    stack = []
    special_indexes = []
    for idx, char in enumerate(text):
        if char == '(' or char == '[':
            stack.append((char, idx))
        elif char in special_char and len(stack) == 0:    # not in parentheses
            special_indexes.append(idx)
        elif char == ')' or char == ']':
            stack.pop()
    if special_indexes:
        text = text[max(special_indexes) + 1:]

    return text


def augment_with_prefix(gold_output, gold_func_list):
    """
    covert func to the format of help(), e.g. reset_index -> df.reset_index
    :param gold_output:
    :param gold_func_list:
    :return:
    """
    # covert func name to format of help()
    func_list_full_name = list()
    gold_output = ' ' + gold_output
    for gold_func in gold_func_list:
        if gold_func not in gold_output: continue  # filter out funcs not in gold output
        # todo: ignore special char and space inside (), [] and {}
        pattern_func = rf"\s{gold_func}"    # for single function
        close_parentheses_pattern = r"\([^()]*\)|\[[^\[\]]*\]|\S"
        pattern_attr = rf"\s(?:{close_parentheses_pattern})+\.{gold_func}"  # for attribution, ignore \s in close parenthesis
        try:
            potential_full_names = re.findall(pattern_func, gold_output) + re.findall(pattern_attr, gold_output)
            for full_name in potential_full_names:
                full_name = remove_unbalanced_and_special_characters(full_name.replace(' ', '').replace('\n', ''))
                func_list_full_name.append(full_name)
        except:
            pass
    func_list_full_name = list(set(func_list_full_name))
    return func_list_full_name


def augment_program_pandas_numpy_eval(data):
    gold_output = data['canonical_solution'][0]
    prompt = data['prompt']
    entry_point = data['entry_point']

    # extract func names
    gold_func_list = extract_func_name(prompt + gold_output)

    # covert to the format of help()
    # todo: this method is not sound
    func_list_full_name = augment_with_prefix(gold_output, gold_func_list)

    program = prompt + gold_output
    lines = program.split('\n')
    last_comment_idx = max((i for i, line in enumerate(lines) if "#" in line), default=-1)
    return_idx = max((i for i, line in enumerate(lines) if "return " in line and "#" not in line), default=-1)
    # gene help command
    indent = get_indent(lines[last_comment_idx])
    help_strings = list()
    for full_name in func_list_full_name:
        help_strings.append('\n' + indent + f'try: help({full_name})\n' + indent + 'except: pass\n')
    # combine help and program
    programs = list()
    for help_string in help_strings:
        if return_idx == -1:
            programs.append(program + help_string)
        else:
            prompt_before_return = "\n".join(lines[:return_idx])
            prompt_after_return = "\n".join(lines[return_idx:])
            programs.append(prompt_before_return + help_string + prompt_after_return)
    # process and add test code
    split_string = data['test'].split('assert')
    if len(split_string) > 1:
        test_code = split_string[0] + 'assert' + split_string[1] + '\n'
        test_code += 'check()' if entry_point == 'none' else f'check({entry_point})'
    else:
        test_code = data['test']
    for idx, program in enumerate(programs):
        programs[idx] = program + '\n' + test_code

    return programs, func_list_full_name


def get_indent(line):
    """
    get indent of a code statement
    :return:
    """
    indent = 0
    for char in line:
        if char == " ":
            indent += 1
        else:
            break
    indent = indent * ' '
    return indent


def postprocess(lib, generated_code: str):
    if lib == "Matplotlib":
        code_lines = generated_code.split("\n")
        postprocessed_lines = []
        for line in code_lines:
            skip_line_flag = False
            # Matplotlib removes function calls that will clear the global figure object
            # Removing these functions helps running execution-based evaluation
            for phrase in ["plt.show()", "plt.clf()", "plt.close()", "savefig"]:
                if phrase in line:
                    skip_line_flag = True
                    break

            if skip_line_flag:
                continue
            else:
                postprocessed_lines.append(line)
        generated_code = "\n".join(postprocessed_lines)
    return generated_code



def augment_program_ds1000(data):
    # ds1000 = DS1000Dataset(source_dir='../data/DS1000/ds1000_data', libs='all', mode='Completion')
    gold_output = data['reference_code']
    code_snippet = data['code_context']
    lib, problem_id = data['qs_id'].split('_')

    # extract func names
    gold_func_list = extract_func_name(code_snippet.replace("[insert]", gold_output))

    # covert to the format of help()
    # todo: this func is not robust
    func_list_full_name = augment_with_prefix(gold_output, gold_func_list)
    # gene augmented gold_output
    output_lines = gold_output.split('\n')
    aug_gold_outputs = list()
    current_gold_output = ''
    aug_gold_outputs = ['' for _ in range(len(func_list_full_name))]    # keep the index of aug_gold_outputs and full_name_list the same
    for line in output_lines:
        line_indent = get_indent(line)
        for idx, full_name in enumerate(func_list_full_name):
            if full_name in line:
                aug_gold_outputs[idx] = current_gold_output + line_indent + f'try: help({full_name})\n' + line_indent + 'except: pass\n'
        current_gold_output = current_gold_output + line + '\n'
        for idx, aug_gold_output in enumerate(aug_gold_outputs):
            aug_gold_outputs[idx] = aug_gold_outputs[idx] + line + '\n'


    # augment the code snippet
    # todo: analysis execution process of ds1000
    # lines = code_snippet.split('\n')
    # insert_idx = [i for i, line in enumerate(lines) if "[insert]" in line]
    # assert len(insert_idx) == 1
    # insert_indent = get_indent(lines[insert_idx[0]])
    # # augment wth [insert_help]
    # aug_prompt = ''
    # for idx, line in enumerate(lines):
    #     aug_prompt = aug_prompt + line + '\n'
    #     if idx == insert_idx:
    #         aug_prompt = aug_prompt + insert_indent + '[insert_help]' + '\n'


    # # remove `with open`
    # lines = code_snippet.split('\n')
    # if lib == 'Matplotlib':
    #     with_open_idx = [i for i, line in enumerate(lines) if "plt.savefig" in line]
    # else:
    #     with_open_idx = [i for i, line in enumerate(lines) if "with open" in line]
    #     if data['qs_id'] in ['Sklearn_35', 'Sklearn_36', 'Sklearn_113']:
    #         ...
    # new_prompt = ''

    programs = list()
    for aug_gold_output in aug_gold_outputs:
        program = code_snippet.replace('[insert]', aug_gold_output)
        programs.append(postprocess(lib, program))

    return programs, func_list_full_name



def augment_program_conala(data):
    gold_output = data['canonical_solution']
    code_snippet = data['prompt']

    program = f"{code_snippet}{gold_output}"
    test_func = f"\n{data['test']}\ncheck({data['entry_point']})"



def match_oracle_doc(data, dataset):
    """
    Given executable code, match API usage utilizing help()
    :param gold_output:
    :param prompt:
    :param api_sign_collection:
    :return:
    """
    # for pandas_numpy_eval
    if dataset == 'PandasNumpyEval':
        programs, func_list_full_name = augment_program_pandas_numpy_eval(data)

    elif dataset == 'DS1000':
        programs, func_list_full_name = augment_program_ds1000(data)

    elif dataset == 'CoNaLa':
        programs, func_list_full_name = augment_program_conala(data)

    # exec and get output
    oracle_docs = list()
    for idx, program in enumerate(programs):
        if dataset == 'PandasNumpyEval':
            printed_output = pandas_numpy_eval_exec(program)
        elif dataset == 'DS1000':
            lib, problem_id = data['qs_id'].split('_')
            problem_path = os.path.join(root_path, 'data/DS1000/ds1000_data', f'{lib}/Completion/q{problem_id}')
            printed_output = ds1000_exec(program, problem_path)
        try:
            if 'Help on' in printed_output:
                api_sign, content = printed_output.split('\n\n', 1)
                # get method
                method = func_list_full_name[idx].split('.')[-1].split(' ')[-1]
                if method in api_sign:
                    # get module
                    if 'built-in' in api_sign:
                        module = 'builtins'
                    else:
                        module = api_sign.split('module ')[1].replace(':', '')
                    api_sign = module + '.' + method
                    oracle_doc = dict(api_sign=api_sign, content=content)
                    oracle_docs.append(oracle_doc['api_sign'])
                else:
                    print(method, api_sign)
        except:
            ...
    return oracle_docs



def ds1000_exec(program, problem_path='../data/DS1000/ds1000_data'):
    """
    an augmented version of DS1000.test
    :return:
    """
    import shutil

    cwd = os.getcwd()
    os.chdir(problem_path)

    # generated outputs will be put into `result`
    result_path = os.path.join(problem_path, "result")
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.mkdir(result_path)

    # exec
    old_stdout = sys.stdout
    new_stdout = StringIO()
    sys.stdout = new_stdout
    try:
        exec(program, {})
        printed_output = new_stdout.getvalue()
    except:
        printed_output = None
    finally:
        sys.stdout = old_stdout
        os.chdir(cwd)

    return printed_output


def pandas_numpy_eval_exec(program):
    old_stdout = sys.stdout
    new_stdout = StringIO()
    sys.stdout = new_stdout
    try:
        exec(program, {})
        printed_output = new_stdout.getvalue()
    except:
        printed_output = None
    finally:
        sys.stdout = old_stdout

    return printed_output



# def match_oracle_doc_old(gold_lib, gold_output, api_sign_collection):
#     """
#     match API usage of gold output
#     :param gold_lib:
#     :param gold_output:
#     :param api_sign_collection: a collection of API signatures
#     :return:
#     """
#     gold_func_list = extract_func_name(gold_output)
#     # for func in common_func_list:
#     #     if func in gold_func_list: gold_func_list.remove(func)
#     gold_func_length = len(gold_func_list)
#     if gold_func_length == 0: return [], 1
#     gold_lib = gold_lib.lower()
#     if gold_lib == 'pytorch': gold_lib = 'torch'
#     for idx in range(len(gold_func_list)): gold_func_list[idx] = gold_func_list[idx].lower()
#
#     matched_api_list = []
#     # match golden lib
#     for api_sign in api_sign_collection[gold_lib]:
#         if api_sign.startswith('tensorflow.compat.v1.'): continue
#         func = api_sign.rsplit('.', 1)[-1].lower()
#         if func in gold_func_list:
#             matched_api_list.append(api_sign)
#             gold_func_list.remove(func)
#     # match other libs
#     for lib in api_sign_collection:
#         if lib == gold_lib: continue
#         for api_sign in api_sign_collection[lib]:
#             func = api_sign.rsplit('.', 1)[-1].lower()
#             if func in gold_func_list:
#                 matched_api_list.append(api_sign)
#                 gold_func_list.remove(func)
#     matched_rate = 1 - len(gold_func_list) / gold_func_length
#
#     # if matched_rate < 1.0:
#     #     print(gold_lib)
#     #     print([oracle['output']])
#     #     print(gold_func_list)
#
#     return matched_api_list, matched_rate


def match_docs(dataset_name):
    # load API signatures
    # api_signs_file = os.path.join(root_path, "data/conala/python_manual_firstpara.tok.id")
    # with open(api_signs_file, 'r') as f:
    #     api_signs = list()
    #     for line in f:
    #         api_signs.append(line.strip())
    # lib_list = [api_sign.split('.')[0] for api_sign in api_signs]
    # # lib_list = list(set(lib_list)): ['tensorflow', 'matplotlib', 'sklearn', 'matplotlib_configuration_api#matplotlib', 'django', 'numpy', 'pygame', 'matplotlib_configuration_api', 'torch', 'python', 'flask', 'django_rest_framework', 'skimage', 'pandas', 'werkzeug']
    # func_list = [api_sign.split('.')[-1] for api_sign in api_signs]
    # api_sign_collection = dict()
    # for idx, (lib, api_sign) in enumerate(zip(lib_list, api_signs)):
    #     if lib not in api_sign_collection.keys(): api_sign_collection[lib.lower()] = []
    #     api_sign_collection[lib.lower()].append(api_sign)

    # match oracle list
    assert dataset_name in ['DS1000', 'PandasNumpyEval']

    # get dataset
    if dataset_name == 'PandasNumpyEval':
        pandas_eval_file = '/Users/zhaoshengming/Code_RAG_Benchmark/data/pandas-numpy-eval/data/PandasEval.jsonl.gz'
        numpy_eval_file = pandas_eval_file.replace('PandasEval', 'NumpyEval')
        import gzip
        pandas_eval_data = list()
        with gzip.open(pandas_eval_file, 'rt') as f:
            for line in f:
                pandas_eval_data.append(json.loads(line))
        numpy_eval_data = list()
        with gzip.open(numpy_eval_file, 'rt') as f:
            for line in f:
                numpy_eval_data.append(json.loads(line))
        dataset = pandas_eval_data + numpy_eval_data
    elif dataset_name == 'DS1000':
        data_file = '../data/DS1000/sampled_data.json'
        dataset = json.load(open(data_file, 'r'))
        # ds1000 = DS1000Dataset(source_dir='../data/DS1000/ds1000_data', libs='all', mode='Completion')
        # dataset = []
        # for lib in ds1000.libs:
        #     for idx, data in enumerate(ds1000[lib]):
        #         data['qs_id'] = lib + '_' + str(idx)
        #         dataset.append(data)
    elif dataset_name == 'CoNaLa':
        data_file = '../data/conala/unittest_docprompting_conala.json'
        dataset = list(json.load(open(data_file, 'r')).values())
    else:
        raise Exception('Unknown dataset')

    # match oracle docs
    oracle_list = []
    for idx, data in enumerate(dataset):
        if dataset_name == 'DS1000':
            qs_id = data['qs_id']
            output = data['reference_code']
        elif dataset_name == 'PandasNumpyEval':
            qs_id = data['task_id']
            output = data['canonical_solution'][0]
            if idx == 19: data['canonical_solution'][0] = data['canonical_solution'][0].replace('    ', '', 1)
        else:
            qs_id = data['task_id']
            output = data['canonical_solution']
        oracle_docs = match_oracle_doc(data=data, dataset=dataset_name)
        oracle_list.append(dict(qs_id=qs_id, oracle_docs=oracle_docs, output=output))

    if dataset_name == 'DS1000':
        save_file = os.path.join(root_path, 'data/DS1000/oracle_docs_matched.json')
    elif dataset_name == 'PandasNumpyEval':
        save_file = os.path.join(root_path, 'data/pandas-numpy-eval/data/oracle_docs_matched.json')
    with open(save_file, 'w+') as f:
        json.dump(oracle_list, f, indent=2)



if __name__ == '__main__':

    match_docs(dataset_name='DS1000')


    # ds1000_loader = DS1000Loader()
    # oracle_list = ds1000_loader.load_oracle_list()
    # dataset = 'DS1000'

    # pandas_numpy_eval_loader = PandasNumpyEvalLoader()
    # oracle_list = pandas_numpy_eval_loader.load_oracle_list()
    # qs_list = pandas_numpy_eval_loader.load_qs_list()
    # dataset = 'PandasNumpyEval'
    # match_docs(dataset=dataset, qs_list=qs_list, oracle_list=oracle_list)

    # import pandas as pd
    #
    # data = {
    #     'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    #     'Age': [25, 30, 35, 40],
    #     'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
    # }
    # df = pd.DataFrame(data)
    # df.dropna()
    # col_name = 'Name'

    # import pandas as pd
    # def f(x):
    #     a = x['Value'].iat[2] - x['Value'].iat[1]
    #     b = x['Value'].iat[3] - x['Value'].iat[0]
    #     c = x['ID'].iat[2] + ' - ' + x['ID'].iat[1]
    #     d = x['ID'].iat[3] + ' - ' + x['ID'].iat[0]
    #     return pd.DataFrame({'Value': [a, b], 'ID': [c, d]})
    # def calculate_row_diff_groupwise(df):
    #     # I need to calculate the difference between two rows groupwise using pandas.
    #     # To calculate the sum I would use pandas.groupby('Group').sum(), but how do you calculate the difference between rows where the row ordering is important?
    #     # I think we need custom function with apply which return DataFrame for each group, for select by position is used iat:
    #     # Return the result
    #     # print(help(df.groupby))
    #     print(help(df.groupby('Group').apply))
    #     # print(help(df.groupby('Group').apply(f).reset_index))
    #     # print(help(df.groupby('Group').apply(f).reset_index(level=1, drop=True).reset_index))
    #     return df.groupby('Group').apply(f).reset_index(level=1, drop=True).reset_index()
    #
    # METADATA = {
    #     'author': 'msra-v-dazan',
    #     'dataset': 'test',
    #     'type': 'groupby_apply_reset_index'
    # }
    # def check(candidate):
    #     assert candidate(pd.DataFrame(
    #         {'Group': ['M1', 'M1', 'M1', 'M1'], 'Value': [3, 3, 5, 4], 'ID': ['dki', 'two', 'three', 'msra']})).equals(
    #         pd.DataFrame({'Group': ['M1', 'M1'], 'Value': [2, 1], 'ID': ['three - two', 'msra - dki']}))
    #
    # check(calculate_row_diff_groupwise)

    # import scipy
    # module_attributes = dir(scipy)
    # print(len(module_attributes))

    # pandas_eval_file = '/Users/zhaoshengming/Code_RAG_Benchmark/data/pandas-numpy-eval/data/PandasEval.jsonl.gz'
    # numpy_eval_file = pandas_eval_file.replace('PandasEval', 'NumpyEval')
    #
    # import gzip
    # import json
    #
    # pandas_eval_data = list()
    # with gzip.open(pandas_eval_file, 'rt') as f:
    #     for line in f:
    #         pandas_eval_data.append(json.loads(line))
    #
    # numpy_eval_data = list()
    # with gzip.open(numpy_eval_file, 'rt') as f:
    #     for line in f:
    #         numpy_eval_data.append(json.loads(line))
    #
    # # data = pandas_eval_data[90]
    # # prompt = data["prompt"]
    # # gold_output = data['canonical_solution'][0]
    # # print(prompt + gold_output)
    # # names = extract_func_name(prompt + gold_output)
    # # # re.search("idx\\(", prompt+gold_output)
    # # print(names)
    #
    # # for data in pandas_eval_data[50:55]:
    # #     print(data['canonical_solution'][0])
    # #     print(extract_func_name(data['prompt']+data['canonical_solution'][0]))
    # #     match_oracle_doc(data, 'pandas_numpy_eval', None)
    # #     print('\n\n')
    #
    # dataset = pandas_eval_data + numpy_eval_data
    # match_docs(dataset=dataset, dataset_name='PandasNumpyEval')
