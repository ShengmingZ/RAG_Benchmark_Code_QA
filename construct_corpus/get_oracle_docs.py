import os
import re
import json
import ast
import random
import string
from io import StringIO
import sys, platform
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from dataset_utils.dataset_configs import DS1000Loader, PandasNumpyEvalLoader
from data.DS1000.ds1000 import DS1000Dataset

random.seed(0)

common_func_list = ['copy', 'format', 'lower', 'len']
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

    try:
        tree = ast.parse(code_string)
    except:
        print(f'invalid code string {code_string}')
        return []

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


# def remove_unbalanced_and_special_characters(text):
#     """
#     remove outer function (e.g. func1(func2())), and split special characters (e.g. x+x.mean())
#     :param text:
#     :return:
#     """
#     # detect and remove the content before unbalanced parenthesis
#     stack = []
#     unbalanced_indexes = []
#     for i, char in enumerate(text):
#         if char == '(' or char == '[':
#             stack.append((char, i))
#         elif char == ')' or char == ']':
#             if not stack:
#                 unbalanced_indexes.append(i)
#             else:
#                 opening, _ = stack.pop()
#                 if (opening == '[' and char != ']') or (opening == '(' and char != ')'):
#                     unbalanced_indexes.append(i)
#     for _, index in stack:
#         unbalanced_indexes.append(index)
#     if unbalanced_indexes:
#         text = text[max(unbalanced_indexes) + 1:]
#
#     # detect special chars and remove the content before them
#     special_char = ['+', '-', '*', '/', '<', '>', ';', '?', '%', '^', '=']
#     stack = []
#     special_indexes = []
#     for idx, char in enumerate(text):
#         if char == '(' or char == '[':
#             stack.append((char, idx))
#         elif char in special_char and len(stack) == 0:    # not in parentheses
#             special_indexes.append(idx)
#         elif char == ')' or char == ']':
#             stack.pop()
#     if special_indexes:
#         text = text[max(special_indexes) + 1:]
#
#     return text


def check_balanced_parentheses(s):
    stack = []
    opening = {'(': ')', '[': ']', '{': '}'}
    closing = {')', ']', '}'}
    for char in s:
        if char in opening:
            stack.append(char)
        elif char in closing:
            if not stack:
                # There's a closing bracket or parenthesis with no matching opening one
                return False
            last_open = stack.pop()
            if opening[last_open] != char:
                # The closing character does not match the last opening character
                return False
    # If the stack is not empty, there are unmatched opening parentheses or brackets
    return len(stack) == 0


def augment_with_prefix(output_code, func_name_list):
    """
    covert func to the format of help(), e.g. reset_index -> df.reset_index
    :param code_string:
    :param func_name_list:
    :return:
    """
    # covert func name to format of help()
    # for func in func_name_list:
    #     if func not in code_string: continue  # filter out funcs not in gold output
    #     pattern_func = rf"\s{func}"    # for single function
    #     close_parentheses_pattern = r"\([^()]*\)|\[[^\[\]]*\]|\S"
    #     pattern_attr = rf"\s(?:{close_parentheses_pattern})+\.{gold_func}"  # for attribution, ignore \s in close parenthesis
    #     try:
    #         potential_full_names = re.findall(pattern_func, code_string) + re.findall(pattern_attr, code_string)
    #         for full_name in potential_full_names:
    #             full_name = remove_unbalanced_and_special_characters(full_name.replace(' ', '').replace('\n', ''))
    #             func_full_name_list.append(full_name)
    #     except:
    #         pass

    output_code = ' ' + output_code
    # greedy match as more potential substrings as possible
    special_chars = list(set(string.printable) - set(string.digits) - set(string.ascii_letters))
    potential_func_full_name_list = list()
    for func_name in func_name_list:
        if func_name not in output_code: continue
        pattern = rf'[^a-zA-Z0-9\.].*{func_name}'
        for idx, char in enumerate(output_code):
            if char in special_chars:
                sub_output_code = output_code[idx:]
                for full_name in re.findall(pattern, sub_output_code):
                    # do some filters to reduce the size of potential substrings
                    full_name = full_name[1:].lstrip()
                    if check_balanced_parentheses(full_name):
                        potential_func_full_name_list.append(full_name)

    func_full_name_list = list(set(potential_func_full_name_list))
    return func_full_name_list


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


def augment_program_pandas_numpy_eval(data):
    gold_output = data['canonical_solution'][0]
    prompt = data['prompt']
    entry_point = data['entry_point']

    # extract func names
    gold_func_list = extract_func_name(prompt + gold_output)

    # covert to the format of help()
    func_list_full_name = augment_with_prefix(output_code=gold_output, func_name_list=gold_func_list)

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

    programs = list()
    for aug_gold_output in aug_gold_outputs:
        program = code_snippet.replace('[insert]', aug_gold_output)
        programs.append(postprocess(lib, program))

    return programs, func_list_full_name



def augment_program_conala(data):
    gold_output = data['canonical_solution']
    code_snippet = data['prompt']
    program = f"{code_snippet}{gold_output}{data['suffix']}"

    def split_by_second_assert(s):
        delimiter = "assert"
        first_index = s.find(delimiter)
        if first_index == -1:
            return s  # Delimiter not found at all, return the original string
        second_index = s.find(delimiter, first_index + len(delimiter))
        if second_index == -1:
            return s  # Only one occurrence found, return the original string
        return s[:second_index]
    data['test'] = split_by_second_assert(data['test'])
    test_func = f"\n{data['test']}\ncheck({data['entry_point']})"

    # extract func names
    gold_func_list = extract_func_name(program)

    # covert to the format of help()
    func_list_full_name = augment_with_prefix(gold_output, gold_func_list)

    program1, program2 = program.split('\n', 1)
    programs = list()
    for full_name in func_list_full_name:
        help_string = '\n\t' + f'try: help({full_name})\n' + '\t' + 'except: pass\n'
        programs.append(program1 + help_string + program2 + test_func)

    return programs, func_list_full_name



def match_oracle_doc(data, dataset):
    """
    match oracle docs for an instance
    :param data:
    :param dataset:
    :return:
    """
    # augment the data with help()
    if dataset == 'pandas-numpy-eval':
        programs, func_full_name_list = augment_program_pandas_numpy_eval(data)
    elif dataset == 'DS1000':
        programs, func_full_name_list = augment_program_ds1000(data)
    else:
        programs, func_full_name_list = augment_program_conala(data)

    # exec the augmented code and get output docs
    oracle_docs = list()
    _func_full_name_list = list()
    for idx, program in enumerate(programs):
        if dataset == 'pandas-numpy-eval':
            printed_output = pandas_numpy_eval_exec(program)
        elif dataset == 'DS1000':
            lib, problem_id = data['qs_id'].split('_')
            problem_path = os.path.join(root_path, 'data/DS1000/ds1000_data', f'{lib}/Completion/q{problem_id}')
            printed_output = ds1000_exec(program, problem_path)
        else:
            printed_output = pandas_numpy_eval_exec(program)
        if printed_output:
            oracle_docs.append(printed_output)
            _func_full_name_list.append(func_full_name_list[idx])
        # try:
        #     if 'Help on' in printed_output:
        #         api_sign, content = printed_output.split('\n\n', 1)
        #         # get method
        #         method = func_list_full_name[idx].split('.')[-1].split(' ')[-1]
        #         if method in api_sign:
        #             # get module
        #             if 'built-in' in api_sign:
        #                 module = 'builtins'
        #             else:
        #                 module = api_sign.split('module ')[1].replace(':', '')
        #             api_sign = module + '.' + method
        #             oracle_doc = dict(api_sign=api_sign, content=content)
        #             oracle_docs.append(oracle_doc['api_sign'])
        #         else:
        #             print(method, api_sign)
        # except:
        #     ...
    return oracle_docs, _func_full_name_list



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


def main(dataset_name):
    # match oracle list
    assert dataset_name in ['DS1000', 'pandas-numpy-eval', 'conala']

    # load dataset
    if dataset_name == 'pandas-numpy-eval':
        pandas_eval_file = os.path.join(root_path, 'data/pandas-numpy-eval/data/PandasEval.jsonl.gz')
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
        data_file = os.path.join(root_path, 'data/DS1000/sampled_data.json')
        dataset = json.load(open(data_file, 'r'))
        # ds1000 = DS1000Dataset(source_dir='../data/DS1000/ds1000_data', libs='all', mode='Completion')
        # dataset = []
        # for lib in ds1000.libs:
        #     for idx, data in enumerate(ds1000[lib]):
        #         data['qs_id'] = lib + '_' + str(idx)
        #         dataset.append(data)
    elif dataset_name == 'conala':
        data_file = os.path.join(root_path, 'data/conala/unittest_docprompting_conala.json')
        dataset = list(json.load(open(data_file, 'r')).values())
    else:
        raise Exception('Unknown dataset')

    # match oracle docs
    oracle_list = []
    for idx, data in enumerate(dataset):
        if dataset_name == 'DS1000':
            qs_id = data['qs_id']
            output = data['reference_code']
        elif dataset_name == 'pandas-numpy-eval':
            qs_id = data['task_id']
            output = data['canonical_solution'][0]
            if idx == 19: data['canonical_solution'][0] = data['canonical_solution'][0].replace('    ', '', 1)
        else:
            qs_id = data['task_id']
            output = data['canonical_solution']
        oracle_docs, full_name_list = match_oracle_doc(data=data, dataset=dataset_name)
        oracle_list.append(dict(qs_id=qs_id, oracle_docs=oracle_docs, output=output, full_name_list=full_name_list))

    if dataset_name == 'DS1000':
        save_file = os.path.join(root_path, 'data/DS1000/oracle_docs_matched.json')
    elif dataset_name == 'pandas-numpy-eval':
        save_file = os.path.join(root_path, 'data/pandas-numpy-eval/data/oracle_docs_matched.json')
    elif dataset_name == 'conala':
        save_file = os.path.join(root_path, 'data/conala/oracle_docs_matched.json')
    with open(save_file, 'w+') as f:
        json.dump(oracle_list, f, indent=2)


# def match_api_sign_with_doc(dataset_name):
#     if dataset_name == 'DS1000':
#         save_file = os.path.join(root_path, 'data/DS1000/oracle_docs_matched_new.json')
#     elif dataset_name == 'PandasNumpyEval':
#         save_file = os.path.join(root_path, 'data/pandas-numpy-eval/data/oracle_docs_matched_new.json')
#     elif dataset_name == 'CoNaLa':
#         save_file = os.path.join(root_path, 'data/conala/oracle_docs_matched_new.json')
#     oracle_list = json.load(open(save_file, 'r'))
#
#     python_doc_id_third, python_doc_id_builtins = [], []
#     with open('../data/python_docs/api_sign_third_party.txt', 'r') as f:
#         for line in f:
#             python_doc_id_third.append(line.strip())
#     with open('../data/python_docs/api_sign_builtin.txt', 'r') as f:
#         for line in f:
#             python_doc_id_builtins.append(line.strip())
#     python_doc_id_list = python_doc_id_third + python_doc_id_builtins
#
#     count = 0
#     for oracle in oracle_list:
#         for oracle_doc in oracle['oracle_docs']:
#             if oracle_doc not in python_doc_id_list:
#                 print(oracle['qs_id'], oracle_doc)
#                 if not oracle_doc.startswith('builtins'): count += 1
#
#     print(count)


if __name__ == '__main__':
    # data_file = os.path.join(root_path, 'data/DS1000/sampled_data.json')
    # dataset = json.load(open(data_file, 'r'))
    # data = dataset[30]
    #
    # gold_output = data['reference_code']
    # code_snippet = data['code_context']
    #
    # # extract func names
    # gold_func_list = extract_func_name(code_snippet.replace("[insert]", gold_output))
    # print(gold_output)
    # _gold_func_list = list()
    # for func in gold_func_list:
    #     if func in gold_output:
    #         _gold_func_list.append(func)
    # gold_func_list = _gold_func_list
    # print(gold_func_list)
    #
    #
    # output_code = ' ' + gold_output
    # func_name_list = gold_func_list
    # special_chars = list(set(string.printable) - set(string.digits) - set(string.ascii_letters))
    # potential_func_full_name_list = list()
    # for idx, char in enumerate(output_code):
    #     if char in special_chars:
    #         sub_output_code = output_code[idx:]
    #         for func_name in func_name_list:
    #             pattern = rf'[^a-zA-Z0-9].*{func_name}'
    #             # pattern = rf'\s.*{func_name}'
    #
    #             def check_balanced_parentheses(s):
    #                 stack = []
    #                 opening = {'(': ')', '[': ']', '{': '}'}
    #                 closing = {')', ']', '}'}
    #                 for char in s:
    #                     if char in opening:
    #                         stack.append(char)
    #                     elif char in closing:
    #                         if not stack:
    #                             # There's a closing bracket or parenthesis with no matching opening one
    #                             return False
    #                         last_open = stack.pop()
    #                         if opening[last_open] != char:
    #                             # The closing character does not match the last opening character
    #                             return False
    #                 # If the stack is not empty, there are unmatched opening parentheses or brackets
    #                 return len(stack) == 0
    #
    #             # print(re.findall(pattern, sub_output_code))
    #             for full_name in re.findall(pattern, sub_output_code):
    #                 full_name = full_name[1:].lstrip()
    #                 if check_balanced_parentheses(full_name):
    #                     potential_func_full_name_list.append(full_name)
    # print(list(set(potential_func_full_name_list)))


    main('conala')
    # main('DS1000')
    # main('pandas-numpy-eval')

    # import os
    # os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    #
    # import evaluate
    # code_eval_metric = evaluate.load("code_eval")
    #
    # data_file = '../data/conala/unittest_docprompting_conala.json'
    # dataset = list(json.load(open(data_file, 'r')).values())
    #
    # preds = []
    # tests = []
    # for data in dataset:
    #     gold_output = data['canonical_solution']
    #     code_snippet = data['prompt']
    #
    #     program = f"{code_snippet}{gold_output}{data['suffix']}"
    #     test_func = f"\n{data['test']}\ncheck({data['entry_point']})"
    #
    #     preds.append([program])
    #     tests.append(test_func)
    #
    # # pass_list = []
    # # for idx, (pred, test) in enumerate(zip(preds, tests)):
    # #     r = code_eval_metric.compute(
    # #         predictions=[pred],
    # #         references=[test],
    # #         k=[1],
    # #         num_workers=1,
    # #     )
    # #     if r[0]['pass@1'] != 1:
    # #         pass_list.append(idx)
    # #
    # # for idx in pass_list:
    # #     print(idx, preds[idx])
    #
    # r = code_eval_metric.compute(
    #     predictions=preds,
    #     references=tests,
    #     k=[1],
    #     num_workers=1,
    # )
    # print(r[0])





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
