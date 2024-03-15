import os
import re
import json
import ast
import random
import sys, platform
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from dataset_configs import DS1000Loader, HumanEvalLoader
from generator.run_model import chatgpt

random.seed(0)

common_func_list = ['copy', 'replace', 'append', 'keys', 'shape', 'map']

# todo: lib name duplicate, can only be analyzed dynamically
def extract_func_name(code_string):
    # clean code
    if 'BEGIN SOLUTION' in code_string: code_string = code_string.split('BEGIN SOLUTION')[1].split('END SOLUTION')[0].replace('\n    ','\n')
    if code_string.startswith('    '): code_string = code_string[4:].replace('\n    ', '\n')
    if code_string.startswith('\n    '): code_string = code_string[5:].replace('\n    ', '\n')
    if code_string.startswith('\n   '): code_string = code_string[4:].replace('\n    ', '\n')
    if code_string.startswith(' \n'): code_string = code_string[1:].replace('\n    ', '\n')
    if code_string.startswith('# def '): code_string = code_string.split('\n', 1)[1].replace('    ', '')

    try:
        tree = ast.parse(code_string)
    except:
        print([code_string])
    # add parent
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node
    func_names = [
        node.id for node in ast.walk(ast.parse(code_string))
        if isinstance(node, ast.Name) and hasattr(node, 'parent') and isinstance(node.parent, ast.Call)
    ]
    attr_names = [
        node.attr for node in ast.walk(ast.parse(code_string))
        if isinstance(node, ast.Attribute)
    ]
    _attr_names = list()
    for attr_name in attr_names:
        if re.search(f'{attr_name}\\(', code_string) or re.search(f'{attr_name}\\[', code_string):
            _attr_names.append(attr_name)
    names = list(set(func_names + _attr_names))
    return names


def match_ds1000_doc(gold_lib, gold_output, api_sign_collection):
    gold_func_list = extract_func_name(gold_output)
    for func in common_func_list:
        if func in gold_func_list: gold_func_list.remove(func)
    gold_func_length = len(gold_func_list)
    if gold_func_length == 0: return [], 1
    gold_lib = gold_lib.lower()
    if gold_lib == 'pytorch': gold_lib = 'torch'
    for idx in range(len(gold_func_list)): gold_func_list[idx] = gold_func_list[idx].lower()

    matched_api_list = []
    # match golden lib
    for api_sign in api_sign_collection[gold_lib]:
        if api_sign.startswith('tensorflow.compat.v1.'): continue
        func = api_sign.rsplit('.', 1)[-1].lower()
        if func in gold_func_list:
            matched_api_list.append(api_sign)
            gold_func_list.remove(func)
    # match other libs
    for lib in api_sign_collection:
        if lib == gold_lib: continue
        for api_sign in api_sign_collection[lib]:
            func = api_sign.rsplit('.', 1)[-1].lower()
            if func in gold_func_list:
                matched_api_list.append(api_sign)
                gold_func_list.remove(func)
    matched_rate = 1 - len(gold_func_list) / gold_func_length

    # if matched_rate < 1.0:
    #     print(gold_lib)
    #     print([oracle['output']])
    #     print(gold_func_list)

    return matched_api_list, matched_rate


def match_ds1000():
    # load and process docs
    docs_file = os.path.join(root_path, "data/conala/conala_docs.json")
    docs = json.load(open(docs_file, 'r'))
    api_signs_file = os.path.join(root_path, "data/conala/python_manual_firstpara.tok.id")
    with open(api_signs_file, 'r') as f:
        api_signs = list()
        for line in f:
            api_signs.append(line.strip())
    lib_list = [api_sign.split('.')[0] for api_sign in api_signs]
    # lib_list = list(set(lib_list)): ['tensorflow', 'matplotlib', 'sklearn', 'matplotlib_configuration_api#matplotlib', 'django', 'numpy', 'pygame', 'matplotlib_configuration_api', 'torch', 'python', 'flask', 'django_rest_framework', 'skimage', 'pandas', 'werkzeug']
    func_list = [api_sign.split('.')[-1] for api_sign in api_signs]
    api_sign_collection = dict()
    for idx, (lib, api_sign) in enumerate(zip(lib_list, api_signs)):
        if lib not in api_sign_collection.keys(): api_sign_collection[lib.lower()] = []
        api_sign_collection[lib.lower()].append(api_sign)

    ds1000_loader = DS1000Loader()
    oracle_list = ds1000_loader.load_oracle_list()
    qs_list = ds1000_loader.load_qs_list()
    match_success_rate = 0
    sampled_num = 0
    result_list = []
    for (qs, oracle) in zip(qs_list, oracle_list):
        lib = oracle['qs_id'].split('_')[0]
        nl = qs['nl']
        if lib.lower() == 'scipy': continue
        matched_api_list, func_match_rate = match_ds1000_doc(lib, oracle['output'], api_sign_collection)
        result_list.append(dict(qs_id=oracle['qs_id'], oracle_docs=matched_api_list, output=oracle['output']))
        match_success_rate += func_match_rate
        sampled_num += 1

    with open(os.path.join(root_path, 'data/DS1000/oracle_docs_matched.json'), 'w+') as f:
        json.dump(result_list, f, indent=2)

    print('Match success rate: ', match_success_rate / sampled_num)


if __name__ == '__main__':
    match_ds1000()


