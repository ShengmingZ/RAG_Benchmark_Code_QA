import os
import re
import json
import ast, astor
import sys, platform
import matplotlib.pyplot as plt
import networkx as nx
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from dataset_configs import DS1000Loader


# clean variable using ast
# def clean_variables(gold_output):
#     output = "list(map(list, set(map(lambda i: tuple(i), testdata))))"
#     tree = ast.parse(output)
#     print(ast.dump(tree))
#
#     class VariableExtractor(ast.NodeVisitor):
#         def __init__(self):
#             self.variables = set()
#
#         def visit_Name(self, node):
#             self.variables.add(node.id)
#
#     # Create an instance of the VariableExtractor and visit the AST
#     variable_extractor = VariableExtractor()
#     variable_extractor.visit(tree)
#
#     # Get the unique variable names
#     variable_names = variable_extractor.variables
#     print(variable_names)
#
#     # Replace each variable name with a new value
#     new_code = output
#     for var_name in variable_names:
#         new_code = new_code.replace(var_name, 'new_value')
#     print(new_code)


# todo: 2 match, improve match rate 3 run method w/o docs 4 test, finish before 5pm
def match_ds1000_doc(gold_lib, gold_output):
    gold_lib = gold_lib.lower()
    gold_output = gold_output.lower()
    matched_api_list, matched_func_list = [], []
    # match with funcs in gold lib
    for (lib, func, api_sign) in zip(lib_list, func_list, api_signs):
        if lib != gold_lib:
            continue
        if re.search(f'{func}\\(', gold_output) or re.search(f'{func}\\[', gold_output):
            if func not in matched_func_list:
                matched_api_list.append(api_sign)
                matched_func_list.append(func)
    # match with funcs in other libs
    if len(matched_api_list) == 0:
        for (lib, func, api_sign) in zip(lib_list, func_list, api_signs):
            if re.search(f'{func}\\(', gold_output) or re.search(f'{func}\\[', gold_output):
                if func not in matched_func_list:
                    matched_api_list.append(api_sign)
                    matched_func_list.append(func)
    return matched_api_list


if __name__ == '__main__':
    # for python docs
    docs_file = os.path.join(root_path, "docprompting_data/conala/conala_docs.json")
    docs = json.load(open(docs_file, 'r'))
    api_signs_file = os.path.join(root_path, "docprompting_data/conala/python_manual_firstpara.tok.id")
    with open(api_signs_file, 'r') as f:
        api_signs = list()
        for line in f:
            api_signs.append(line.strip())
    lib_list = [api_sign.split('.')[0] for api_sign in api_signs]
    # lib_list = list(set(lib_list)): ['tensorflow', 'matplotlib', 'sklearn', 'matplotlib_configuration_api#matplotlib', 'django', 'numpy', 'pygame', 'matplotlib_configuration_api', 'torch', 'python', 'flask', 'django_rest_framework', 'skimage', 'pandas', 'werkzeug']
    func_list = [api_sign.split('.')[-1] for api_sign in api_signs]

    ds1000_loader = DS1000Loader()
    oracle_list = ds1000_loader.load_oracle_list()
    match_success_rate = 0
    sample_num = 0
    result_list = []
    for oracle in oracle_list[:1000]:
        lib = oracle['qs_id'].split('_')[0]
        if lib.lower() == 'scipy': continue
        matched_api_list = match_ds1000_doc(lib, oracle['output'])
        result_list.append(dict(qs_id=oracle['qs_id'], oracle_docs=matched_api_list, output=oracle['output']))
        if len(matched_api_list) > 0: match_success_rate += 1
        else:
            print(lib)
            print([oracle['output']])
        sample_num += 1


    with open('../DS-1000/oracle.json', 'w') as f:

    print('Match success rate: ', match_success_rate/sample_num)


