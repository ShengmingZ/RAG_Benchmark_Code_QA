import json
from run_model import chatgpt
from sparse_retriever import tldr_ret_result_file_line, tldr_ret_result_file_whole, tldr_qs_file, tldr_doc_file_whole, tldr_doc_file_line

tldr_questions = json.load(open(tldr_qs_file, 'r'))
tldr_ret_result_whole = json.load(open(tldr_ret_result_file_whole, 'r'))
tldr_ret_result_line = json.load(open(tldr_ret_result_file_line, 'r'))
tldr_doc_whole = json.load(open(tldr_doc_file_whole, 'r'))
tldr_doc_line = json.load(open(tldr_doc_file_line, 'r'))

ret_whole = tldr_ret_result_whole[0]
top_k_whole = 5
whole_ret_cmd = [item['lib_key'] for item in ret_whole[:top_k_whole]]
ret_material_whole = ''
for cmd in whole_ret_cmd:
    ret_material_whole += tldr_doc_whole[cmd]
    ret_material_whole += '\n'
# print(ret_material_whole, end='\n\n\n')
print(len(ret_material_whole))

ret_line = tldr_ret_result_line[0]
processes_lib_key = list()
for single_cmd_ret_line in ret_line:
    processes_lib_key.append([item['lib_key'] for item in single_cmd_ret_line])
ret_material_line = ''
for cmd in processes_lib_key:
    for lib_key in cmd:
        ret_material_line += tldr_doc_line[lib_key]
    ret_material_line += '\n'
# print(ret_material_line)
print(len(ret_material_line))

tldr_qs = json.load(open(tldr_qs_file, 'r'))
qs = tldr_qs[0]
messages = "write shell code according to the natural language description, " \
           "and I also provide some information that might help\n" \
           f"Description: {qs['nl']}\n" \
           f"info: {ret_material_line}"
print(messages)
print(chatgpt(messages)[0])


# 1 generate with top-5 manual
# 2 generate with top-5 manual's retrieved sentence
# 3 generate with oracle manual
# 4 generate with oracle manual's retrieved sentence
# 5 interactive generate and retrieve

# create dataset for retrieval-based code generation
# discover the flaws of famous code datasets (ineffective in evaluating the retrieval-based code generation system)
# open domain code generation: larger docs, incomplete material, mixed of docs and code snippets
# human-eval + retrieval
# mitigate the influence of wrong retrieval (improper retrieve can mislead the model (use bm25 on conala))
# how to design retrieval corpus
# the problem of RAG in production: generators can be in-context, retrievers preform bad off-shelf

# explore more datasets and retrievers
# conduct further experiments
