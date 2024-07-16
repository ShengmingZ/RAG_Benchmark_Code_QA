import json
import sys, platform
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from generator.generate_utils import generate_config
from dataset_utils.conala_utils import ConalaLoader


in_program_call = '--action eval_pred --model gpt-3.5-turbo-0125 --temperature 0.0 --dataset conala --retriever openai-embedding --analysis_type retrieval_recall --n 1 --ret_acc 1'
args = generate_config(in_program_call)
eval_file = args.result_save_file.replace('.json', '_eval.json')
eval_datas = json.load(open(eval_file))
retrieval_records = eval_datas['retrieval_records']
ret_eval_records = eval_datas['ret_eval_records']
eval_records = eval_datas['eval_records']
output_records = eval_datas['output_records']
qs_list = ConalaLoader().load_qs_list()


ret_eval_true_eval_false_count = 0
ret_eval_true_eval_true_count = 0
ret_eval_false_eval_true_count = 0
ret_eval_false_eval_false_count = 0
for qs in qs_list:
    qs_id = qs['qs_id']
    print(eval_records[qs_id])
    eval = eval_records[qs_id]['0'][0][1]['passed']
    retrieval = retrieval_records[qs_id]
    ret_eval = ret_eval_records[qs_id]
    output = output_records[qs_id]


