#
#mv ../data/NQ/new_results/model_llama2-13b-chat_n_1_retrieval_recall_0.0.json ../data/NQ/new_results/recall-0_llama2-13b.json
#
#mv ../data/NQ/new_results/model_llama2-13b-chat_n_1_retrieval_recall_0.2.json ../data/NQ/new_results/recall-0.2_llama2-13b.json
#
#mv ../data/NQ/new_results/model_llama2-13b-chat_n_1_retrieval_recall_0.4.json ../data/NQ/new_results/recall-0.4_llama2-13b.json
#
#mv ../data/NQ/new_results/model_llama2-13b-chat_n_1_retrieval_recall_0.6.json ../data/NQ/new_results/recall-0.6_llama2-13b.json
#
#mv ../data/NQ/new_results/model_llama2-13b-chat_n_1_retrieval_recall_0.8.json ../data/NQ/new_results/recall-0.8_llama2-13b.json
#
#mv ../data/NQ/new_results/model_llama2-13b-chat_n_1_retrieval_recall_1.0.json ../data/NQ/new_results/oracle_llama2-13b.json


python Eval.py --dataset NQ --model openai-new --mode DocNum --k 1

python Eval.py --dataset NQ --model openai-new --mode DocNum --k 3

python Eval.py --dataset NQ --model openai-new --mode DocNum --k 5

python Eval.py --dataset NQ --model openai-new --mode DocNum --k 10

python Eval.py --dataset NQ --model openai-new --mode DocNum --k 15

python Eval.py --dataset NQ --model openai-new --mode DocNum --k 20

python Eval.py --dataset NQ --model openai-new --mode DocNum --k 30

python Eval.py --dataset NQ --model openai-new --mode DocNum --k 40


