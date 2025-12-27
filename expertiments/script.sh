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


#python Eval.py --dataset NQ --model openai-new --mode DocNum --k 1
#
#python Eval.py --dataset NQ --model openai-new --mode DocNum --k 3
#
#python Eval.py --dataset NQ --model openai-new --mode DocNum --k 5
#
#python Eval.py --dataset NQ --model openai-new --mode DocNum --k 10
#
#python Eval.py --dataset NQ --model openai-new --mode DocNum --k 15
#
#python Eval.py --dataset NQ --model openai-new --mode DocNum --k 20
#
#python Eval.py --dataset NQ --model openai-new --mode DocNum --k 30
#
#python Eval.py --dataset NQ --model openai-new --mode DocNum --k 40


# more retriever exps CoNaLa

#python RunExps.py --dataset conala --model openai-new --retriever miniLM --mode DocNum --k 1 &
#
#python RunExps.py --dataset conala --model openai-new --retriever miniLM --mode DocNum --k 3 &
#
#python RunExps.py --dataset conala --model openai-new --retriever miniLM --mode DocNum --k 5 &
#
#python RunExps.py --dataset conala --model openai-new --retriever miniLM --mode DocNum --k 7 &
#
#python RunExps.py --dataset conala --model openai-new --retriever miniLM --mode DocNum --k 10 &
#
#python RunExps.py --dataset conala --model openai-new --retriever miniLM --mode DocNum --k 13 &
#
#python RunExps.py --dataset conala --model openai-new --retriever miniLM --mode DocNum --k 16 &
#
#python RunExps.py --dataset conala --model openai-new --retriever miniLM --mode DocNum --k 20 &

# more retriever exps DS1000 BM25

#python RunExps.py --dataset DS1000 --model openai-new --retriever BM25 --mode DocNum --k 1 &
#
#python RunExps.py --dataset DS1000 --model openai-new --retriever BM25 --mode DocNum --k 3 &
#
#python RunExps.py --dataset DS1000 --model openai-new --retriever BM25 --mode DocNum --k 5 &
#
#python RunExps.py --dataset DS1000 --model openai-new --retriever BM25 --mode DocNum --k 7 &
#
#python RunExps.py --dataset DS1000 --model openai-new --retriever BM25 --mode DocNum --k 10 &
#
#python RunExps.py --dataset DS1000 --model openai-new --retriever BM25 --mode DocNum --k 13 &
#
#python RunExps.py --dataset DS1000 --model openai-new --retriever BM25 --mode DocNum --k 16 &
#
#python RunExps.py --dataset DS1000 --model openai-new --retriever BM25 --mode DocNum --k 20 &


#python RunExps.py --dataset pandas_numpy_eval --model openai-new --retriever miniLM --mode DocNum --k 1 &
#
#python RunExps.py --dataset pandas_numpy_eval --model openai-new --retriever miniLM --mode DocNum --k 3 &
#
#python RunExps.py --dataset pandas_numpy_eval --model openai-new --retriever miniLM --mode DocNum --k 5 &
#
#python RunExps.py --dataset pandas_numpy_eval --model openai-new --retriever miniLM --mode DocNum --k 7 &
#
#python RunExps.py --dataset pandas_numpy_eval --model openai-new --retriever miniLM --mode DocNum --k 10 &
#
#python RunExps.py --dataset pandas_numpy_eval --model openai-new --retriever miniLM --mode DocNum --k 13 &
#
#python RunExps.py --dataset pandas_numpy_eval --model openai-new --retriever miniLM --mode DocNum --k 16 &
#
#python RunExps.py --dataset pandas_numpy_eval --model openai-new --retriever miniLM --mode DocNum --k 20 &


#python RunExps.py --dataset NQ --model openai-new --retriever miniLM --mode DocNum --k 1 &
#
#python RunExps.py --dataset NQ --model openai-new --retriever miniLM --mode DocNum --k 3 &
#
#python RunExps.py --dataset NQ --model openai-new --retriever miniLM --mode DocNum --k 5 &
#
#python RunExps.py --dataset NQ --model openai-new --retriever miniLM --mode DocNum --k 10 &
#
#python RunExps.py --dataset NQ --model openai-new --retriever miniLM --mode DocNum --k 15 &
#
#python RunExps.py --dataset NQ --model openai-new --retriever miniLM --mode DocNum --k 20 &
#
#python RunExps.py --dataset NQ --model openai-new --retriever miniLM --mode DocNum --k 30 &
#
#python RunExps.py --dataset NQ --model openai-new --retriever miniLM --mode DocNum --k 40 &
#
#
#python RunExps.py --dataset TriviaQA --model openai-new --retriever BM25 --mode DocNum --k 1 &
#
#python RunExps.py --dataset TriviaQA --model openai-new --retriever BM25 --mode DocNum --k 3 &
#
#python RunExps.py --dataset TriviaQA --model openai-new --retriever BM25 --mode DocNum --k 5 &
#
#python RunExps.py --dataset TriviaQA --model openai-new --retriever BM25 --mode DocNum --k 10 &
#
#python RunExps.py --dataset TriviaQA --model openai-new --retriever BM25 --mode DocNum --k 15 &
#
#python RunExps.py --dataset TriviaQA --model openai-new --retriever BM25 --mode DocNum --k 20 &
#
#python RunExps.py --dataset TriviaQA --model openai-new --retriever BM25 --mode DocNum --k 30 &
#
#python RunExps.py --dataset TriviaQA --model openai-new --retriever BM25 --mode DocNum --k 40 &
#
#
#python RunExps.py --dataset hotpotQA --model openai-new --retriever miniLM --mode DocNum --k 1 &
#
#python RunExps.py --dataset hotpotQA --model openai-new --retriever miniLM --mode DocNum --k 3 &
#
#python RunExps.py --dataset hotpotQA --model openai-new --retriever miniLM --mode DocNum --k 5 &
#
#python RunExps.py --dataset hotpotQA --model openai-new --retriever miniLM --mode DocNum --k 10 &
#
#python RunExps.py --dataset hotpotQA --model openai-new --retriever miniLM --mode DocNum --k 15 &
#
#python RunExps.py --dataset hotpotQA --model openai-new --retriever miniLM --mode DocNum --k 20 &
#
#python RunExps.py --dataset hotpotQA --model openai-new --retriever miniLM --mode DocNum --k 30 &
#
#python RunExps.py --dataset hotpotQA --model openai-new --retriever miniLM --mode DocNum --k 40 &


#python RunExps.py --dataset conala --model openai-new --retriever openai-embedding --mode realistic_recall --recall 0 &
#
#python RunExps.py --dataset conala --model openai-new --retriever openai-embedding --mode realistic_recall --recall 0.2 &
#
#python RunExps.py --dataset conala --model openai-new --retriever openai-embedding --mode realistic_recall --recall 0.4 &
#
#python RunExps.py --dataset conala --model openai-new --retriever openai-embedding --mode realistic_recall --recall 0.6  &
#
#python RunExps.py --dataset conala --model openai-new --retriever openai-embedding --mode realistic_recall --recall 0.8 &
#
#python RunExps.py --dataset conala --model openai-new --retriever openai-embedding --mode realistic_recall --recall 1.0 &


#python RunExps.py --dataset pandas_numpy_eval --model openai-new --retriever openai-embedding --mode realistic_recall --recall 0 &
#
#python RunExps.py --dataset pandas_numpy_eval --model openai-new --retriever openai-embedding --mode realistic_recall --recall 0.2 &
#
#python RunExps.py --dataset pandas_numpy_eval --model openai-new --retriever openai-embedding --mode realistic_recall --recall 0.4 &
#
#python RunExps.py --dataset pandas_numpy_eval --model openai-new --retriever openai-embedding --mode realistic_recall --recall 0.6  &
#
#python RunExps.py --dataset pandas_numpy_eval --model openai-new --retriever openai-embedding --mode realistic_recall --recall 0.8 &
#
#python RunExps.py --dataset pandas_numpy_eval --model openai-new --retriever openai-embedding --mode realistic_recall --recall 1.0 &


python RunExps.py --dataset NQ --model openai-new --retriever openai-embedding --mode realistic_recall --recall 0 &

python RunExps.py --dataset NQ --model openai-new --retriever openai-embedding --mode realistic_recall --recall 0.2 &

python RunExps.py --dataset NQ --model openai-new --retriever openai-embedding --mode realistic_recall --recall 0.4 &

python RunExps.py --dataset NQ --model openai-new --retriever openai-embedding --mode realistic_recall --recall 0.6  &

python RunExps.py --dataset NQ --model openai-new --retriever openai-embedding --mode realistic_recall --recall 0.8 &

python RunExps.py --dataset NQ --model openai-new --retriever openai-embedding --mode realistic_recall --recall 1.0 &


python RunExps.py --dataset TriviaQA --model openai-new --retriever openai-embedding --mode realistic_recall --recall 0 &

python RunExps.py --dataset TriviaQA --model openai-new --retriever openai-embedding --mode realistic_recall --recall 0.2 &

python RunExps.py --dataset TriviaQA --model openai-new --retriever openai-embedding --mode realistic_recall --recall 0.4 &

python RunExps.py --dataset TriviaQA --model openai-new --retriever openai-embedding --mode realistic_recall --recall 0.6  &

python RunExps.py --dataset TriviaQA --model openai-new --retriever openai-embedding --mode realistic_recall --recall 0.8 &

python RunExps.py --dataset TriviaQA --model openai-new --retriever openai-embedding --mode realistic_recall --recall 1.0 &


python RunExps.py --dataset hotpotQA --model openai-new --retriever openai-embedding --mode realistic_recall --recall 0 &

python RunExps.py --dataset hotpotQA --model openai-new --retriever openai-embedding --mode realistic_recall --recall 0.2 &

python RunExps.py --dataset hotpotQA --model openai-new --retriever openai-embedding --mode realistic_recall --recall 0.4 &

python RunExps.py --dataset hotpotQA --model openai-new --retriever openai-embedding --mode realistic_recall --recall 0.6  &

python RunExps.py --dataset hotpotQA --model openai-new --retriever openai-embedding --mode realistic_recall --recall 0.8 &

python RunExps.py --dataset hotpotQA --model openai-new --retriever openai-embedding --mode realistic_recall --recall 1.0 &
