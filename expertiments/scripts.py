# run single llm
python RunOracleSingle.py --dataset conala --model openai-new --mode single --test-prompt

# transfer result to server
scp -P 34123 Code_RAG_Benchmark/data/conala/new_results/single_gpt-4o-mini.json zhaoshengming@129.128.209.206:~/Code_RAG_Benchmark/data/conala/new_results/single_gpt-4o-mini.json

# eval
python Eval.py --dataset conala --model openai-new --mode single
