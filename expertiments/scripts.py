# run single llm
python RunOracleSingle.py --dataset conala --model openai-new --mode single --test-prompt

# transfer result to server
scp -P 34123 Code_RAG_Benchmark/data/conala/new_results/single_gpt-4o-mini.json zhaoshengming@129.128.209.206:~/Code_RAG_Benchmark/data/conala/new_results

# eval
python Eval.py --dataset conala --model openai-new --mode single


# run oracle llm
python RunOracleSingle.py --dataset conala --model openai-new --mode oracle --test-prompt

# transfer result to server
scp -P 34123 Code_RAG_Benchmark/data/conala/new_results/oracle_gpt-4o-mini.json zhaoshengming@129.128.209.206:~/Code_RAG_Benchmark/data/conala/new_results

# eval
python Eval.py --dataset conala --model openai-new --mode oracle


# run recall exp
python RunOracleSingle.py --dataset conala --model openai-new --mode recall --recall 1.0 --test-prompt