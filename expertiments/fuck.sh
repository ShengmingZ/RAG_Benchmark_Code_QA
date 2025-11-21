# DATASET='TriviaQA'
# MODEL='llama-old-qa'

# cp ../old_data/$DATASET/results/model_gpt-3.5-turbo-0125_n_1_retrieval_doc_selection_top_1.json ../data/$DATASET/new_results/DocNum/1_gpt-3-5-turbo.json

# cp ../old_data/$DATASET/results/model_gpt-3.5-turbo-0125_n_1_retrieval_doc_selection_top_5.json ../data/$DATASET/new_results/DocNum/5_gpt-3-5-turbo.json

# cp ../old_data/$DATASET/results/model_gpt-3.5-turbo-0125_n_1_retrieval_doc_selection_top_10.json ../data/$DATASET/new_results/DocNum/10_gpt-3-5-turbo.json

# cp ../old_data/$DATASET/results/model_gpt-3.5-turbo-0125_n_1_retrieval_doc_selection_top_15.json ../data/$DATASET/new_results/DocNum/15_gpt-3-5-turbo.json

# cp ../old_data/$DATASET/results/model_gpt-3.5-turbo-0125_n_1_retrieval_doc_selection_top_20.json ../data/$DATASET/new_results/DocNum/20_gpt-3-5-turbo.json

# cp ../old_data/$DATASET/results/model_gpt-3.5-turbo-0125_n_1_retrieval_doc_selection_top_30.json ../data/$DATASET/new_results/DocNum/30_gpt-3-5-turbo.json

# cp ../old_data/$DATASET/results/model_gpt-3.5-turbo-0125_n_1_retrieval_doc_selection_top_40.json ../data/$DATASET/new_results/DocNum/40_gpt-3-5-turbo.json



# python Eval.py --dataset $DATASET --model $MODEL --mode DocNum --k 1

# python Eval.py --dataset $DATASET --model $MODEL --mode DocNum --k 5

# python Eval.py --dataset $DATASET --model $MODEL --mode DocNum --k 10

# python Eval.py --dataset $DATASET --model $MODEL --mode DocNum --k 15

# python Eval.py --dataset $DATASET --model $MODEL --mode DocNum --k 20

# python Eval.py --dataset $DATASET --model $MODEL --mode DocNum --k 30

# python Eval.py --dataset $DATASET --model $MODEL --mode DocNum --k 40


MODEL='llama-old-qa'

# python RunOracleSingle.py --dataset NQ --model $MODEL --mode prompt --prompt emotion &
# python RunOracleSingle.py --dataset TriviaQA --model $MODEL --mode prompt --prompt emotion &

python RunOracleSingle.py --dataset hotpotQA --model llama-old-qa --mode prompt --prompt emotion &
python RunOracleSingle.py --dataset NQ --model llama-old-qa --mode prompt --prompt zero-shot-CoT &
wait

python RunOracleSingle.py --dataset TriviaQA --model llama-old-qa --mode prompt --prompt zero-shot-CoT &
python RunOracleSingle.py --dataset hotpotQA --model llama-old-qa --mode prompt --prompt zero-shot-CoT &
wait

# OLD_MODEL='gpt-3.5-turbo-0125'
# NEW_MODEL='gpt-3-5-turbo'
# DATASET='NQ'

# cp ../old_data/$DATASET/results/model_${OLD_MODEL}_n_1_prompt_method_3shot.json ../data/$DATASET/new_results/Prompt/few-shot_${NEW_MODEL}.json

# cp ../old_data/$DATASET/results/model_${OLD_MODEL}_n_1_prompt_method_cot.json ../data/$DATASET/new_results/Prompt/CoT_${NEW_MODEL}.json

# cp ../old_data/$DATASET/results/model_${OLD_MODEL}_n_1_prompt_method_least_to_most.json ../data/$DATASET/new_results/Prompt/Least-to-Most_${NEW_MODEL}.json

# cp ../old_data/$DATASET/results/model_${OLD_MODEL}_n_1_prompt_method_plan_and_solve.json ../data/$DATASET/new_results/Prompt/Plan-and-Solve_${NEW_MODEL}.json

# cp ../old_data/$DATASET/results/model_${OLD_MODEL}_n_1_prompt_method_self-refine.json ../data/$DATASET/new_results/Prompt/self-refine_${NEW_MODEL}.json

# cp ../old_data/$DATASET/results/model_${OLD_MODEL}_n_1_prompt_method_con.json ../data/$DATASET/new_results/Prompt/CoN_${NEW_MODEL}.json