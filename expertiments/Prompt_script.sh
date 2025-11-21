# script for LLAMA2 + CODE dataset + Single Oracle analyze

MODEL='llama-old-code'

DATASET='pandas_numpy_eval'

python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt few-shot &
wait

python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt CoT &
wait

python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt emotion &
python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt zero-shot-CoT &
wait

python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt Plan-and-Solve &
wait

python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt CoN &
wait

python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt Least-to-Most &
wait

python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt self-refine &
wait


