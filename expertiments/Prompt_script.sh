# script for LLAMA2 + CODE dataset + Single Oracle analyze

MODEL='llama-old-code'

DATASET='conala'

python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt few-shot &
python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt emotion &
wait

python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt CoT &
python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt zero-shot-CoT &
wait

python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt Least-to-Most &
python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt Plan-and-Solve &
wait

python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt self-refine &
python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt CoN &
wait


