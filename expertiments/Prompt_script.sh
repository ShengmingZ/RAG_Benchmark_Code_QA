# script for LLAMA2 + CODE dataset + Single Oracle analyze

MODEL='openai-new'

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

#DATASET='NQ'
#
#/opt/homebrew/bin/python3.8 RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt few-shot &
#/opt/homebrew/bin/python3.8 RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt emotion &
## wait
#
#/opt/homebrew/bin/python3.8 RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt CoT &
#/opt/homebrew/bin/python3.8 RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt zero-shot-CoT &
## wait
#
#/opt/homebrew/bin/python3.8 RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt Least-to-Most &
#/opt/homebrew/bin/python3.8 RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt Plan-and-Solve &
## wait
#
#/opt/homebrew/bin/python3.8 RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt self-refine &
#/opt/homebrew/bin/python3.8 RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt CoN &
## wait


python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt Least-to-Most &
wait

python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt self-refine &
wait




DATASET='TriviaQA'

/opt/homebrew/bin/python3.8 RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt few-shot &
/opt/homebrew/bin/python3.8 RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt emotion &
# wait

/opt/homebrew/bin/python3.8 RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt CoT &
/opt/homebrew/bin/python3.8 RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt zero-shot-CoT &
# wait

/opt/homebrew/bin/python3.8 RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt Least-to-Most &
/opt/homebrew/bin/python3.8 RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt Plan-and-Solve &
# wait

/opt/homebrew/bin/python3.8 RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt self-refine &
/opt/homebrew/bin/python3.8 RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt CoN &
# wait




DATASET='hotpotQA'

/opt/homebrew/bin/python3.8 RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt few-shot &
/opt/homebrew/bin/python3.8 RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt emotion &
# wait

/opt/homebrew/bin/python3.8 RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt CoT &
/opt/homebrew/bin/python3.8 RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt zero-shot-CoT &
# wait

/opt/homebrew/bin/python3.8 RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt Least-to-Most &
/opt/homebrew/bin/python3.8 RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt Plan-and-Solve &
# wait

/opt/homebrew/bin/python3.8 RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt self-refine &
/opt/homebrew/bin/python3.8 RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt CoN &
# wait