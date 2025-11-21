# script for LLAMA2 + CODE dataset + Single Oracle analyze

DATASET='hotpotQA'
MODEL='openai-new'

python RunOracleSingle.py --dataset hotpotQA --model openai-new --mode single &
python RunOracleSingle.py --dataset hotpotQA --model openai-new --mode oracle &
python RunOracleSingle.py --dataset hotpotQA --model openai-new --mode recall --recall 0.0 &
python RunOracleSingle.py --dataset hotpotQA --model openai-new --mode recall --recall 0.2 &
wait

python RunOracleSingle.py --dataset hotpotQA --model openai-new --mode recall --recall 0.4 &
python RunOracleSingle.py --dataset hotpotQA --model openai-new --mode recall --recall 0.6 &
python RunOracleSingle.py --dataset hotpotQA --model openai-new --mode recall --recall 0.8 &
wait



DATASET='TriviaQA'

python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode single &
python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode oracle &
python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode recall --recall 0.0 &
python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode recall --recall 0.2 &
wait

python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode recall --recall 0.4 &
python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode recall --recall 0.6 &
python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode recall --recall 0.8 &
wait



DATASET='hotpotQA'

python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode single &
python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode oracle &
python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode recall --recall 0.0 &
python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode recall --recall 0.2 &
wait

python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode recall --recall 0.4 &
python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode recall --recall 0.6 &
python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode recall --recall 0.8 &
wait
