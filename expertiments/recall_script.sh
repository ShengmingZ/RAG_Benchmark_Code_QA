# script for LLAMA2 + CODE dataset + Single Oracle analyze

DATASET='conala'
MODEL='openai-new'

python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode single &
python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode oracle &
wait

python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode recall --recall 0.0 &
python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode recall --recall 0.2 &
wait

python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode recall --recall 0.4 &
python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode recall --recall 0.6 &
wait

python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode recall --recall 0.8 &
wait



DATASET='DS1000'

python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode single &
python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode oracle &
wait

python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode recall --recall 0.0 &
python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode recall --recall 0.2 &
wait

python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode recall --recall 0.4 &
python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode recall --recall 0.6 &
wait

python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode recall --recall 0.8 &
wait



DATASET='pandas_numpy_eval'

python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode single &
python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode oracle &
wait

python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode recall --recall 0.0 &
python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode recall --recall 0.2 &
wait

python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode recall --recall 0.4 &
python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode recall --recall 0.6 &
wait

python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode recall --recall 0.8 &
wait
