# script for LLAMA2 + CODE dataset + Single Oracle analyze

MODEL='openai-old'


python RunOracleSingle.py --dataset NQ --model $MODEL --mode DocNum --k 3 &
# wait

python RunOracleSingle.py --dataset TriviaQA --model $MODEL --mode DocNum --k 3 &
# wait

python RunOracleSingle.py --dataset hotpotQA --model $MODEL --mode DocNum --k 3 &





# python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 1 &
# python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 3 &
# wait

# python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 5 &
# python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 7 &
# wait

# python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 10 &
# python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 13 &
# wait

# python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 16 &
# wait

# python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 20 &
# wait
