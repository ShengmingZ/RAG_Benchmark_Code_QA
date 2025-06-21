# script for LLAMA2 + CODE dataset + Single Oracle analyze

python RunOracleSingle.py --dataset conala --model llama-old-code --mode single &

python RunOracleSingle.py --dataset conala --model llama-old-code --mode oracle

python RunOracleSingle.py --dataset conala --model llama-old-code --mode recall --recall 0.0

python RunOracleSingle.py --dataset conala --model llama-old-code --mode recall --recall 0.2

python RunOracleSingle.py --dataset conala --model llama-old-code --mode recall --recall 0.4

python RunOracleSingle.py --dataset conala --model llama-old-code --mode recall --recall 0.6

python RunOracleSingle.py --dataset conala --model llama-old-code --mode recall --recall 0.8



python RunOracleSingle.py --dataset DS1000 --model llama-old-code --mode single &

python RunOracleSingle.py --dataset DS1000 --model llama-old-code --mode oracle

python RunOracleSingle.py --dataset DS1000 --model llama-old-code --mode recall --recall 0.0

python RunOracleSingle.py --dataset DS1000 --model llama-old-code --mode recall --recall 0.2

python RunOracleSingle.py --dataset DS1000 --model llama-old-code --mode recall --recall 0.4

python RunOracleSingle.py --dataset DS1000 --model llama-old-code --mode recall --recall 0.6

python RunOracleSingle.py --dataset DS1000 --model llama-old-code --mode recall --recall 0.8



python RunOracleSingle.py --dataset pandas_numpy_eval --model llama-old-code --mode single &

python RunOracleSingle.py --dataset pandas_numpy_eval --model llama-old-code --mode oracle

python RunOracleSingle.py --dataset pandas_numpy_eval --model llama-old-code --mode recall --recall 0.0

python RunOracleSingle.py --dataset pandas_numpy_eval --model llama-old-code --mode recall --recall 0.2

python RunOracleSingle.py --dataset pandas_numpy_eval --model llama-old-code --mode recall --recall 0.4

python RunOracleSingle.py --dataset pandas_numpy_eval --model llama-old-code --mode recall --recall 0.6

python RunOracleSingle.py --dataset pandas_numpy_eval --model llama-old-code --mode recall --recall 0.8
