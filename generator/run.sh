#!/bin/zsh

model=$1
dataset=$2
retriever=$3
analysis_type=$4

# ret acc
if [[ $analysis_type == "retrieval_recall" ]]
then
  python --model $model --dataset $dataset --retriever $retriever --analysis_type $analysis_type --ret_acc 1
  python --model llama3-8b --dataset conala --retriever BM25 --analysis_type retrieval_recall --ret_acc 0.8
  python --model llama3-8b --dataset conala --retriever BM25 --analysis_type retrieval_recall --ret_acc 0.6
  python --model llama3-8b --dataset conala --retriever BM25 --analysis_type retrieval_recall --ret_acc 0.4
  python --model llama3-8b --dataset conala --retriever BM25 --analysis_type retrieval_recall --ret_acc 0.2
  python --model llama3-8b --dataset conala --retriever BM25 --analysis_type retrieval_recall --ret_acc 0
fi