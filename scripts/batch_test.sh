#!/bin/bash


date
for i in `seq 1 6`
do
{
echo "go"
curl -X POST http://127.0.0.1:8000/inference -d '{"x": "MOSS is an AI assistant developed by the FudanNLP Lab and Shanghai AI Lab. Below is a conversation between MOSS and human. [Human]: hello ! <eoh> [MOSS]: hello! <eoa> [Human]: Can you tell me the java code of quicksort ? <eoh> "}'

} &
done
wait  ##等待所有子后台进程结束
date

