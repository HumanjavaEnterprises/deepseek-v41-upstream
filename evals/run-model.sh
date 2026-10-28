#!/bin/bash
# run-model.sh TAG GGUF — identical setup for every model: upstream de5122d8, layers 0-24 on GPU, 16 threads
set -u
E=$(dirname "$0"); TAG=$1; M=$2; BIN=${LLAMA_BIN:?set LLAMA_BIN to llama.cpp/build/bin}
OT='blk\.(2[5-9]|3[0-9])\.ffn_(up|down|gate)_exps=CPU'
$BIN/llama-perplexity -m "$M" -f $E/wikitext-2-test.txt -c 512 --chunks 150 -b 2048 -ub 2048 -ngl 99 -t 16 -ot "$OT" > $E/$TAG.ppl.log 2>&1
grep -E "Final estimate" $E/$TAG.ppl.log | sed "s/^/$TAG /"
$BIN/llama-server -m "$M" -c 8192 -np 1 -ngl 99 -t 16 -ub 2048 -ot "$OT" --host 127.0.0.1 --port 18926 > $E/$TAG.server.log 2>&1 &
P=$!
for i in $(seq 300); do curl -sf 127.0.0.1:18926/health >/dev/null && break; sleep 2; done
python3 $E/evals.py 18926 $TAG && mv $E/$TAG.jsonl $E/results/ 2>/dev/null
kill -TERM $P; wait $P
