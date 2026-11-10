#!/bin/bash
# 2026-11-01 key-target pre-bind chains (GPU0 only). Each step appends START/END to timings.log.
#   run 1: ./run_chain.sh names fit1 eval1 guard1
#   run 2: ./run_chain.sh fit2 eval2 eval2_seeds2 guard2 guard2_seeds2 paired2
S=$(cd "$(dirname "$0")" && pwd)
L=${DETANGLE_RESEARCH_ROOT:-$PWD/detangle-research}/runs/2026-11-01-knn-prebind
PY=${DETANGLE_COMFY_PYTHON:-python}
mkdir -p "$L"
cd $S
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=${DETANGLE_COMFYUI:?set DETANGLE_COMFYUI}/pinned_site
used=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$used" -gt 4000 ]; then echo "GPU0 busy ($used MiB used); not starting" >&2; exit 2; fi
step() { name=$1; shift; echo "START $name $(date -Iseconds)" >> $L/timings.log; $PY -u knn_prebind.py "$@"; rc=$?; echo "END $name $(date -Iseconds) rc=$rc" >> $L/timings.log; return $rc; }
for s in "$@"; do
  case $s in
    names) step names names ;;
    fit1) step fit1 fit --tags knn1_200,knn3_200,mean_cap487_400 ;;
    eval1) step eval1 eval --tags knn1_200,knn3_200,mean_cap487_400 --check ;;
    guard1) step guard1 guard --tags knn1_200,knn3_200,mean_cap487_400 ;;
    fit2) step run2_fit fit --tags knn1_nopugh_200,span_200,hull_200 ;;
    eval2) step run2_eval_span_hull eval --tags span_200,hull_200 --out-suffix _v2 ;;
    eval2_seeds2) step run2_eval_nopugh_seeds2 eval --tags knn1_nopugh_200 --seeds2 ;;
    guard2) step run2_guard guard --tags knn1_200,span_200,hull_200 --out-suffix _v2 ;;
    guard2_seeds2) step run2_guard_seeds2 guard --tags knn1_nopugh_200 --seeds2 ;;
    paired2) $PY paired.py --arms span_200,hull_200 --refs pb_b0_27_200,knn1_200 --guard guard_v2.json > $L/paired_v2.txt && \
             $PY paired.py --seeds2 --arms knn1_nopugh_200 --guard guard_seeds2.json > $L/paired_seeds2.txt ;;
  esac || exit 1
done
