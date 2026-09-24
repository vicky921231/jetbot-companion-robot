#!/bin/bash
# try_orders.sh -- find the inputOrder that the TensorRT 7 NMS plugin accepts.
#
# Run inside the JetBot docker container:
#     cd /workspace/ssd_build && bash try_orders.sh
#
# WHY A SHELL LOOP AND NOT A PYTHON LOOP
# --------------------------------------
# A wrong inputOrder makes TensorRT's NMS plugin fail a C++ assertion
# ("#assertionnmsPlugin.cpp,246") which calls abort(). That kills the whole
# process -- Python cannot catch it, so each permutation needs its own process.
#
# 1,2,0 is what jetbot hardcodes and it is already known to fail on TensorRT 7,
# so it is tried last.

set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/ssd_mobilenet_v2_coco.engine"
LOG="$HERE/try_orders.log"

ORDERS="0,2,1 1,0,2 2,1,0 0,1,2 2,0,1 1,2,0"

rm -f "$OUT"
: > "$LOG"

echo "=============================================="
echo "Trying inputOrder permutations"
echo "Each attempt takes a few minutes. Be patient."
echo "Full output is appended to: $LOG"
echo "=============================================="
echo ""

for ORDER in $ORDERS; do
    echo "----------------------------------------------"
    echo ">>> trying inputOrder = $ORDER"
    echo "----------------------------------------------"
    {
        echo ""
        echo "########## inputOrder $ORDER ##########"
    } >> "$LOG"

    python3 "$HERE/build_ssd_engine.py" --input-order "$ORDER" 2>&1 | tee -a "$LOG"

    if [ -s "$OUT" ]; then
        echo ""
        echo "=============================================="
        echo "SUCCESS -- working inputOrder is $ORDER"
        echo "engine: $OUT"
        ls -lh "$OUT"
        echo ""
        echo "Now copy it next to the notebook:"
        echo "  cp $OUT /workspace/jetbot/notebooks/object_following/"
        echo "=============================================="
        exit 0
    fi

    echo ">>> $ORDER failed, trying next"
    echo ""
done

echo ""
echo "=============================================="
echo "All six permutations failed."
echo "Send back the last 60 lines of:"
echo "  $LOG"
echo "=============================================="
exit 1
