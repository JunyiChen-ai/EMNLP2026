#!/bin/bash
while true; do
    count=$(ps aux | grep "run_ablations_v2" | grep -v grep | grep -v conda | wc -l)
    if [ "$count" -eq 0 ]; then
        echo "ALL ABLATION EXPERIMENTS COMPLETED at $(date)"
        echo "=== Results ==="
        for f in /home/junyi/EMNLP2026/logs/ablation_*.log; do
            echo "=== $(basename $f) ==="
            cat "$f"
            echo
        done
        break
    fi
    sleep 60
done
