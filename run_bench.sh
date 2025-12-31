#!/bin/bash

# Define file names
SRC="nbody_bench.c"
OUT_CSV="benchmark_results.csv"

# 1. Compile CPU Version
echo "[1/3] Compiling CPU version..."
nvc -O3 -mp -o nbody_cpu $SRC -lm

# 2. Compile GPU Version
echo "[2/3] Compiling GPU version..."
nvc -acc -gpu=ccnative,mem:managed -mp -o nbody_gpu $SRC -lm

# 3. Initialize CSV File with NEW columns
echo "N,CPU Time(s),CPU Perf(GInt/s),GPU Time(s),GPU Perf(GInt/s)" > $OUT_CSV

# Define Test Sizes
SIZES=(20480 40960 65536 81920 131072 139264 278528)

echo ""
echo "Starting Benchmark Loop..."
echo "----------------------------------------------------------------------------------"
printf "%-10s | %-12s | %-15s | %-12s | %-15s\n" "N" "CPU Time" "CPU Perf" "GPU Time" "GPU Perf"
echo "----------------------------------------------------------------------------------"

for N in "${SIZES[@]}"; do
    # --- Run CPU ---
    # The C code outputs: N,Time,Perf
    RAW_CPU=$(./nbody_cpu $N)
    IFS=',' read -r N_VAL CPU_TIME CPU_PERF <<< "$RAW_CPU"

    # --- Run GPU ---
    RAW_GPU=$(./nbody_gpu $N)
    IFS=',' read -r N_VAL GPU_TIME GPU_PERF <<< "$RAW_GPU"

    # --- Save to CSV ---
    echo "$N,$CPU_TIME,$CPU_PERF,$GPU_TIME,$GPU_PERF" >> $OUT_CSV

    # --- Print to Screen ---
    printf "%-10s | %-12s | %-15s | %-12s | %-15s\n" "$N" "$CPU_TIME" "$CPU_PERF" "$GPU_TIME" "$GPU_PERF"
done

echo "----------------------------------------------------------------------------------"
echo "Benchmark Complete."
echo "Results saved to: $OUT_CSV"
