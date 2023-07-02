# GEMM

## General GPU

- Consists of threads, which contain an ALU + FPU
- ALU emulates FP-ops with INT-ops (takes several cycles)
- FPU achieves almost one FP-op per cycle
- three-level compute hierarchy: Grid -> Block -> Thread

## GEMM specs

Ran this on my RTX 3070 Laptop GPU

- Max throughput per datasheet: 16 TFLOP/s
- Max memory BW: 384 GB/s
- ALUs: 5120
- Clock speed: 1935 MHz (measured)
- Measured max throughput: 5120 \* 1935 MHz \* 2 FLOP = 20 TFLOP/s

Multiplying two 2048 matrices requires

- flops: 2 * (2048)^3 = 17 GFLOP
- reads: 3 \* (2048)^2 \* 4B = 50 MB
- write: (2048)^2 * 4B = 17 MB

Theoretical time for the operations:

- compute: 17 GFLOP / 16 TFLOP/s = 1.1 ms
- memory: 70 MB / 384 GB/s = 0.2 ms

## Compute model

- 32 * 32 = 1024 threads

## Naive kernel

~ x100 slower than optimum

- time: 125 ms
- throughput: 138 GFLOP/S

Why?

- Every thread
