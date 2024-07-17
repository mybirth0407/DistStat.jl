#!/bin/bash
mpirun -np 4 julia -t auto accord2.jl -i C_Xhp_5k.npy -o out -l 0.1 -t 0.5 -e 1e-7 --max_outer 20
