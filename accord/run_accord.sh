#!/bin/bash
mpirun -np 2 julia accord2.jl -i C_Xhp_5k.npy -o out -l 0.1 -t 0.5