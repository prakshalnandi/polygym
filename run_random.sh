rm -rf /tmp/procs_out
mkdir /tmp/procs_out

# PolyEnv
#POLYITEDIR=/home/s2136718/MLPC LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/s2136718/MLPC/scala-isl-utils/libs OMP_NUM_THREADS=6 PYTHONPATH="" taskset -c 0-5   python train_random.py --out_dir /home/s2136718/tmp/out_bias_select_dep --with_polyenv --stop_at 25 --with_polyenv_sampling_bias bias_coeff0 >> /tmp/procs_out/poly_coeff0.out 2>&1 &

POLYITEDIR=/home/s2136718/MLPC LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/s2136718/MLPC/scala-isl-utils/libs  python train_random.py --out_dir /home/s2136718/tmp/out_bias_select_dep --with_polyenv --stop_at 1 -with_polyenv_sampling_bias bias_select_dep >> /tmp/procs_out/poly_select_dep.out 2>&1 &

#POLYITEDIR=/home/s2136718/MLPC LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/s2136718/MLPC/scala-isl-utils/libs  python -m pudb train_random.py --out_dir /home/s2136718/tmp/out_bias_select_dep --with_polyenv --stop_at 3 -with_polyenv_sampling_bias bias_select_dep

#wait

# Baselines
#POLYITEDIR=/home/s2136718/MLPC LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/s2136718/MLPC/scala-isl-utils/libs OMP_NUM_THREADS=6 PYTHONPATH="" taskset -c 0-5   python train_random.py --out_dir /home/s2136718/tmp/out --with_baselines >> /tmp/procs_out/0.out 2>&1 &
wait
