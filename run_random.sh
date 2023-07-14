rm -rf /tmp/procs_out
mkdir /tmp/procs_out
#import time


# PolyEnv
#POLYITEDIR=/home/s2136718/MLPC LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/s2136718/MLPC/scala-isl-utils/libs OMP_NUM_THREADS=6 PYTHONPATH="" taskset -c 0-5   python train_random.py --out_dir /home/s2136718/tmp/out_bias_select_dep --with_polyenv --stop_at 25 --with_polyenv_sampling_bias bias_coeff0 >> /tmp/procs_out/poly_coeff0.out 2>&1 &

#starttime = time.time()

### Basic RL
#POLYITEDIR=/home/s2136718/MLPC LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/s2136718/MLPC/scala-isl-utils/libs python train_random.py --out_dir /home/s2136718/tmp/out_bias_select_dep --with_polyenv --stop_at 5 --dep_rep complex_full -with_polyenv_sampling_bias bias_select_dep >> /tmp/procs_out/poly_select_dep.out 2>&1 &

POLYITEDIR=/home/s2136718/MLPC LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/s2136718/MLPC/scala-isl-utils/libs srun python train_random.py --out_dir /home/s2136718/tmp/out_bias_select_dep --with_polyenv --stop_at 5 --dep_rep complex_full -with_polyenv_sampling_bias bias_select_dep >> /tmp/procs_out/poly_select_dep.out 2>&1 &

### Deep RL
#POLYITEDIR=/home/s2136718/MLPC LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/s2136718/MLPC/scala-isl-utils/libs  python train_dl.py --out_dir /home/s2136718/tmp/out_bias_select_dep --with_polyenv --stop_at 1 --dep_rep complex_full -with_polyenv_sampling_bias bias_select_dep >> /tmp/procs_out/poly_select_dep.out 2>&1 &

### Deep RL with scalene profiling
#POLYITEDIR=/home/s2136718/MLPC LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/s2136718/MLPC/scala-isl-utils/libs  python -m scalene train_dl.py --out_dir /home/s2136718/tmp/out_bias_select_dep --with_polyenv --stop_at 1 -with_polyenv_sampling_bias bias_select_dep >> /tmp/procs_out/poly_select_dep.out 2>&1 &
#endtime = time.time()

#print("time taken: ", endtime - starttime)

### Basic RL with debug pudb
#POLYITEDIR=/home/s2136718/MLPC LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/s2136718/MLPC/scala-isl-utils/libs  python -m pudb train_random.py --out_dir /home/s2136718/tmp/out_bias_select_dep --with_polyenv --stop_at 5 --dep_rep complex_full -with_polyenv_sampling_bias bias_select_dep

#wait

# Baselines
#POLYITEDIR=/home/s2136718/MLPC LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/s2136718/MLPC/scala-isl-utils/libs OMP_NUM_THREADS=6 PYTHONPATH="" taskset -c 0-5   python train_random.py --out_dir /home/s2136718/tmp/out --with_baselines >> /tmp/procs_out/0.out 2>&1 &
wait
