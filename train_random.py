import csv
from collections import namedtuple
import numpy as np
import os
import random
import sys
import pandas as pd

from absl import app
from absl import flags

#import tqdm

import environment
import polygym
import schedule_eval

import copy
import pickle 
import time

from buffer import Buffer
from agents import QLearningAgent, QLApproxAgent

import pudb
from environment import Status

flags.DEFINE_string('out_dir', '', 'Root dir to store the results.')
flags.DEFINE_boolean('with_baselines', False, 'Benchmark baselines.')
flags.DEFINE_boolean('with_isl_tuning', False, 'Benchmark isl and tune.')
flags.DEFINE_boolean('with_polyenv', False, 'Benchmark polyenv random walk.')
flags.DEFINE_string('with_polyenv_sampling_bias', None, 'A sampling bias.')

flags.DEFINE_integer('stop_at', None, 'Number of OK samples to stop at.')

flags.DEFINE_string('with_action_import_sample_name', '', 'The sample name of the action sequence to import.')
flags.DEFINE_string('with_action_import_actions', '', 'The action sequence to import.')

flags.DEFINE_string('dep_rep', '', 'The type of state representation.') #simple #complex_coeff

FLAGS = flags.FLAGS

PARAMETERS = {
    "gamma": 0.95,
    "alpha": 0.1,
    "epsilon": 0.9,
}


def create_csv_if_not_exists(filename, fieldnames):
    if not os.path.isfile(filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()


def append_to_csv(filename, row):
    with open(filename, 'a') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(row)


def gen_and_bench_random_schedule(env, sample_name, agent, agent_ex,  sampling_bias=None, predef_actions=None, blnTraining=True, iDep_rep='simple'):
    with_ast_and_map = True if sample_name in ['gemm', 'matvect'] else False
    state = env.reset(sample_name, with_repr = True, with_ast_and_map=with_ast_and_map, dep_rep=iDep_rep)

    actions = []
    done = False
    reward = None

    isExplore = False

    exec_time = None
    memory = Buffer(int(1e4))
    memory_ex = Buffer(int(1e4))
    #agent = QLearningAgent(
    #    action_space=env.action_space,
    #    obs_space=state,
    #    gamma=PARAMETERS["gamma"],
    #    alpha=PARAMETERS["alpha"],
    #    epsilon=PARAMETERS["epsilon"],
    #)

    try:
        mask = state['action_mask']
        if predef_actions:
            predef_actions_idx = 0
        while not done and len(mask) != 0:
            mask = state['action_mask']
            if env.status == Status.construct_space:

                actList = [index for index,value in enumerate(mask) if value == 1]
                #print("mask", mask)
                #print("actList", actList)
                if(len(actList) == 0):
                    action_idx = 2
                else:
                    action_idx = agent.choose_action(tuple(np.array(state['observation'],dtype=object)), actList, blnTraining, isExplore)
                #action_idx = agent.choose_action(tuple(np.array(state['observation'],dtype=object)), actList, blnTraining, isExplore)
                #print("agent_act_const", action_idx)

            else:
                if(isExplore == False):
                    state['observation'] = [None] * 53
                    #state['observation'] = [0] * 36
                    isExplore = True
                actList = [index for index,value in enumerate(mask) if value == 1]
                #print("mask", mask)
                #print("actList", actList
                if not actList:
                    print("empty actList: ", actList)
                    print("empty mask: ", mask)
                    action_idx = 3
                else:
                    #action_idx = agent_ex.choose_action(tuple(np.array(state['observation'],dtype=object)), actList, blnTraining, isExplore)
                    #print("action_idx", action_idx)
                    action_idx = 3
                #print("agent_act_explore:", action_idx)
            
            action = list(environment.Action)[action_idx]
            actions.append(action_idx)      
            #prev_obs = state['observation']
            nstate, reward, done, info = env.step(action, True, dep_rep=iDep_rep)
            #print("status after action : ", env.status)
            if blnTraining:
                if env.status == Status.construct_space:
                #if action_idx in [0,1,2]:
                    #print("adding state to memory: ", np.array(state['observation']))
                    memory.add(np.array(state['observation'],dtype=object),
                        np.array([action_idx]),
                        np.array(nstate['observation'],dtype=object),
                        np.array([done]),
                        )   
                else:

                    if isExplore:
                        memory_ex.add(np.array(state['observation'],dtype=object),
                            np.array([action_idx]),
                            np.array(nstate['observation'],dtype=object),
                            np.array([done]),
                        )         
                #state = copy.deepcopy(nstate)
            #state = copy.deepcopy(nstate)
            state = pickle.loads(pickle.dumps(nstate))
            #state = dict((k,v) for (k,v) in nstate.items())


        #speedup = env.reward_to_speedup(reward)
        #print('speedup :' + str(speedup))
        #exec_time = env.speedup_to_execution_time(speedup)
        #print('exectime :' + str(exec_time))
        status = info['status']
        ast = info['ast'] if 'ast' in info else None
        isl_map = info['isl_map'] if 'isl_map' in info else None

    except (
            polygym.ChernikovaTimeoutException, schedule_eval.LLVMTimeoutException,
            schedule_eval.InvalidScheduleException,
            schedule_eval.LLVMInternalException, schedule_eval.ScheduleTreeNotLoadedException,
            schedule_eval.OutputValidationException) as e:
        status = e.__class__.__name__

    return reward, status, actions, ast if 'ast' in locals() else None, isl_map if 'isl_map' in locals() else None, memory, memory_ex 


def bench(invocation, optimization=None, clang_exe=None, additional_params=None):
    sys.argv = invocation
    #pudb.set_trace()
    print("Inside bench")
    _, compilation_params, config, scop_file, jsonp, scop = polygym.parse_args()

    if clang_exe:
        config.clang_benchmark_exe = clang_exe

    if optimization == 'O3':
        execution_time = schedule_eval.benchmark_schedule(compilation_params,
                                                          config, scop_file,
                                                          num_iterations=1,
                                                          compilation_timeout=100)
    elif optimization == 'ISL':
        execution_time = schedule_eval.benchmark_schedule(compilation_params,
                                                          config, scop_file,
                                                          with_polly=True,
                                                          num_iterations=1,
                                                          compilation_timeout=100,
                                                          additional_params=additional_params)

    return execution_time, config.clang_benchmark_exe


def main(argv):
    start = time.time()
    if FLAGS.with_baselines:
        clang_exes = [
                None,
                '/home/s2136718/MLPC/llvm_root/llvm_build/bin/clang',
                #'clang-10',
                #'/tmp/llvm-10/build/bin/clang',
                #'/tmp/llvm-12/build/bin/clang',
#                '/tmp/llvm-master/build/bin/clang',
                ]
        optimizations = ['O3', 'ISL']
        for sample_name, invocation in polygym.polybench_invocations.items():
            csv_filename = os.path.join(FLAGS.out_dir, sample_name + '_baselines.csv')
            create_csv_if_not_exists(csv_filename, ['method', 'compiler', 'execution_time'])

            for optimization in optimizations:
                for clang_exe in clang_exes:
                    # Bench
                    exec_time, clang_exe_used = bench(invocation, optimization=optimization, clang_exe=clang_exe)

                    # Save result
                    append_to_csv(csv_filename, [optimization, clang_exe_used, exec_time])
    if FLAGS.with_isl_tuning:
        clang_exe = '/home/s2136718/MLPC/llvm_root/llvm_build/bin/clang'

        isl_options = {
                'polly-opt-optimize-only': ['all', 'raw'],
                'polly-opt-fusion': ['min', 'max'],
                'polly-opt-max-constant-term': [-1] + list(range(500)),
                'polly-opt-max-coefficient': [-1] + list(range(500)),
                'polly-opt-fusion': ['min', 'max'],
                'polly-opt-maximize-bands': ['yes', 'no'],
                'polly-opt-outer-coincidence': ['yes', 'no'],
                }
        polly_options = {
                'polly-prevect-width': [2**x for x in range(1,6)],
                'polly-target-latency-vector-fma': [2**x for x in range(1,6)],
                'polly-target-throughput-vector-fma': [2**x for x in range(1,4)],
                'polly-default-tile-size': [2**x for x in range(1,7)],
                'polly-register-tiling': [True, False],
                'polly-pattern-matching-based-opts': [True, False]
                }

        for i in range(FLAGS.stop_at):
            for sample_name, invocation in polygym.polybench_invocations.items():
                csv_filename = os.path.join(FLAGS.out_dir, sample_name + '_isl_baselines.csv')
                create_csv_if_not_exists(csv_filename, ['method', 'execution_time', 'additional_params'])

                additional_params = []

                for option_name, option_vals in isl_options.items():
                    if random.random() < 0.5:
                        continue

                    option_val = random.choice(option_vals)

                    param = '-' + option_name
                    if type(option_val) is not bool:
                        param += '=' + str(option_val)
                    additional_params.append(param)
                
                # Bench
                try:
                    (exec_time, exception), clang_exe_used = bench(invocation, optimization='ISL', clang_exe=clang_exe, additional_params=additional_params)
                except Exception as e:
                    print(e)
                    continue

                # Save result
                c1.onfig = ''
                append_to_csv(csv_filename, ['ISL', exec_time, str(additional_params)])

    if FLAGS.with_polyenv:
        env_config = {'invocations': polygym.polybench_invocations}
        env = environment.PolyEnv(env_config)
        blnTesting = False
        to_process = list(polygym.polybench_invocations.keys())
        #test_process = copy.deepcopy(to_process[-1])
        #print("to process: ", to_process)
        #print("testing sample: ", to_process[-1])
        #agent = QLApproxAgent(
        #            action_space=env.action_space,
        #            gamma=PARAMETERS["gamma"],
        #            alpha=PARAMETERS["alpha"],
        #            epsilon=PARAMETERS["epsilon"],
        #            num_weights=291,
        #            )

        #agent_ex = QLApproxAgent(
        #            action_space=env.action_space,
        #            gamma=PARAMETERS["gamma"],
        #            alpha=PARAMETERS["alpha"],
        #            epsilon=PARAMETERS["epsilon"],
        #            num_weights=36,
        #            )

        #print("Initializing Agents")
        #i = 0
        #while True:
        #for sample_name in tqdm.tqdm(to_process[:-1]):
        for idp in range(len(to_process)):
            train_process = copy.deepcopy(to_process)
            test_process = to_process[idp]
            
            print("Initializing Agents")
            
            agent = QLApproxAgent(
                    action_space=env.action_space,
                    gamma=PARAMETERS["gamma"],
                    alpha=PARAMETERS["alpha"],
                    epsilon=PARAMETERS["epsilon"],
                    num_weights=291,
                    )

            agent_ex = QLApproxAgent(
                    action_space=env.action_space,
                    gamma=PARAMETERS["gamma"],
                    alpha=PARAMETERS["alpha"],
                    epsilon=PARAMETERS["epsilon"],
                    num_weights=36,
                    )
            for i in range(FLAGS.stop_at):
            #for i in range(FLAGS.stop_at):
                #print('to_process: ' + str(to_process))
                #print('len(to_process): ' + str(len(to_process)))
                agent.schedule_hyperparameters(i, FLAGS.stop_at)
                agent_ex.schedule_hyperparameters(i, FLAGS.stop_at)
                #for i in range(FLAGS.stop_at):
                #for sample_name in tqdm.tqdm(train_process):
                train_process_it = copy.deepcopy(to_process)
                for j in range(len(train_process_it)):
                    sample_name = random.choice(train_process_it)
                    if(sample_name != test_process):
                    #for sample_name in tqdm.tqdm(to_process):
                        #agent.schedule_hyperparameters(i, FLAGS.stop_at)
                        csv_filename = os.path.join(FLAGS.out_dir, sample_name + '.csv')

                        # Remove sample from eval if its evaluated enough already
                        if FLAGS.stop_at and os.path.isfile(csv_filename) and i % 1 == 0:
                            df = pd.read_csv(csv_filename, sep="\t")
                            num_ok = len(df[df['status'] == 'ok'])
                            print('sample_name: %s, num_ok: %i' % (sample_name, num_ok))
                            if num_ok > FLAGS.stop_at:
                                train_process.remove(sample_name)
                                print('removed element: ' + sample_name)
                                continue
                        # Bench
                        blnTesting = False
                        reward, status, actions, ast, isl_map, memory, memory_ex = gen_and_bench_random_schedule(env, sample_name, agent, agent_ex,  FLAGS.with_polyenv_sampling_bias, iDep_rep=FLAGS.dep_rep)

                        speedup = env.reward_to_speedup(reward)
                        #print('speedup :' + str(speedup))
                        print('speedup :' + str(speedup), " for sample name: ", sample_name)
                        exec_time = env.speedup_to_execution_time(speedup)
                        print('exectime :' + str(exec_time))
                        for io in range(memory.entries):
                            #print("memory record state : ", memory.records.states[io])                    
                            if (memory.records.actions[io][0] in [0,1,2]) and (blnTesting == False):
                                agent.update_weights(tuple(memory.records.states[io]),
                                        memory.records.actions[io][0],
                                        reward,
                                        tuple(memory.records.next_states[io]),
                                        memory.records.done[io][0])

                        #for io2 in range(memory_ex.entries):
                        #    if (memory_ex.records.actions[io2][0] in [3,4,5])and (blnTesting == False):
                        #        #print("memory_ex record state : ", memory_ex.records.states[io2])
                        #        agent_ex.update_weights(tuple(memory_ex.records.states[io2]),
                        #                memory_ex.records.actions[io2][0],
                        #                reward,
                        #                tuple(memory_ex.records.next_states[io2]),
                        #                memory_ex.records.done[io2][0])

                        create_csv_if_not_exists(csv_filename, ['method', 'execution_time', 'status', 'actions', 'ast', 'isl_map'])
                        append_to_csv(csv_filename, ['PolyEnv-random', exec_time, status, str(actions), str(ast), str(isl_map)])
                        train_process_it.remove(sample_name)

            sample_name = test_process
            total_speedup = 0
            blnTesting = True
            for i in range(FLAGS.stop_at):
            #for sample_name in tqdm.tqdm(to_process):
                agent.schedule_hyperparameters(i, FLAGS.stop_at)
                csv_filename = os.path.join(FLAGS.out_dir, sample_name + '.csv')

                # Remove sample from eval if its evaluated enough already
                if FLAGS.stop_at and os.path.isfile(csv_filename) and i % 1 == 0:
                    df = pd.read_csv(csv_filename, sep="\t")
                    num_ok = len(df[df['status'] == 'ok'])
                    print('sample_name: %s, num_ok: %i' % (sample_name, num_ok))
                    if num_ok > FLAGS.stop_at:
                        train_process.remove(sample_name)
                        print('removed element: ' + sample_name)
                        continue
                print("Testing ",sample_name)
                #blnTesting = True
                reward, status, actions, ast, isl_map, memory, memory_ex = gen_and_bench_random_schedule(env, sample_name, agent, agent_ex,  FLAGS.with_polyenv_sampling_bias, blnTraining=False, iDep_rep=FLAGS.dep_rep)

                speedup = env.reward_to_speedup(reward)
                #print('speedup :' + str(speedup))
                print('testing speedup :' + str(speedup), " for sample name: ", sample_name)
                exec_time = env.speedup_to_execution_time(speedup)
                print('exectime :' + str(exec_time))
                total_speedup += speedup

                create_csv_if_not_exists(csv_filename, ['method', 'execution_time', 'status', 'actions', 'ast', 'isl_map'])
                append_to_csv(csv_filename, ['PolyEnv-random', exec_time, status, str(actions), str(ast), str(isl_map)])

            average_speedup = total_speedup/FLAGS.stop_at
            print("Average speedup for : ", sample_name, "is ", average_speedup)
            csv_deletename = os.path.join(FLAGS.out_dir,  '*.csv')
            os.system('rm ' + csv_deletename)

    if FLAGS.with_action_import_sample_name and FLAGS.with_action_import_actions:
        env_config = {'invocations': polygym.polybench_invocations}
        env = environment.PolyEnv(env_config)

        sample_name = FLAGS.with_action_import_sample_name
        actions = eval(FLAGS.with_action_import_actions)
        exec_time, status, actions, ast, isl_map = gen_and_bench_random_schedule(env, sample_name, False, actions)

        print('isl_map: ' + isl_map)
        print('ast: '+ ast)
        print('exec_time: '+ exec_time)

    end = time.time()
    print("Time taken to run the script: ", end - start)

if __name__ == "__main__":
    app.run(main)
