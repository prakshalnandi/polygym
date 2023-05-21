import csv
from collections import namedtuple
import numpy as np
import os
import random
import sys
import pandas as pd

from absl import app
from absl import flags

import tqdm

import environment
import polygym
import schedule_eval

import copy
import pickle 
import time

from buffer import Buffer
from agents import QLearningAgent
from deep_agents import DQN

import pudb
import torch
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
    "epsilon": 0.55555,
}

CONFIG= {
    "env": "CartPole-v1",
    "episode_length": 200,
    "max_timesteps": 20000,
    "max_time": 30 * 60,
    "eval_freq": 1000, # HOW OFTEN WE EVALUATE (AND RENDER IF RENDER=TRUE)
    "eval_episodes": 5,
    "learning_rate": 1e-4,
    "hidden_size": (512,256),
    "target_update_freq": 500,
    "batch_size": 32,
    "gamma": 0.99,
    "buffer_capacity": int(1e6),
    "plot_loss": False, # SET TRUE FOR 3.3 +
    # (Understanding the Loss)
    "save_filename": None,
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


###@profile
def gen_and_bench_random_schedule(env, sample_name, agent, agent_ex, deep_agent,  sampling_bias=None, predef_actions=None, blnTraining=True, iDep_rep='simple'):
    with_ast_and_map = True if sample_name in ['gemm', 'matvect'] else False
    state = env.reset(sample_name, with_repr = True, with_ast_and_map=with_ast_and_map, dep_rep=iDep_rep)

    actions = []
    done = False
    reward = None

    isExplore = False

    exec_time = None
    memory = Buffer(int(1e4))
    memory_ex = Buffer(int(1e4))

    try:
        if predef_actions:
            predef_actions_idx = 0
        while not done:
            if predef_actions:
                action_idx = predef_actions[predef_actions_idx]
                predef_actions_idx += 1
            elif sampling_bias:
                mask = state['action_mask']
                possibilities = mask * range(len(mask))
                if sampling_bias == 'bias_coeff0':
                    p = mask * [1, 1, 1, 1, 0.15, 0.15]
                elif sampling_bias == 'bias_select_dep':
                    p = mask * [0.2, 0.2, 0.6, 1, 1, 1]
                else:
                    raise Exception
                p /= p.sum()        # Normalize
                action_idx = int(np.random.choice(possibilities, p=p))
            else:
                action_idx = np.random.choice(np.nonzero(state['action_mask'])[0])
            if env.status == Status.construct_space:

                actList = [index for index,value in enumerate(mask) if value == 1]
                #print("actList: ", actList)
                action_idx = agent.choose_action(tuple(np.array(state['observation'])), actList, blnTraining)
                deep_act = deep_agent.act(np.array(state['observation'], dtype=np.int32), actList, blnTraining)
                #print("deep_act: ", deep_act)
                #print("action taken: ", action_idx)
                action_idx = deep_act

            else:
                if(isExplore == False):
                    state['observation'] = [None] * 2
                    isExplore = True
                actList = [index for index,value in enumerate(mask) if value == 1]
                action_idx = agent_ex.choose_action(tuple(np.array(state['observation'],dtype=object)), actList, blnTraining)
                #deep_act = deep_agent.act(np.array(state['observation']), actList, blnTraining)
            
            action = list(environment.Action)[action_idx]
            actions.append(action_idx)      
            #prev_obs = state['observation']
            nstate, reward, done, info = env.step(action, True, dep_rep=iDep_rep)
            
            if env.status == Status.construct_space:
            #if action_idx in [0,1,2]:
                #print("adding state to memory: ", np.array(state['observation']))
                memory.add(np.array(state['observation'], dtype=np.int32),
                    np.array([action_idx]),
                    np.array(nstate['observation'], dtype=np.int32),
                    np.array([done]),
                )
            else:
                #print("explore state: ",  nstate['observation'])
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

        to_process = list(polygym.polybench_invocations.keys())

        blnTesting = False

        print("initializing agent")

        intParam = 4
        if(FLAGS.dep_rep == 'complex_coeff'):
            intParam = 211
        elif (FLAGS.dep_rep == 'complex_full'):
            intParam = 291
        else:
            intParam = 4 


        agent = QLearningAgent(
                    action_space=env.action_space,
                    gamma=PARAMETERS["gamma"],
                    alpha=PARAMETERS["alpha"],
                    epsilon=PARAMETERS["epsilon"],
                    )

        agent_ex = QLearningAgent(
                    action_space=env.action_space,
                    gamma=PARAMETERS["gamma"],
                    alpha=PARAMETERS["alpha"],
                    epsilon=PARAMETERS["epsilon"],
                    )
        """
        deep_agent = DQN(
                    action_space=np.concatenate([np.ones(3), np.zeros(3)]), observation_space=np.array([None] * 4), **CONFIG
                    )
        
        deep_agent = DQN(
                    action_space=np.ones(3), observation_space=np.array([None] * 4), **CONFIG
                    )
        """
        deep_agent = DQN(
                    action_space=np.ones(3), observation_space=np.full((intParam),-1), **CONFIG
                    )
        #i = 0
        #while True:
        for i in range(FLAGS.stop_at):
            print('to_process: ' + str(to_process))
            print('len(to_process): ' + str(len(to_process)))

            for sample_name in tqdm.tqdm(to_process):

                if (i > FLAGS.stop_at - 2):
                    sample_name = 'mvt'
                csv_filename = os.path.join(FLAGS.out_dir, sample_name + '.csv')

                # Remove sample from eval if its evaluated enough already
                if FLAGS.stop_at and os.path.isfile(csv_filename) and i % 1 == 0:
                    df = pd.read_csv(csv_filename, sep="\t")
                    num_ok = len(df[df['status'] == 'ok'])
                    print('sample_name: %s, num_ok: %i' % (sample_name, num_ok))
                    if num_ok > FLAGS.stop_at:
                        to_process.remove(sample_name)
                        print('removed element: ' + sample_name)
                        continue
                #agent = QLearningAgent(
                #    action_space=env.action_space,
                #    gamma=PARAMETERS["gamma"],
                #    alpha=PARAMETERS["alpha"],
                #    epsilon=PARAMETERS["epsilon"],
                #    )
                #if (sample_name == 'mvt'):
                if (i > FLAGS.stop_at - 4):
                    #if (i > FLAGS.stop_at):
                    #pudb.set_trace()
                    
                    blnTesting = True
                    print("Testing")
                    reward, status, actions, ast, isl_map, memory, memory_ex = gen_and_bench_random_schedule(env, sample_name, agent, agent_ex, deep_agent, FLAGS.with_polyenv_sampling_bias, blnTraining=False, iDep_rep=FLAGS.dep_rep)
                else:
                    # Bench
                    blnTesting = False
                    reward, status, actions, ast, isl_map, memory, memory_ex = gen_and_bench_random_schedule(env, sample_name, agent, agent_ex, deep_agent, FLAGS.with_polyenv_sampling_bias, iDep_rep=FLAGS.dep_rep)

                speedup = env.reward_to_speedup(reward)
                #print('speedup :' + str(speedup))
                if blnTesting:
                    print('Testing speedup :' + str(speedup))
                else:
                    print('speedup :' + str(speedup))
                exec_time = env.speedup_to_execution_time(speedup)
                print('exectime :' + str(exec_time))

                #print("previous Q Table")
                #print(agent.q_table)
                if(blnTesting == False):
                    for io in range(memory.entries):
                        if (memory.records.actions[io][0] in [0,1,2]):
                            agent.update_q_table(tuple(memory.records.states[io]),
                                    memory.records.actions[io][0],
                                    reward,
                                    tuple(memory.records.next_states[io]),
                                    memory.records.done[io][0])
                    #print("Updated Q table")
                    #print(agent.q_table)

                    for io2 in range(memory_ex.entries):
                        if (memory_ex.records.actions[io2][0] in [3,4,5]):
                            agent_ex.update_q_table(tuple(memory_ex.records.states[io2]),
                                    memory_ex.records.actions[io2][0],
                                    reward,
                                    tuple(memory_ex.records.next_states[io2]),
                                    memory_ex.records.done[io2][0])

                    loss = deep_agent.update(memory.records, reward)["q_loss"]
                    print("loss: ", loss)
                #i Save result
                create_csv_if_not_exists(csv_filename, ['method', 'execution_time', 'status', 'actions', 'ast', 'isl_map'])
                append_to_csv(csv_filename, ['PolyEnv-random', exec_time, status, str(actions), str(ast), str(isl_map)])

            #i += 1

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
