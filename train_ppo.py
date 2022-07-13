from ray import tune
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from gym_bb_attack.envs.attack_env import *
import sys
import os
from custom_callbacks import *

# So that np prints arrays completely, making it easier to debug
np.set_printoptions(threshold=sys.maxsize)

# Disables GPU learning for now
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def env_creator(env_config):
    return Attack_Env(env_config)


ray.init(address='auto')  # Uncomment on master node
#ray.init()  # Comment on master node

register_env('bb_attack-v0', env_creator)

tune.run(PPOTrainer, name="bb_attack",
                            config={"env": "bb_attack-v0",
                            "env_config": {"test": False,
                                           'train_class': tune.grid_search([i for i in range(10)])},
                             "num_gpus": 0,
                            "render_env": False,
                             "kl_coeff": 0.4,
                             "num_workers": 3, #number of parallel environments = total available CPU cores - 1
                             "framework": "torch",
                             "lr": 0.003,  # Initial learning rate
                             "lr_schedule": [[0, 0.003],  # Learning rate schedule to gradually change LR.
                                             [70000000, 0.001],  # Will change LR at timesteps in column 1 to column 2
                                             [100000000, 0.0004]],
                             "num_sgd_iter": 5,  # PPO hyperparameter
                             "train_batch_size": 1024,  # PPO hyperparameter
                             "sgd_minibatch_size": 32,  # PPO hyperparameter
                             "lambda": 0.95,  # PPO hyperparameter
                             # "vf_clip_param": 40.0,
                             # "vf_loss_coeff": 0.05,
                             # "exploration_config": {
                             #     "type": "PerWorkerEpsilonGreedy",
                             #     "initial_epsilon": 1.0,
                             #     "final_epsilon": 0.1,
                             #     "epsilon_timesteps": 28000000,  # Timesteps over which to anneal epsilon.
                             #
                             # },

                            # "evaluation_interval": 1,
                            # "evaluation_num_episodes": 100,
                            # "evaluation_config": {
                            #     "env_config": {"test": True},
                            #          "explore": False
                            #      },

                             'model': {
                                 'fcnet_hiddens': [256, 128, 64]
                             },
                             'callbacks' : attack_stats,
                             },
         stop={"timesteps_total": 500000}, checkpoint_freq=10000, checkpoint_at_end=True)
