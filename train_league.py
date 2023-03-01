import ray
import gym
import logging
import argparse
from matplotlib import pyplot as plt
from utils import ProgressBar,ma_sample,get_winrate_and_weight,register_restore_weight_trainer
#from custom_model import CustomFullyConnectedNetwork,KerasBatchNormModel,BatchNormModel,OriginalNetwork
logger = logging.getLogger()
logger.setLevel(logging.INFO)
from ray.rllib import agents
import numpy as np
import pickle
from ray.rllib.utils import try_import_tf
import os
import pandas as pd
tf = try_import_tf()

from ray.rllib.agents.impala.vtrace_policy import VTraceTFPolicy
from ray.rllib.agents.impala.impala import DEFAULT_CONFIG
from ray.rllib.env.atari_wrappers import is_atari

from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray import tune

from agi.nl_holdem_env import NlHoldemEnvWithOpponent
from agi.nl_holdem_net import NlHoldemNet
from agi.nl_holdem_lg_net import NlHoldemLgNet

ModelCatalog.register_custom_model('NlHoldemNet', NlHoldemNet)
ModelCatalog.register_custom_model('NlHoldemLgNet', NlHoldemLgNet)

from agi.league import League
from ray.rllib.agents.impala.impala import ImpalaTrainer
from agi.mis import init_cluster_ray

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

parser = argparse.ArgumentParser()
parser.add_argument('--conf',  type=str)
parser.add_argument('--gap',  type=int, default=1000)
parser.add_argument('--sp',  type=float, default=0.0)
parser.add_argument('--exg_oppo_prob',  type=float, default=0.01)
parser.add_argument('--upwin',  type=float, default=1)
parser.add_argument('--kbest',  type=int, default=5)
parser.add_argument('--league_tracker_n',  type=float, default=10000)
parser.add_argument('--last_num',  type=int, default=100000)
parser.add_argument('--rwd_update_ratio',  type=float, default=1.0)
parser.add_argument('--restore',  type=str,default=None)
parser.add_argument('--output_dir',  type=str,default="league/history_agents")
parser.add_argument('--mode', type=str, default="local")
parser.add_argument('--experiment_name', default='run_trial_1', type=str)  # please change a new name
args = parser.parse_args()

if args.mode == "local":
    ray.init()
elif args.mode == "remote":
    init_cluster_ray(log_to_driver=False)
else:
    raise RuntimeError("unknown mode: {}".format(args.mode))


conf = eval(open(args.conf).read().strip())

register_env("NlHoldemEnvWithOpponent", lambda config: NlHoldemEnvWithOpponent(
        conf
))

league = League.remote(
    n=args.league_tracker_n,
    last_num=args.last_num,
    kbest=args.kbest,
    output_dir=args.output_dir,
)

def get_train(weight):
    if weight is None:
        pweight = None
    else:
        pweight = {}
        for k,v in weight.items():
            k = k.replace("oppo_policy","default_policy")
            pweight[k] = v
            
    def train_fn_load(config, reporter):
        agent = ImpalaTrainer(config=config)
        print("LOAD: after init, before load")

        if pweight is not None:
            agent.workers.local_worker().get_policy().set_weights(pweight)
            agent.workers.sync_weights()

        print("LOAD: before train, after load")
        while True:
            result = agent.train()
            reporter(**result)
        agent.stop()

    return train_fn_load

if args.restore is not None:
    get_winrate_and_weight(args.restore,league)
    pid = ray.get(league.get_latest_policy_id.remote())
    print("latest pid: {}".format(pid))
    weight = ray.get(league.get_weight.remote(pid))
    #register_restore_weight_trainer(weight)
    train_func = get_train(weight)
else:
    train_func = get_train(None)

@static_vars(league=league)
def on_episode_end(info):
    envs = info["env"]
    policies = info['policy']
    default_policy = policies["default_policy"]
    
    for env in envs.vector_env.envs:
        if env.is_done:
            # 1. 更新结果到league
            last_reward = env.last_reward
            pid = env.oppo_name
            
            if np.random.random() < args.rwd_update_ratio:
                if pid == "self":
                    ray.get(on_episode_end.league.update_result.remote(None,last_reward,selfplay=True))
                else:
                    ray.get(on_episode_end.league.update_result.remote(pid,last_reward,selfplay=False))

            # 2. 更新对手权重
            
            # 以0.2的概率self play
            if np.random.random() < args.exg_oppo_prob:
                if np.random.random() < args.sp:
                    p_weights = default_policy.get_weights()
                    weight = {}
                    for k,v in p_weights.items():
                        k = k.replace("default_policy","oppo_policy")
                        weight[k] = v
                    env.oppo_name = "self"
                    env.oppo_policy.set_weights(weight)
                else:
                    pid,weight = ray.get(on_episode_end.league.select_opponent.remote())
                    env.oppo_name = pid
                    env.oppo_policy.set_weights(weight)
            
@static_vars(league=league)
def on_episode_start(info):
    envs = info["env"]
    policies = info['policy']
    default_policy = policies["default_policy"]
    
    # 如果league 没有第一个权重，那么使用当前policy中的权重当作第一个
    if not ray.get(on_episode_start.league.initized.remote()):
        p_weights = default_policy.get_weights()
        weight = {}
        for k,v in p_weights.items():
            k = k.replace("default_policy","oppo_policy")
            weight[k] = v
        ray.get(on_episode_start.league.initize_if_possible.remote(weight))
        
    for env in envs.vector_env.envs:
        if env.oppo_name is None:
            pid,weight = ray.get(on_episode_start.league.select_opponent.remote())
            env.oppo_name = pid
            env.oppo_policy.set_weights(weight)

@static_vars(league=league)
def on_episode_step(info):
    pass

@static_vars(league=league,count=0)
def on_train_result(info):
    winrates_pd = ray.get(on_train_result.league.get_statics_table.remote())
    winrates_pd.to_csv("winrates.csv",header=False,index=False)
    
    table_t = winrates_pd.T
    table_t["mbb/h"] = np.asarray(table_t["winrate"] / 2.0 * 1000.0,np.int)
    info['result']['winrates'] = table_t.T
    on_train_result.count += 1
    
    gap = args.gap
    if ray.get(on_train_result.league.winrate_all_match.remote(args.upwin))  \
         or on_train_result.count % gap == gap - 1:
        trainer = info["trainer"]
        p_weights = trainer.get_weights()["default_policy"]
        weight = {}
        for k,v in p_weights.items():
            k = k.replace("default_policy","oppo_policy")
            weight[k] = v 
        ray.get(on_train_result.league.add_weight.remote(weight))
        if not os.path.exists("weights"):
            os.makedirs("weights")
        with open('output_weight.pkl','wb') as whdl:
            pickle.dump(weight,whdl)
        with open('weights/output_weight_{}.pkl'.format(on_train_result.count),'wb') as whdl:
            pickle.dump(weight,whdl)

tune_config = {
    'max_sample_requests_in_flight_per_worker': 1,
    'num_data_loader_buffers': 4,
    "callbacks": {
        "on_episode_end": on_episode_end,
        "on_episode_start": on_episode_start,
        "on_episode_step": on_episode_step,
        "on_train_result": on_train_result,
    },
}

tune_config.update(conf)

tune.run(
    train_func,
    config=tune_config,
    stop={
        'timesteps_total': 10000000000,
    },
    local_dir='log/',
    #resources_per_trial=ImpalaTrainer.default_resource_request,
    #resources_per_trial=ImpalaTrainer.default_resource_request(tune_config),
    resources_per_trial={'cpu':1,'gpu':1},
)