# define dataset class to feed the model
import numpy as np 
import os
import cv2
import sys
import time
import pandas as pd
import pickle
from ray.tune.registry import register_trainable

class ProgressBar():
    def __init__(self,worksum,info="",auto_display=True):
        self.worksum = worksum
        self.info = info
        self.finishsum = 0
        self.auto_display = auto_display
    def startjob(self):
        self.begin_time = time.time()
    def complete(self,num):
        self.gaptime = time.time() - self.begin_time
        self.finishsum += num
        if self.auto_display == True:
            self.display_progress_bar()
    def display_progress_bar(self):
        percent = self.finishsum * 100 / self.worksum
        eta_time = self.gaptime * 100 / (percent + 0.001) - self.gaptime
        strprogress = "[" + "=" * int(percent // 2) + ">" + "-" * int(50 - percent // 2) + "]"
        str_log = ("%s %.2f %% %s %s/%s \t used:%ds eta:%d s" % (self.info,percent,strprogress,self.finishsum,self.worksum,self.gaptime,eta_time))
        sys.stdout.write('\r' + str_log)

def ma_sample(spaces):
    retval = {}
    for k,v in spaces.items():
        retval[k] = v.sample()
    return retval

def get_winrate_and_weight(logdir,league):
    wr_path = os.path.join(logdir,'winrates.csv')
    weight_path = os.path.join(logdir,'weights')
    
    wr = pd.read_csv(wr_path)
    winrates = wr.values[0][1:]
    
    weights = os.listdir(weight_path)
    weights = [i for i in weights if i.split('.')[-1] == 'pkl']
    
    minlen = min(len(weights),len(winrates))
    
    winrates = winrates[-minlen:]
    weights = weights[-minlen:]
    
    weights = sorted(weights,key=lambda x:int(x.split('.')[0].split("_")[-1]))
    
    assert(len(weights) == len(winrates))
    
    weights = [pickle.load(open(os.path.join(weight_path,i), "rb")) for i in weights]
    
    for weight in weights:
        league.add_weight.remote(weight)
    league.set_winrates.remote(winrates)
    
def register_restore_weight_trainer(weight):
    pweight = {}
    for k,v in weight.items():
        k = k.replace("oppo_policy","default_policy")
        pweight[k] = v

    from ray.rllib.agents.impala.impala import build_trainer,DEFAULT_CONFIG,VTraceTFPolicy
    from ray.rllib.agents.impala.impala import validate_config,choose_policy,make_aggregators_and_optimizer
    from ray.rllib.agents.impala.impala import OverrideDefaultResourceRequest
    
    def my_defer_make_workers(trainer, env_creator, policy, config):
        def load_history(worker):
            for p, policy in worker.policy_map.items():
                print("loading weights" + "|" * 100)
                policy.set_weights(pweight)
        
        # Defer worker creation to after the optimizer has been created.
        workers = trainer._make_workers(env_creator, policy, config, 0)
        print("inside my defer make workers")
        
        workers.local_worker().apply(load_history)
        for one_worker in workers.remote_workers():
            one_worker.apply(load_history)
        return workers

    MyImpalaTrainer = build_trainer(
        name="IMPALA",
        default_config=DEFAULT_CONFIG,
        default_policy=VTraceTFPolicy,
        validate_config=validate_config,
        get_policy_class=choose_policy,
        make_workers=my_defer_make_workers,
        make_policy_optimizer=make_aggregators_and_optimizer,
        mixins=[OverrideDefaultResourceRequest])
    
    register_trainable("IMPALA", MyImpalaTrainer)

