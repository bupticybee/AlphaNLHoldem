import os
import numpy as np
import ray
import pandas as pd
import pickle

def pfsp(win_rates, weighting="squared"):
    win_rates = [min(i,0.95) for i in win_rates]
    weightings = {
        "variance": lambda x: x * (1 - x),
        "linear": lambda x: 1 - x,
        "linear_capped": lambda x: np.minimum(0.5, 1 - x),
        "squared": lambda x: (1 - x) ** 2,
    }
    fn = weightings[weighting]
    probs = fn(np.asarray(win_rates))
    norm = probs.sum()
    if norm < 1e-10:
        return np.ones_like(win_rates) / len(win_rates)
    return probs / norm

def kbsp(win_rates, k=5):
    sorted_wr = sorted(win_rates)
    
    if len(sorted_wr) < k:
        baseline_val = sorted_wr[-1]
    else:
        baseline_val = sorted_wr[k-1]
        
    probs = np.asarray(np.asarray(win_rates) <= baseline_val,np.float)
    norm = probs.sum()
    if norm == 0:
        probs = np.ones_like(win_rates)
        norm = probs.sum()
    else:
        probs = np.asarray(np.asarray(win_rates) <= baseline_val,np.float)
        norm_rest = float(probs.sum()) * 0.15
        
        z_cnt = 0
        for i in range(len(probs)):
            if probs[i] == 0:
                z_cnt += 1
        
        for i in range(len(probs)):
            if probs[i] == 0:
                probs[i] = norm_rest / float(z_cnt)
        norm = probs.sum()
        #print("distribute:",probs,probs / norm,norm_rest)
    
    return probs / norm

class WinrateTracker():
    def __init__(self,nmin=500,nmax=500):
        self.n = 0
        self.v = 0
        
        self.nmin = nmin
        self.nmax = nmax
    
    def update(self,v):
        self.n += 1
        self.clp_n = np.clip(self.n,self.nmin,self.nmax)
        self.v = self.v * (self.clp_n - 1) / self.clp_n + v / self.clp_n
        
@ray.remote
class League():
    def __init__(self,initial_weight=None,n=500,last_num=1000,kbest=5,output_dir=None):
        self.weights_dic = {}
        self.current_pid = -1
        self.pids = []
        self.winrates = None
        self.n = n
        self.last_num = last_num
        self.output_dir = output_dir
        self.kbest = kbest
        if initial_weight is not None:
            self.add_weight(initial_weight)

    def get_all_weights_dic(self):
        return self.weights_dic
    
    def get_all_policy_ids(self):
        pids = self.pids
        return pids
    
    def get_latest_policy_id(self):
        pid = self.pids[-1]
        return pid
    
    def get_weight(self,policy_id):
        weight = self.weights_dic[policy_id]
        return weight
    
    def select_opponent(self):
        probs = kbsp([i.v for i in self.winrates[-self.last_num:]],k=self.kbest)
        policy_id = np.random.choice(self.pids[-self.last_num:],p=probs)
        weight = self.get_weight(policy_id)
        return policy_id,weight
    
    def initized(self):
        return len(self.pids) > 0
    
    def initize_if_possible(self,new_weight):
        if not self.initized():
            self.add_weight(new_weight)
        
    def add_weight(self,new_weight):
        self.current_pid += 1
        self.pids.append(self.current_pid)
        self.weights_dic[self.current_pid] = new_weight
        n = self.n
        if self.winrates is None:
            self.winrates = [WinrateTracker(n,n) for i in self.pids]
        else:
            old_winrates = self.winrates
            self.winrates = [WinrateTracker(n,n) for i in self.pids]
            for i in range(min(len(self.winrates),len(old_winrates))):
                self.winrates[i].v = old_winrates[i].v
        self.selfplay_winrate = WinrateTracker(n,n)
        
        
        output_dir = self.output_dir
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if not os.path.exists(os.path.join(output_dir,"weights")):
                os.makedirs(os.path.join(output_dir,"weights"))
            fname = os.path.join(output_dir, 'weights', 'c_{}.pkl'.format(self.current_pid))
            with open(fname, 'wb') as whdl:
                pickle.dump(new_weight, whdl)


    def set_winrates(self,winrates):
        assert(len(winrates) == len(self.winrates))
        for i in range(len(winrates)):
            wr = winrates[i]
            self.winrates[i].v = wr
    
    def update_result(self,policy_id,result,selfplay=False):
        if selfplay:
            self.selfplay_winrate.update(result)
        else:
            self.winrates[policy_id].update(result)
            
    def winrate_all_match(self,winrate):
        return np.all([i.v > winrate for i in self.winrates[-self.last_num:]])
    
    def get_statics_table(self,dump=True):
        names = ["self-play",] + ["c_" + str(i) for i in self.pids]
        winrates = [self.selfplay_winrate.v,] + [i.v for i in self.winrates]
        nums = [self.selfplay_winrate.n,] + [i.n for i in self.winrates]
        table = pd.DataFrame({
            "oppo": names,
            "winrate": winrates,
            "matches": nums,
        }).T
        
        output_dir = self.output_dir
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            table.to_csv(os.path.join(output_dir,"winrates.csv"),header=False,index=False)
            
        return table