import sys
sys.path.append("../")
from ray.rllib.models import ModelCatalog
from agi.nl_holdem_env import NlHoldemEnvWrapper
from agi.nl_holdem_net import NlHoldemNet
ModelCatalog.register_custom_model('NlHoldemNet', NlHoldemNet)
import numpy as np
from tqdm import tqdm
import pandas as pd
from agi.evaluation_tools import NNAgent,death_match

#%%

conf = eval(open("../confs/nl_holdem.py").read().strip())

#%%

env = NlHoldemEnvWrapper(
        conf
)

#%%

i = 1048
nn_agent = NNAgent(env.observation_space,
                       env.action_space,
                       conf,
                       f"../weights/c_{i}.pkl",
                       f"oppo_c{i}")

#%%

for i in tqdm(range(10)):
    obs = env.reset()
    d = False
    while not d:
        action_ind = nn_agent.make_action(obs)
        obs,r,d,i = env.step(action_ind)
    #break

#%%

print(
    env.env.get_state(0)["raw_obs"]["hand"],\
    env.env.get_state(1)["raw_obs"]["hand"],\
    env.env.get_state(1)["raw_obs"]["public_cards"],\
    env.env.get_state(1)["action_record"]
    )


#%%

print(env.env.get_payoffs())

