from ray.rllib.utils import try_import_tf
from ray.rllib.agents.impala.vtrace_policy import VTraceTFPolicy
import pandas as pd
from ray.rllib.models import ModelCatalog
tf = try_import_tf()
from tqdm import tqdm

class NNAgent():
    def __init__(self,observation_space,action_space,policy_config,weights,variable_scope="oppo_policy"):
        self.oppo_preprocessor = ModelCatalog.get_preprocessor_for_space(observation_space, policy_config.get("model"))
        self.graph = tf.Graph()
        self.name = variable_scope
        with self.graph.as_default():
            with tf.variable_scope(variable_scope):
                self.oppo_policy = VTraceTFPolicy(
                    obs_space=self.oppo_preprocessor.observation_space,
                    action_space=action_space,
                    config=policy_config,
                )
        if weights is not None:
            import pickle
            with open(weights,'rb') as fhdl:
                weights = pickle.load(fhdl)
            new_weights = {}
            for k,v in weights.items():
                new_weights[k.replace("oppo_policy",variable_scope)] = v
            self.oppo_policy.set_weights(new_weights)
            
    def make_action(self,obs):
        observation = self.oppo_preprocessor.transform(obs)
        action_ind = self.oppo_policy.compute_actions([observation])[0][0]
        return action_ind
    
def death_match(agent1,agent2,env):
    rewards = []
    for i in tqdm(range(5000)):
        obs = env.reset()
        d = False
        while not d:
            legal_moves = obs["legal_moves"]
            #action_ind = np.random.choice(np.where(legal_moves)[0])
            if env.my_agent() == 0:
                action_ind = agent1.make_action(obs)
            elif env.my_agent() == 1:
                action_ind = agent2.make_action(obs)
            else:
                raise
            obs,r,d,i = env.step(action_ind)
        rewards.append(r[0])
    
    for i in tqdm(range(5000)):
        obs = env.reset()
        d = False
        while not d:
            legal_moves = obs["legal_moves"]
            #action_ind = np.random.choice(np.where(legal_moves)[0])
            if env.my_agent() == 0:
                action_ind = agent2.make_action(obs)
            elif env.my_agent() == 1:
                action_ind = agent1.make_action(obs)
            else:
                raise
            obs,r,d,i = env.step(action_ind)
        rewards.append(r[1])
    return rewards