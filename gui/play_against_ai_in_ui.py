import os
import sys
sys.path.append("../")
import tqdm
import pickle
import numpy as np
from ray.rllib.agents.impala.vtrace_policy import VTraceTFPolicy
from flask import Flask, render_template
from flask_socketio import SocketIO,emit
import time
from threading import Thread
import threading
import random
import json
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

conf = eval(open("../confs/nl_holdem.py").read().strip())
env = NlHoldemEnvWrapper(
    conf
)
weight_index = 1048
nn_agent = NNAgent(env.observation_space,
                        env.action_space,
                        conf,
                        f"../weights/c_{weight_index}.pkl",
                        f"oppo_c{weight_index}")


class MyThread():
    def __init__(self, args=(), kwargs=None):
        Thread.__init__(self, args=(), kwargs=None)
        self.daemon = True
        self.messages = []
        self._stop_event = threading.Event()

        self.env = NlHoldemEnvWrapper(
            conf
        )

    def gen_obs(self,r,d):
        legal_actions = [
            ["Fold",0],
            ["Check/Call", 0],
            ["Raise Half Pot", 0],
            ["Raise Pot", 0],
            ["Allin", 0],
            ["Next Game", 0],
        ]

        hand_p0 = self.env.env.get_state(0)["raw_obs"]["hand"]
        hand_p1 = self.env.env.get_state(1)["raw_obs"]["hand"]
        public = self.env.env.get_state(1)["raw_obs"]["public_cards"]
        all_chip = self.env.env.get_state(1)["raw_obs"]["all_chips"]
        stakes = self.env.env.get_state(1)["raw_obs"]["stakes"]
        pot = self.env.env.get_state(1)["raw_obs"]["pot"]
        all_chip = [int(i) for i in all_chip]
        stakes = [int(i) for i in stakes]
        pot = int(pot)
        actions = self.env.env.get_state(1)["action_record"]

        action_recoards = []
        for pid,one_action in actions:
            a_name = one_action.name
            if a_name == "CHECK_CALL":
                a_name = "check/call"
            else:
                a_name = a_name.replace("_"," ").lower()
            action_recoards.append(
                [int(pid), a_name]
            )

        if d:
            legal_actions[-1][1] = 1
            payoffs = [int(i) for i in r]
        else:
            for i,one_action in enumerate(self.env.env.get_state(1)["raw_obs"]["legal_actions"]):
                legal_actions[one_action.value][1] = 1
            payoffs = [0,0]

        message = {
            "text" : "game action",
            "data" :{
                "ai_id": self.ai_id,
                "hand_p0": hand_p0,
                "hand_p1": hand_p1,
                "public": public,
                "chip": all_chip,
                "stakes": stakes,
                "pot": pot,
                "legal_actions": legal_actions,
                "done": d,
                "payoffs": payoffs,
                "action_recoards": action_recoards,
            }
        }
        return message

    def run(self):
        self.ai_id = random.randint(0,1)
        obs = self.env.reset()
        d = False
        r = [0,0]
        if self.env.my_agent() == self.ai_id and not d:
            action_ind = nn_agent.make_action(obs)
            obs, r, d, i = self.env.step(action_ind)

        socketio.emit('message_from_server', self.gen_obs(r,d))

    def send_message(self, message):
        action_id = message["action_id"]
        if action_id != 5:
            obs, r, d, i = self.env.step(message["action_id"])
            while self.env.my_agent() == self.ai_id and not d:
                action_ind = nn_agent.make_action(obs)
                obs, r, d, i = self.env.step(action_ind)
            socketio.emit('message_from_server', self.gen_obs(r,d))
        else:
            d = False
            r = [0,0]
            self.env = NlHoldemEnvWrapper(
                conf
            )
            self.ai_id = random.randint(0, 1)
            obs = self.env.reset()
            if self.env.my_agent() == self.ai_id and not d:
                action_ind = nn_agent.make_action(obs)
                obs, r, d, i = self.env.step(action_ind)
            socketio.emit('message_from_server', self.gen_obs(r,d))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app,logger=True, async_mode='threading', engineio_logger=False)

t = None

# Display the HTML Page & pass in a username parameter
@app.route('/')
def html():
    return render_template('index.html', username="tester")

# Receive a message from the front end HTML
@socketio.on('send_message')
def message_recieved(data):
    global t
    if data['text'] == "start":
        if t is None:
            t = MyThread()
            t.run()
        else:
            t = MyThread()
            t.run()
    if data['text'] == "restart":
        t = MyThread()
        t.run()
    elif data['text'] == "load":
        t.resend_last_message()
    else:
        if "action_id" in data:
            t.send_message(data)

# Actually Start the App
if __name__ == '__main__':
    """ Run the app. """
    #import webbrowser
    #webbrowser.open("http://localhost:8000")
    socketio.run(app,host="0.0.0.0", port=8000, debug=False)
