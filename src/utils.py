import json
from VI.valueIteration import SampleVI
from policy import RandomPolicy, VI, DQNPolicy, GreedyPolicy
from const import *
import os

def parse_config(configFile):
    configFile = os.path.dirname(os.path.realpath(__file__)) + '/' + configFile
    f = open(configFile,)
    return json.load(f)
    
def policy_dict(policy_name):
    if policy_name == "RANDOM":
        return RandomPolicy
    if policy_name == "VI":
        return VI
    if policy_name == "DQN":
        return DQNPolicy
    if policy_name == "SampleVI":
        return SampleVI
    if policy_name == "GREEDY":
        return GreedyPolicy
