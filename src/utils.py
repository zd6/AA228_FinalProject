import json
from policy import RandomPolicy, VI
from const import *

def parse_config(configFile):
    f = open(configFile,)
    return json.load(f)
    
def policy_dict(policy_name):
    if policy_name == "RANDOM":
        return RandomPolicy
    if policy_name == "VI":
        return VI
    if policy_name == "DQN":
        return RandomPolicy
