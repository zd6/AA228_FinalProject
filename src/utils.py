import json
from policy import *

def parse_config(configFile):
    f = open("config.json",)
    return json.load(f)
    
def policy_dict(policy_name):
    if policy_name == "RANDOM":
        return random_policy

def 2