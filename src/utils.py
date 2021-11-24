import json
from policy import RandomPolicy, VI

CHANNELS = 3
ROAD = 0
CITY = 1
PACKAGE = 2
DAY = 0
NIGHT = 1
PACKAGE_APPR = 10
directions = [[-1,0],[1,0],[0,-1],[0,1]]

def parse_config(configFile):
    f = open("config.json",)
    return json.load(f)
    
def policy_dict(policy_name):
    if policy_name == "RANDOM":
        return RandomPolicy
    if policy_name == "VI":
        return VI
