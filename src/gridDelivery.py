import numpy as np
from utils import *
from collections import deque
import matplotlib.pyplot as plt
CHANNELS = 3
ROAD = 0
CITY = 1
PACKAGE = 2
PACKAGE_APPR = 10
directions = [[-1,0],[1,0],[0,-1],[0,1]]

class GridDelivery:
    """
    Class contructor takes dimensions m and n by defualt they will be 5 and 5
    self.grid::np.ndarray([m*n, c])
    [       "
        0 0 0 0 0
        0 0 0 0 0
       "0 0 0 0 0"
        0 0 0 0 0
        0 0 0 0 0
            "
    ]
    State:
    (m*n) * (m*n)^k
    for 5 by 5 with 5 package and two part of traffic zones and two traffic level
    = 5^5*4 = 12500
    Action:
    4 >^<v
    pickup package without extra action


    2 levels of traffic for main and auxiliary roads
    60% if 9 < hour <18
    10% if hour < 9 or hour > 18
    Main road have +20%

    Packages(k):
    Anywhere with 5%
    Next to Main Road +20%
    max_waiting = w

    Rewards:
    Collecting Package:
        100  
    For every hour during rush operating cost:
        -5
    For every hour during outside rush operating cost:
        -10
    """
    def __init__(self) -> None:
        self.config()
        
    """
    Configurate the environment with file
    """
    def config(self, configFile = "config.json"):
        # parse parameter configuration file
        configDict = parse_config(configFile)

        # initialize meta data
        # current time stamp
        self.hour = configDict["START TIME"]
        # rush hour start time
        self.rush_hour_start = configDict["RUSH START"]
        # rush hour end time
        self.rush_hour_end = configDict["RUSH END"]
        # how many layers of city resident around main road, 
        # eg. 1 layer is only the houses on the main road, 
        #     2 layers are houses on the next to the main road
        self.city_layers = configDict["CITY"]
        # maximum number of packages environment can have
        self.max_package = configDict["MAX_PACKAGES"]
        # package generation probabilities
        self.package_prob = configDict["PACKAGE"]
        # traffic probablity at city cell and during rush hour 
        # the truck has self.traffic_prob["CITY"][0](%) chance to stuck at its position
        self.traffic_prob = configDict["TRAFFIC"]
        # reward function on operational cost and pickup earnings
        self.rewards = configDict["REWARDS"]
        # policy for step function
        self.policy = policy_dict(configDict["POLICY"])


        # initialize maps
        self.roads = np.array(configDict["ROAD MAP"])
        self.m, self.n = self.roads.shape
        assert(self.m == self.n and self.m %2 == 1)
        self.channels = CHANNELS
        self.grid = np.zeros((self.channels, self.m, self.n), dtype=np.int64)
        self.grid[ROAD,:,:] = self.roads
        self._initialize_city()
        # initialize truck and packages
        center = (self.m//2, self.n//2)
        """      
        [
            x 0 x 0 x
            0 0 0 0 0
            x 0 x 0 x
            0 0 0 0 0
            x 0 x 0 x
        ]
        """
        self.truck = (self.m//2, self.n//2)
        self.package_ankers = [(0, 0),         (0, center[1]),       (0, self.n-1),\
                               (center[0], 0), center,               (center[0], self.n-1),\
                               (self.m-1, 0),  (self.m-1,center[1]), (self.m-1, self.n-1),\
                               (-1, -1)]
        self._initialize_pkg()
        self.state_vector_dims = tuple([self.m, self.n] + [PACKAGE_APPR for _ in range(self.max_package)])
    

    """
    Initialize package dictionary and package queue
    """
    def _initialize_pkg(self):
        self.package_queue = deque()
        self.packages = {i:(-1,-1) for i in range(self.max_package)}
        self.packages_pos_to_id = {}
        self.generate_packages()
        for ID in self.packages.keys():
            if not self.package_queue:
                break
            pkgPos = self.package_queue.popleft()
            self.packages[ID] = pkgPos
            self.packages_pos_to_id[pkgPos] = ID
            self.grid[PACKAGE, pkgPos[0], pkgPos[1]] = 1
        self.generate_packages()


    """
    Encode state into index
    """
    def encode_state(self):
        package_idx = []
        for pos in self.packages.values():
            package_idx.append(self.find_nearest_anker_idx(pos))
        cur_state_idx = [*self.truck] + package_idx
        return np.ravel_multi_index(cur_state_idx, self.state_vector_dims)


    """
    Find the anker index of current package position
    """
    def find_nearest_anker_idx(self, pos):
        minDist = np.Inf
        minIdx = 0
        if pos == (-1,-1):
            return 9
        for idx, (x, y) in enumerate(self.package_ankers):
            dist = abs(pos[0] - x) + abs(pos[1] - y)
            if dist < minDist:
                minIdx = idx
                minDist = dist
        return minIdx
    

    """
    Make one minute step into future with self.policy
    """
    def step(self):
        reward = self._get_operational_cost()
        prev = tuple(self.truck)
        if self.grid[PACKAGE, self.truck[0], self.truck[1]]:
            reward += self.rewards["PACKAGE"]
            self.package_update(self.truck)
        action = self.policy(self.encode_state())
        nextTruck = self.move(action)
        if self._in_bound(*nextTruck):
            self.truck = nextTruck
        self.hour += 1/60
        return prev, action, reward, self.encode_state()



    """
    Get operational cost of this moment
    """
    def _get_operational_cost(self):
        if self._is_rush():
            return self.rewards["DAY"]
        return self.rewards["NIGHT"]

    """
    Check if is rush hour
    """
    def _is_rush(self):
        return self.rush_hour_start < self.hour < self.rush_hour_end


    """
    Try to move truck, considering traffic condition
    """
    def move(self, action):
        if self.grid[ROAD, self.truck[0], self.truck[1]] == 1:
            if np.random.rand() < self.traffic_prob["HIGHWAY"][self._is_rush()]/100:
                return self.truck
        else:
            if np.random.rand() < self.traffic_prob["STREET"][self._is_rush()]/100:
                return self.truck
        x, y = self.truck[0] + directions[action][0], self.truck[1] + directions[action][1]
        if self._in_bound(x, y):
            return (x, y)
        else:
            return self.truck
    
    """
    Generate packages until package queue is full
    """
    def generate_packages(self):
        for _ in range(self.m*self.n):
            i, j = np.random.randint(0, self.m, size = 2)
            if self.grid[PACKAGE, i, j] == 1 or self.truck == (i, j):
                continue
            if self.grid[CITY, i, j] == 1:
                if np.random.rand() < self.package_prob["CITY"][self._is_rush()]/100:
                    self.package_queue.append((i, j))
            else:
                if np.random.rand() < self.package_prob["RURAL"][self._is_rush()]/100:
                    self.package_queue.append((i, j))

    """
    Upadate self.grid package layer when reaching a package
    """
    def package_update(self, doneWithPos):
        ID = self.packages_pos_to_id[doneWithPos]
        self.grid[PACKAGE, self.truck[0], self.truck[1]] = 0
        del self.packages_pos_to_id[doneWithPos]
        if not self.package_queue:
            self.generate_packages()
        self.packages[ID] = self.package_queue.popleft()
        self.packages_pos_to_id[self.packages[ID]] = ID
        self.grid[PACKAGE, self.packages[ID][0], self.packages[ID][1]] = 1

    """
    Configurate city areas for package distribution, store in self.grid
    """
    def _initialize_city(self):
        self.city_map = np.array(self.roads)
        todo = deque(list(zip(*np.where(self.roads == 1))))
        layer = 1
        # Breath First Search to construct city around main road
        while todo:
            if layer >= self.city_layers:
                break
            level = len(todo)
            for _ in range(level):
                x, y = todo.popleft()
                for d1, d2 in [[-1,0],[1,0],[0,1],[0,-1]]:
                    if self._2d_is_valid(x+d1, y+d2, self.city_map, checkV = 0):
                        self.city_map[x+d1, y+d2] = 1
            layer += 1
        self.grid[CITY,:,:] = self.city_map
    
    """
    takes coordinate and map, chech if in bound
    """
    def _2d_is_valid(self, x, y, map, checkV = None):
        return self._in_bound(x,y, map = map) and (checkV == None or map[x, y] == checkV)

    def _in_bound(self, x, y, map = None):
        if map is None:
            map = self.grid[0]
        m, n = map.shape
        return x >= 0 and y >= 0 and x < m and y < n
    """
    Generate Gray scale grid maps for debug
    """
    def plot(self):
        truckLayer = np.zeros_like(self.grid[ROAD])
        truckLayer[self.truck] = 2
        plt.subplot(1,2,1)
        plt.imshow(self.grid[ROAD] + truckLayer*2, cmap="gray")
        plt.subplot(1,2,2)
        plt.imshow(self.grid[CITY] + self.grid[PACKAGE]*2, cmap="gray")
        plt.show()