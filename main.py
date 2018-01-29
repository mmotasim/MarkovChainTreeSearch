import random
import math
import hashlib
import argparse
import pandas as pd
import numpy as np
import datetime
#load data
df = pd.read_csv('taxi_1.csv')
df['trip_total'] = df.trip_total.str.replace('$', '').astype(float)
dnbhd=pd.read_csv('neighbours.csv')
dnbhd['Neighbour'] = dnbhd.Neighbour.str.split(',')
K=5
cost=1
term=100
driver=np.random.choice(df['taxi_ID'])
dnbhd = dnbhd.iloc[0:77]
#df=df[df.taxi_ID==driver]
llist=list(df['trip_start_timestamp'])
plist=[]
for l in llist:
    plist.append(datetime.datetime.strptime(l, "%Y-%m-%d %H:%M:%S"))
plist.sort()
starttime=plist[0]


class State():
    locprime = 22
    R = 0
    loc = []

    def __init__(self, R=0, locprime="Start", current_loc=1, K=K):
        self.locprime = locprime
        self.K = K
        self.R = R
        self.loc.append(current_loc)#working as we want it to. But, there is an inconsistency
        self.current_loc = current_loc

    def next_state(self):
        P = Passengers(self.current_loc)
        nbhdv = nbhd(self.current_loc)
        if P is not None:
            a = nbhdv + ['T'] + list(P[0])
        else:
            a = nbhdv + ['T']
        R = 0
        self.action = random.choice(a)
        if self.action == 'T':
            next_loc = 'Term'
        elif self.action[0:1] == 'P':
            next_loc = P[1][int(self.action[1:2])]
            R = 1
        else:
            next_loc = self.action
            R = 0
        # next=State(R,self.loc,current_loc, self.K-1)
        # return next
        return State(R, self.current_loc, next_loc, self.K - 1)

    def terminal(self):
        if self.K == 0:
            return True
        return False

    def reward(self):
        r=0
        if self.action[0:1] == 'P':
            P = Passengers(self.current_loc)
            r = P[2][int(self.action[1:2])] - P[3][int(self.action[1:2])] * cost
        elif self.action == 'T':
            r = -term
        else:
            r = -22 #Write function to calculate distance between communities
        return r

    def __hash__(self):
        return int(hashlib.md5(str(self.moves).encode('utf-8')).hexdigest(), 16)

    def __eq__(self, other):
        if hash(self) == hash(other):
            return True
        return False

    def __repr__(self):
        s = "Value: %d; Moves: %s" % (self.value, self.moves)
        return s


class Node():
    def __init__(self, state, parent=None):
        self.visits = 1
        self.reward = 0.0
        self.state = state
        self.children = []
        self.parent = parent

    def add_child(self, child_state):
        child = Node(child_state, self)
        self.children.append(child)

    def update(self, reward):
        self.reward += reward
        self.visits += 1

    def fully_expanded(self):
        if len(self.children) == self.state.num_moves:
            return True
        return False

    def __repr__(self):
        s = "Node; children: %d; visits: %d; reward: %f" % (len(self.children), self.visits, self.reward)
        return s


def Passengers(loc):
    dfp = df[df.pickup_community_area == loc]
    w = df[df.pickup_community_area == loc].count()
    a = w[1]
    if a>0:
        x = dfp[['dropoff_community_area', 'trip_total', 'trip_miles']]
        dca = np.array(x)
        dfpa = dfp.groupby(['pickup_community_area'], as_index=False)['pickup_community_area'].agg(['count'],
                                                                                                   as_index=False)
        passcount = int(dfpa['count'])
        P = [[y for x in range(4)] for y in range(a)]
        for j in range(a):
            P[j][0] = 'P' + str(P[j][0])  # passenger ID
            P[j][1] = dca[j][0]  # dropoff community area
            P[j][2] = dca[j][1] / passcount  # Expected fare for each passenger
            P[j][3] = dca[j][2]  # trip distance for each trip
        P = pd.DataFrame(P)
        return P
    else:
        return None


def nbhd(loc):
    a = dnbhd[dnbhd.Community == loc]
    print(a)
    nbhdv = a.iloc[0][1]
    return nbhdv


def UCTSEARCH(budget, root):
    for iter in range(int(budget)):
        front = TREEPOLICY(root)
        reward = DEFAULTPOLICY(front)
        BACKUP(front, reward)
    return BESTCHILD(root, 0)


def TREEPOLICY(node):
    # a hack to force 'exploitation' in a game where there are many options, and you may never/not want to fully expand first
    SCALAR=0
    while node.state.terminal() == False:
        if len(node.children) == 0:
            return EXPAND(node)
        elif random.uniform(0, 1) < .5:
            node = BESTCHILD(node, SCALAR)
        else:
            if node.fully_expanded() == False:
                return EXPAND(node)
            else:
                node = BESTCHILD(node, SCALAR)
    return node


def EXPAND(node):
    tried_children = [c.state for c in node.children]
    new_state = node.state.next_state()
    while new_state in tried_children:
        new_state = node.state.next_state()
    node.add_child(new_state)
    return node.children[-1]


# current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)
def BESTCHILD(node, scalar):
    bestscore = -100000
    bestchildren = []
    for c in node.children:
        exploit = c.reward / c.visits
        explore = math.sqrt(2.0 * math.log(node.visits) / float(c.visits))
        score = exploit + scalar * explore
        if score == bestscore:
            bestchildren.append(c)
        if score > bestscore:
            bestchildren = [c]
            bestscore = score
    if len(bestchildren) == 0:
        print("OOPS: no best child found, probably fatal")
    return random.choice(bestchildren)


def DEFAULTPOLICY(front):
    reward=0
    while front.state.terminal() == False:
        reward=reward+front.parent.state.reward()
        state = front.state.next_state()
        front = Node(state,front)
    return reward


def BACKUP(node, reward):
    while node != None:
        node.visits += 1
        node.reward += reward
        node = node.parent
    return


num_sims = 1000
current_node = Node(State())
for l in range(K):
    current_node = UCTSEARCH(num_sims / (l + 1), current_node)
    print("level %d" % l)
    print("Num Children: %d" % len(current_node.children))
    for i, c in enumerate(current_node.children):
        print(i, c)
    print("Best Child: %s" % current_node.state)

    print("--------------------------------")