'''
Structure written by: Nils Napp
Modified & Solution by: Jiwon Choi (F18 HW script)
'''
import numpy as np
from markov import *
import random

# Actions of Maze problem
actions = ['up', 'left', 'down', 'right', 'stop']

class MDPMaze:
    def __init__(self, maze, stateReward):

        self.maze = maze
        self.stateReward = stateReward
        self.stateSize = maze.stateSize
        self.stateReward.resize(self.stateSize)

        self.eps = 0.30
        self.gamma = 0.9
        self.rewardM = np.ones(self.stateSize) * (-1)

        # place holders for computing transition matrices
        self.Aup = None
        self.Aleft = None
        self.Adown = None
        self.Aright = None
        self.Astop = None

        # computeTransitionMatrices function should compute self.Aup, self.Aleft, self.Adown, self.Aright and self.Astop
        # update the 5 matrices inside computeTransitionMatrices()
        self.computeTransitionMatrices()

        self.value = np.zeros(self.stateSize)
        self.policy = []

        # You can use this to construct the noisy matrices

    def ARandomWalk(self):
        A = np.zeros((self.stateSize, self.stateSize))

        for col in range(self.stateSize):
            nbrs = self.maze.nbrList(col)
            p = 1 / (len(nbrs) + 1)
            A[col, col] = p
            for r in nbrs:
                A[r, col] = p
        return A

    def computeTransitionMatrices(self):
        Arandom = self.ARandomWalk()

        Aup_perfect = np.zeros((self.stateSize, self.stateSize))
        Aleft_perfect = np.zeros((self.stateSize, self.stateSize))
        Adown_perfect = np.zeros((self.stateSize, self.stateSize))
        Aright_perfect = np.zeros((self.stateSize, self.stateSize))
        Astop_perfect = np.zeros((self.stateSize, self.stateSize))

        for i in range(0, self.stateSize):
            action = self.maze.actionList(i)
            position = self.maze.nbrList(i)

            # Aup_perfect
            if 'U' not in action:
                r, c = self.maze.state2coord(i)
                cord = self.maze.coord2state((r, c))
                Aup_perfect[cord][i] = 1
            else:
                p = action.index('U')
                Aup_perfect[position[p]][i] = 1

            # Aleft_perfect
            if 'L' not in action:
                r, c = self.maze.state2coord(i)
                cord = self.maze.coord2state((r, c))
                Aleft_perfect[cord][i] = 1
            else:
                p = action.index('L')
                Aleft_perfect[position[p]][i] = 1

            # Adown_perfect
            if 'D' not in action:
                r, c = self.maze.state2coord(i)
                cord = self.maze.coord2state((r, c))
                Adown_perfect[cord][i] = 1
            else:
                p = action.index('D')
                Adown_perfect[position[p]][i] = 1

            # Aright_perfect
            if 'R' not in action:
                r, c = self.maze.state2coord(i)
                cord = self.maze.coord2state((r, c))
                Aright_perfect[cord][i] = 1
            else:
                p = action.index('R')
                Aright_perfect[position[p]][i] = 1

            # Astop_perfect
            r, c = self.maze.state2coord(i)
            cord = self.maze.coord2state((r, c))
            Astop_perfect[cord][i] = 1

        self.Aup = ((1 - self.eps) * Aup_perfect) + (self.eps * Arandom)
        self.Aleft = ((1 - self.eps) * Aleft_perfect) + (self.eps * Arandom)
        self.Adown = ((1 - self.eps) * Adown_perfect) + (self.eps * Arandom)
        self.Aright = ((1 - self.eps) * Aright_perfect) + (self.eps * Arandom)
        self.Astop = Astop_perfect

    def valIter(self):
        ''' YOUR CODE HERE
        Update self.value
        # '''
        u1 = []
        for i in range(0, self.stateSize):
            u1.append(i)
        # T = self.computeTransitionMatrices()
        R = self.stateReward
        gamma = self.gamma
        while True:
            u = u1.copy()
            delta = 0
            for s in range(0, self.stateSize):
                # 五连dot分别算出 上下左右的value 算出u1【s】
                downvalue = np.dot(self.Adown[:, s], self.value)
                leftvalue = np.dot(self.Aleft[:, s], self.value)
                rightvalue = np.dot(self.Aright[:, s], self.value)
                upvalue = np.dot(self.Aup[:, s], self.value)
                stopvalue = np.dot(self.Astop[:, s], self.value)
                maxvalue = max(downvalue, leftvalue, rightvalue, upvalue, stopvalue)
                u1[s] = R[s] + gamma * maxvalue
                self.value[s] = u1[s]
                delta = max(delta, abs(u1[s] - u[s]))
            if delta <= self.eps * (1 - gamma)/gamma:
                return self.value
        # return self.value

    def polIter(self):
        ''' YOUR CODE HERE
        Update self.policy
        '''
        u1 = []
        if len(self.policy) == 0:
            for count in range(0, self.stateSize):
                self.policy.append("")
        for i in range(0, self.stateSize):
            u1.append(i)
        pi = {s: random.choice(actions) for s in range(0, self.stateSize)}
        while True:
            while True:
                u = u1.copy()
                delta = 0
                for s in range (0, self.stateSize):
                    if pi[s] == "up":
                        u1[s] = self.stateReward[s] + self.gamma * np.dot(self.Aup[:, s], self.value)
                    if pi[s] == "down":
                        u1[s] = self.stateReward[s] + self.gamma * np.dot(self.Adown[:, s], self.value)
                    if pi[s] == "left":
                        u1[s] = self.stateReward[s] + self.gamma * np.dot(self.Aleft[:, s], self.value)
                    if pi[s] == "right":
                        u1[s] = self.stateReward[s] + self.gamma * np.dot(self.Aright[:, s], self.value)
                    if pi[s] == "stop":
                        u1[s] = self.stateReward[s] + self.gamma * np.dot(self.Astop[:, s], self.value)
                    self.policy[s] = pi[s]
                    self.value[s] = u1[s]
                    delta = max(delta, abs(u1[s] - u[s]))
                if delta <= self.eps * (1 - self.gamma)/self.gamma:
                    break
            unchanged = True
            for i in range (0, self.stateSize):
                downvalue = np.dot(self.Adown[:, i], self.value)
                leftvalue = np.dot(self.Aleft[:, i], self.value)
                rightvalue = np.dot(self.Aright[:, i], self.value)
                upvalue = np.dot(self.Aup[:, i], self.value)
                stopvalue = np.dot(self.Astop[:, i], self.value)
                maxvalue = max(downvalue, leftvalue, rightvalue, upvalue, stopvalue)
                if maxvalue == downvalue:
                    a = "down"
                if maxvalue == leftvalue:
                    a = "left"
                if maxvalue == rightvalue:
                    a = "right"
                if maxvalue == upvalue:
                    a = "up"
                if maxvalue == stopvalue:
                    a = "stop"
                if a != pi[i]:
                    pi[i] = a
                    unchanged = False
            if unchanged:
                return self.policy
        # return self.policy

# ------------------------------------------------------------- #
if __name__ == "__main__":
    myMaze = maze(np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]))

    stateReward = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 100, 0, 0, 0, 0, 0],
        [-1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000]])

    mdp = MDPMaze(myMaze, stateReward)

    iterCount = 100
    printSkip = 10

    for i in range(iterCount):
        # mdp.valIter()
        mdp.polIter()
        if np.mod(i, printSkip) == 0:
            print("Iteration ", i)
            print (mdp.policy)
            # print (mdp.value)
