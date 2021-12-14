import numpy as np
from itertools import product
from markovGames.gameDefs.mdpDefs import *


class JointPolicy(Policy):
    def __init__(self, policyList):
        states = policyList[0].states
        acSet = list(product(*[p.acSet for p in policyList]))
        Policy.__init__(self, states, None, acSet)
        self.policyList = policyList  # list of agent policies

    def paction(self, s):
        # probability distribution of action given state
        return [self.aGivenS(s, a) for a in self.acSet]

    def aGivenS(self, s, a):
        # probability of action given state
        return np.prod([p.aGivenS(s, ai) for ai, p in zip(a, self.policyList)])

    def toVec(self):
        # represent policy as vector of probabilities [p(a | s1), p(a | s2)]
        return np.array([self.aGivenS(s, a) for s in self.states for a in self.acSet])


class MultiMDP(MDP):
    def __init__(self, globalReward, rewardList, gamma, TS, stDist):
        MDP.__init__(self, None, gamma, TS, stDist)
        self.globalReward = globalReward
        self.rewardList = rewardList

    def Q(self, policy, i=None):
        self.deCentral(i)
        return MDP.Q(self, policy)

    def A(self, policy, i=None):
        self.deCentral(i)
        return MDP.A(self, policy)

    def V(self, policy, i=None):
        self.deCentral(i)
        return MDP.V(self, policy)

    def J(self, policy, i=None):
        self.deCentral(i)
        return MDP.J(self, policy)

    def deCentral(self, i):
        if i == -1:
            rewardF = self.globalReward
        elif i is not None:
            rewardF = self.rewardList[i]
        else:
            rewardF = self.rewardF
        self.rewardF = rewardF

    def avgVal(self, i, func, policyList):
        notiPolicy = JointPolicy([p for j, p in enumerate(policyList) if j != i])
        vali = func(notiPolicy, i)
        return np.dot(vali, notiPolicy.toVec())

    def gradJ(self, i, policyList):
        ### TODO ###
        jPolicy = JointPolicy(policyList)
        dvDist = np.dot(self.discVisit(self.TS.genPtrans(jPolicy)), self.stDist)
        aN = len(policyList[i].acSet)
        expandDV = np.kron(dvDist, np.ones((aN,)))
        Qvec = self.avgVal(i, self.Q, policyList)
        return (1. / (1. - self.gamma)) * np.dot(expandDV, Qvec)
