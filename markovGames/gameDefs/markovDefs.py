#!/usr/bin/env python
# Author : Rohit Konda
# Copyright (c) 2020 Rohit Konda. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""
Class definitions for a Joint Policy and a Markov game.
"""

import numpy as np
from itertools import product
from markovGames.gameDefs.mdpDefs import *


class JointPolicy(Policy):
    """
    Class definition of Joint Policy pi = pi_1 times ... pi_n.
    """
    def __init__(self, policyList):
        states = policyList[0].states 
        acSet = list(product(*[p.acSet for p in policyList]))
        Policy.__init__(self, states, None, acSet)
        self.policyList = policyList  # list of agent policies.

    def paction(self, s):
        # probability distribution of action given state - returns \Delta A_i.
        return [self.aGivenS(s, a) for a in self.acSet]

    def aGivenS(self, s, a):
        # probability of action a given state s
        return np.prod([p.aGivenS(s, ai) for ai, p in zip(a, self.policyList)])

    def toVec(self):
        # represent policy as vector of probabilities [p(a | s1), p(a | s2)].
        return np.array([self.aGivenS(s, a) for s in self.states for a in self.acSet])


class MultiMDP(MDP):
    """
    Class definition of Markov Game (additional definitions to MDP).
    """
    def __init__(self, globalReward, rewardList, gamma, TS, stDist):
        MDP.__init__(self, None, gamma, TS, stDist)
        self.globalReward = globalReward  # global reward R(s, a).
        self.rewardList = rewardList  # agent reward [r_i for agent i].

    def Q(self, jointPolicy, i=None):
        # calculate Q, i = agent id ; i='G' corresponds to global reward.
        self._deCentral(i)
        return MDP.Q(self, jointPolicy)

    def A(self, jointPolicy, i=None):
        # calculate A, i = agent id ; i='G' corresponds to global reward.
        self._deCentral(i)
        return MDP.A(self, jointPolicy)

    def V(self, jointPolicy, i=None):
        # calculate V, i = agent id ; i='G' corresponds to global reward.
        self._deCentral(i)
        return MDP.V(self, jointPolicy)

    def J(self, jointPolicy, i=None):
        # calculate J, i = agent id ; i='G' corresponds to global reward.
        self._deCentral(i)
        return MDP.J(self, jointPolicy)

    def _deCentral(self, i):
        # set reward function when i in {0, n-1} or 'G'.
        if i == 'G':
            rewardF = self.globalReward
        elif i in range(len(self.rewardList)):
            rewardF = self.rewardList[i]
        else:
            rewardF = self.rewardF
        self.rewardF = rewardF

    def avgVal(self, saValPairs, jointPolicy, i):
        # 
        # saPairs = jointPolicy.saPairs()
        # ST = self.TS.states
        # AC = jointPolicy.acSet

        # for s in ST:

        # np.reshape(saValPairs, )
        # np.zeros((a, ai))
        # [[join]]

        # _pactions = 
        # vali = func(notiPolicy, i)
        # return np.dot(vali, notiPolicy.toVec())
        # dictSA = dict(zip(jointPolicy.saPairs(), saValPairs))
        # print(dic)
        pass

    def gradJ(self, i, jointPolicy):
        # calculate gradient given a policy (Policy) and agent i.
        # returns [\grad(s, a_i) for all s, a_i].
        ptrans = self.TS.genPtrans(jointPolicy)
        dvDist = np.dot(self.discVisit(ptrans), self.stDist)
        policyI = jointPolicy.policyList[i]
        aNi = len(policyI.acSet)
        expandDV = np.repeat(dvDist, aNi)
        Qvec = self.Q(jointPolicy, i)
        Qavgi = self.avgVal(Qvec, jointPolicy, i)
        return (1. / (1. - self.gamma)) * np.multiply(expandDV, Qavgi)
