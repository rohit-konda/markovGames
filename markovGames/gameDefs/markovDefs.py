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

    def Q(self, policyList, i=None):
        # calculate Q, i = agent id ; i='G' corresponds to global reward.
        self._deCentral(i)
        return MDP.Q(self, JointPolicy(policyList))

    def A(self, policyList, i=None):
        # calculate A, i = agent id ; i='G' corresponds to global reward.
        self._deCentral(i)
        return MDP.A(self, JointPolicy(policyList))

    def V(self, policyList, i=None):
        # calculate V, i = agent id ; i='G' corresponds to global reward.
        self._deCentral(i)
        return MDP.V(self, JointPolicy(policyList))

    def J(self, policyList, i=None):
        # calculate J, i = agent id ; i='G' corresponds to global reward.
        self._deCentral(i)
        return MDP.J(self, JointPolicy(policyList))

    def _deCentral(self, i):
        # set reward function when i in {0, n-1} or 'G'.
        if i == 'G':
            rewardF = self.globalReward
        elif i in range(len(self.rewardList)):
            rewardF = self.rewardList[i]
        else:
            rewardF = self.rewardF
        self.rewardF = rewardF

    def avgVal(self, i, func, policyList):
        #
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
