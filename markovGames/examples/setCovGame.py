#!/usr/bin/env python
# Author : Rohit Konda
# Copyright (c) 2020 Rohit Konda. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""

"""


from markovGames.gameDefs.mdpDefs import TS
from itertools import product
import numpy as np


class SetCovTS(TS):
    def __init__(self, states):
        TS.__init__(self, states)

    def ptrans(self, s, snext, a):
        covRes = self.cov(a)
        probs = [self.ptransR(j, s[j], snext[j], covRes[j]) for j in range(len(s))]
        return np.prod(probs)

    def ptransR(self, j, res, resNext, aRes):
        # j - index of resource, res - state of resource
        # resNext - state of next Resource, aRes - num agents covering resource
        raise NotImplementedError

    def cov(self, a):
        return np.sum(a, axis=0)


class BinarySCTS(SetCovTS):
    def __init__(self, baseState):
        SetCovTS.__init__(self, genBinStates(baseState))
        self.baseState = baseState

    def ptransR(self, j, res, resNext, aRes):
        p = 1
        if aRes > 0:
            if resNext == self.baseState[j]:
                return 1 - p
            elif resNext == 0:
                return p
            else:
                raise ValueError('not correct res')
        else:
            if resNext == self.baseState[j]:
                return p
            elif resNext == 0:
                return 1 - p
            else:
                raise ValueError('not correct res')


def setCovR(s, a):
    cov = np.sum(a, axis=0)
    val = np.dot(cov > 0, s)
    return val


def genBinStates(baseState):
    return list(product(*[[0, bR] if bR != 0 else [0] for bR in baseState]))
