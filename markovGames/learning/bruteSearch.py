import numpy as np
from itertools import product
from markovGames.gameDefs.mdpDefs import Policy


def getAllDetPol(numStates, numActions):
    detProbs = [np.array([1 if j == i else 0 for j in range(numActions)]) for i in range(numActions)]
    return product(detProbs, repeat=numStates)


def getPolList(states, acSet):
    # list of possible deterministic policies
    numStates = len(states)
    numActions = len(acSet)
    detPol = getAllDetPol(numStates, numActions)
    return [Policy(states, pol, acSet) for pol in detPol]


def prodPolList(states, listActions):
    # get policies for each action Set
    polList = [getPolList(states, ac) for ac in listActions]
    return polList


def getPayoff(utilMap, listAcSet):
    # utilMap: maps list of agent policies to real numbers,
    # allPolicyList: list of agent i (list of possible policies)
    def utilInd(index):
        jointAc = [listAcSet[j][ind] for j, ind in enumerate(index)]
        val = utilMap(jointAc)
        return val

    numPL = [len(pL) for pL in listAcSet]
    payoff = np.zeros(numPL)
    for ind in product(*[range(nI) for nI in numPL]):
        payoff[ind] = utilInd(ind)
    return payoff


def getArgOpt(tensor):
    return np.unravel_index(np.argmax(tensor), tensor.shape)


def bruteFindNash(payoffList):
    TOLERANCE = 1e-7
    cpnes = list(np.argwhere(payoffList[0] > np.amax(payoffList[0], 0) - TOLERANCE))
    cpnes = [tuple(cpne) for cpne in cpnes]
    N = len(payoffList)

    for i in range(1, N):
        pMat = payoffList[i]
        for cpne in cpnes[:]:
            ind = cpne[:i] + (slice(None),) + cpne[i + 1:]
            if pMat[cpne] < np.max(pMat[ind]) - TOLERANCE:
                cpnes.pop(cpnes.index(cpne))
    return cpnes


def getEfficiency(cpnes, welfareMat):
    # welfareMat - matrix form of welfare
    pneWelf = [welfareMat[cpne] for cpne in cpnes]
    opt = np.max(welfareMat)
    priceRatios = [float(pne) / opt for pne in pneWelf]
    return priceRatios


def getPoA(cpnes, welfareMat):
    return min(getEfficiency(cpnes, welfareMat))
