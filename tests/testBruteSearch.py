import unittest
from markovGames.gameDefs.markovDefs import *
from markovGames.examples.prisonerDilemma import *
from markovGames.learning.bruteSearch import *


pol1 = Policy([1, 2], [np.array([1, 0]), np.array([0, 1])], ['C', 'D'])
pol2 = Policy([1, 2], [np.array([1, 0]), np.array([0, 1])], ['C', 'D'])

jpol = JointPolicy([pol1, pol2])
jpolExpl = Policy([1, 2], [np.array([1, 0, 0, 0]), np.array([0, 0, 0, 1])], [('C', 'C'), ('C', 'D'), ('D', 'C'), ('D', 'D')])


greward = ParetoPrisonerReward
rewardList = [ParetoPrisonerReward2, ParetoPrisonerReward2]
gamma = .5
pdts = PDTS()
stDist = [1, 0]
MG = MultiMDP(greward, rewardList, gamma, pdts, stDist)


class TestGetAllPol(unittest.TestCase):

    def test_getAllDetPol(self):
        allPols = list(getAllDetPol(len(jpol.states), len(jpol.acSet)))
        self.assertEqual(len(allPols), 16)

    def test_getPolList(self):
        states = jpol.states
        acSet = jpol.acSet
        policyList = getPolList(states, acSet)
        self.assertEqual(policyList[0].aGivenS(1, ('C', 'C')), 1)
        self.assertEqual(len(policyList), 16)


class TestBruteForce(unittest.TestCase):

    def test_getPayoff(self):
        def W(polList):
            return MG.J(JointPolicy(polList), -1)
        states = MG.TS.states
        acSet1 = pol1.acSet
        acSet2 = pol2.acSet
        prodList = prodPolList(states, [acSet1, acSet2])
        WMat = getPayoff(W, prodList)
        self.assertEqual(np.shape(WMat), (4, 4))
        self.assertEqual(WMat[1][1], MG.J(jpolExpl, -1))

        cpnes = bruteFindNash([WMat, WMat])
        getEfficiency(cpnes, WMat)
        getPoA(cpnes, WMat)


if __name__ == '__main__':
    unittest.main(exit=False)
