import unittest
from markovGames.gameDefs.markovDefs import *
from markovGames.examples.prisonerDilemma import *


pol1 = Policy([1, 2], [np.array([1, 0]), np.array([0, 1])], ['C', 'D'])
pol2 = Policy([1, 2], [np.array([1, 0]), np.array([0, 1])], ['C', 'D'])

jpol = JointPolicy([pol1, pol2])
jpolExpl = Policy([1, 2], [np.array([1, 0, 0, 0]), np.array([0, 0, 0, 1])], [('C', 'C'), ('C', 'D'), ('D', 'C'), ('D', 'D')])


def avgVal(saValPairs, jointPolicy, i, TS):
    def averageAi(rspair, ai, s):
        def probaNoti(ac):
            probAc = jointPolicy.aGivenS(st, ac)
            probAci = jointPolicy.policyList[i].aGivenS(st, ac[i])
            return probAc / probAci
        return sum(rspair[j]*probaNoti(ac) for j, ac in enumerate(AC) if ac[i] == ai)
    dictSA = dict(zip(jointPolicy.saPairs(), saValPairs))
    ST = TS.states
    AC = jointPolicy.acSet
    ACi = jointPolicy.policyList[i].acSet
    print(dictSA)
    rsPairs = np.reshape(saValPairs, (len(ST), len(AC)))
    print(rsPairs)
    avgVal = [averageAi(rsPairs[j, :], ai, st) for j, st in enumerate(ST) for ai in ACi]
    return avgVal




class TestJointPolicy(unittest.TestCase):

    def test_instanceVar(self):
        self.assertTrue(jpol.states == [1, 2])
        self.assertTrue(jpol.acSet == [('C', 'C'), ('C', 'D'), ('D', 'C'), ('D', 'D')])

    def test_paction(self):
        for s in jpol.states:
            self.assertTrue(all(jpol.paction(s) == jpolExpl.paction(s)))

    def test_aGivenS(self):
        for s in jpol.states:
            for a in jpol.acSet:
                self.assertTrue(jpol.aGivenS(s, a) == jpolExpl.aGivenS(s, a))

    def test_toVec(self):
        self.assertTrue(all(jpol.toVec() == jpolExpl.toVec()))


class TestMultiMDP(unittest.TestCase):

    def test_init(self):
        self.assertEqual(self.MG.gamma, .5)

    def test_decentralize(self):
        gamma = .5
        pdts = PDTS()
        stDist = [1, 0]
        mdp = MDP(ParetoPrisonerReward, gamma, pdts, stDist)
        mg = self.MG
        self.assertEqual(mg.J(jpol, 'G'), mdp.J(jpol))
        self.assertNotEqual(mg.J(jpol, 0), mdp.J(jpol))
        self.assertEqual(mg.J(jpol, 1), mg.J(jpol, 0))

    def test_avgVal(self):
        mg = self.MG
        Qvec = mg.Q(jpol, 0)
        avgVal(Qvec, jpol, 0, mg.TS)

    def setUp(self):
        greward = ParetoPrisonerReward
        rewardList = [ParetoPrisonerReward2, ParetoPrisonerReward2]
        gamma = .5
        pdts = PDTS()
        stDist = [1, 0]
        self.MG = MultiMDP(greward, rewardList, gamma, pdts, stDist)


if __name__ == '__main__':
    unittest.main(exit=False)
