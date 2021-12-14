import unittest
from markovGames.gameDefs.mdpDefs import *
from markovGames.examples.prisonerDilemma import *


class TestPolicy(unittest.TestCase):

    def test_ptrans(self):
        self.assertTrue(np.array_equal(self.JP.paction(1), [1, 0, 0, 0]))

    def test_aGivenS(self):
        self.assertTrue(self.JP.aGivenS(1, ('C', 'C')), 1)

    def test_toVec(self):
        self.assertTrue(np.array_equal(self.JP.toVec(), [1, 0, 0, 0, 0, 0, 0, 1]))

    def setUp(self):
        self.JP = Policy([1, 2], [np.array([1, 0, 0, 0]), np.array([0, 0, 0, 1])], [('C', 'C'), ('C', 'D'), ('D', 'C'), ('D', 'D')])


class TestTS(unittest.TestCase):

    def test_ptrans(self):
        self.JP = Policy([1, 2], [np.array([1, 0, 0, 0]), np.array([0, 0, 0, 1])], [('C', 'C'), ('C', 'D'), ('D', 'C'), ('D', 'D')])
        pdts = PDTS()
        ptrans = pdts.genPtrans(self.JP)
        self.assertTrue(pdts.ptrans(1, None, ('C', 'C')), [.9, 1])
        
        self.assertTrue(np.array_equal(ptrans, [[0.9, 0.2], [0.1, 0.8]]))
        stepDist = pdts.step(ptrans, [1, 0])
        self.assertTrue(np.array_equal(stepDist, np.array([.9, .1])))


    def test_toVec(self):
        self.JP = Policy([1, 2], [np.array([1, 0, 0, 0]), np.array([0, 0, 0, 1])], [('C', 'C'), ('C', 'D'), ('D', 'C'), ('D', 'D')])
        pdts = PDTS()
        vec1 = pdts.toVec(1)
        vec2 = pdts.toVec(2)
        self.assertTrue(np.array_equal(vec1, [1, 0]))
        self.assertTrue(np.array_equal(vec2, [0, 1]))


class TestProb(unittest.TestCase):

    def test_discvisit(self):
        ptrans = self.mdp.TS.genPtrans(self.JP)
        gamma = self.mdp.gamma

        approxDV = (1 - gamma) * sum([gamma**t * np.linalg.matrix_power(ptrans, t) for t in range(10)])
        DV = self.mdp.discVisit(ptrans)
        self.assertLess(np.linalg.norm(DV - approxDV), .01)

    def test_getVJ(self):
        V = self.mdp.V(self.JP)
        J = self.mdp.J(self.JP)
        self.assertLess(abs(V[0] - J), .01)

    def test_Q(self):
        Q1 = self.mdp.Q(self.JP, 'Bellman')
        Q2 = self.mdp.Q(self.JP, 'BellmanValue')
        self.assertLess(np.linalg.norm(Q1 - Q2), .001)

    def test_gradJ(self):
        self.mdp.gradJ(self.JP)

    def testA(self):
        A = self.mdp.A(self.JP)
        avgA = np.dot(self.JP.toVec(), A)
        self.assertLess(abs(avgA), .001)


    def setUp(self):
        self.JP = Policy([1, 2], [np.array([1, 0, 0, 0]), np.array([0, 0, 0, 1])], [('C', 'C'), ('C', 'D'), ('D', 'C'), ('D', 'D')])
        
        gamma = .5
        pdts = PDTS()
        stDist = [1, 0]
        self.mdp = MDP(ParetoPrisonerReward, gamma, pdts, stDist)



if __name__ == '__main__':
    unittest.main(exit=False)