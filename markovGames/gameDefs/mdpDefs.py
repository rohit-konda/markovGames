import numpy as np


def R(s, a):
    # reward given state and action pair
    pass


class Policy():
    def __init__(self, states, pactions, acSet):
        self.states = states
        self._pactions = pactions
        self.acSet = acSet

    def paction(self, s):
        # probability distribution of action given state
        return self._pactions[self.states.index(s)]

    def aGivenS(self, s, a):
        # probability of action given state
        return self.paction(s)[self.acSet.index(a)]

    def toVec(self):
        # represent policy as vector of probabilities [p(a | s1), p(a | s2)]
        return np.array(self._pactions).flatten()

    def saPairs(self):
        return [(s, a) for s in self.states for a in self.acSet]


class TS():
    def __init__(self, states=None):
        self.states = states  # state labels

    def ptrans(self, s, snext, a):
        # calculate probability of transition from s to snext given action a
        raise NotImplementedError

    def step(self, ptrans, stDist):
        # propogate probabilities one step
        return np.dot(ptrans, stDist)

    def genPtrans(self, policy):
        # generate probability transition matrix given a policy
        def avgtrans(s, snext):
            expTrans = [self.ptrans(s, snext, a) for a in policy.acSet]
            pact = policy.paction(s)
            return np.dot(expTrans, pact)

        Ptrans = np.array([[avgtrans(s, snext) for s in self.states] for snext in self.states])
        return Ptrans

    def toVec(self, s):
        # vector identification of the state
        return np.array([int(s == si) for si in self.states])


class MDP():
    def __init__(self, rewardF, gamma, TS, stDist):
        self.rewardF = rewardF
        self.gamma = gamma
        self.TS = TS
        self.stDist = stDist

    def Q(self, policy, method='Bellman'):
        # calculate Q function through Bellman Function
        # for all state, action pairs
        if method == 'Bellman':
            ST = self.TS.states
            AC = policy.acSet
            rVec = [self.rewardF(s, a) for s in ST for a in AC]

            pT = self.TS.ptrans
            aGS = policy.aGivenS
            PTS = np.array([[pT(s, sn, a) * aGS(sn, an) for sn in ST for an in AC] for s in ST for a in AC])

            M = np.linalg.inv(np.identity(len(ST) * len(AC)) - self.gamma * PTS)
            Qvec = np.dot(M, rVec)
            return Qvec

        if method == 'BellmanValue':
            def expV(st, ac):
                sTrans = [self.TS.ptrans(st, sn, ac) for sn in self.TS.states]
                V = self.V(policy)
                return np.dot(sTrans, V)

            ST = self.TS.states
            AC = policy.acSet
            rVec = np.array([self.rewardF(s, a) for s in ST for a in AC])
            iterBellV = np.array([expV(s, a) for s in ST for a in AC])
            return rVec + self.gamma * iterBellV

    def A(self, policy):
        # get advantage function
        aN = len(policy.acSet)
        expandV = np.kron(self.V(policy), np.ones((aN,)))
        Avec = self.Q(policy) - expandV
        return Avec

    def V(self, policy):
        # get Value function through
        # row vector for each state (V(s))_{s in S}
        def avgReward(s):
            expTrans = [self.rewardF(s, a) for a in policy.acSet]
            pact = policy.paction(s)
            return np.dot(expTrans, pact)

        states = self.TS.states
        ptrans = self.TS.genPtrans(policy)
        DV = self.discVisit(ptrans)
        avgReward = np.array([avgReward(s) for s in states], dtype=np.float32)
        return (1. / (1. - self.gamma)) * np.dot(avgReward, DV)

    def J(self, policy):
        # get reward function
        V = self.V(policy)
        return np.dot(V, self.stDist)

    def discVisit(self, ptrans):
        # calculate discounted state visitation sum_t P(s^t = s_i | s^0 = s_j)
        M = np.identity(np.shape(ptrans)[0]) - self.gamma * ptrans
        # inverse might be computationally costly
        return (1 - self.gamma) * np.linalg.inv(M)

    def gradJ(self, policy):
        # calculate gradient, returning in a form of policy.toVec()
        dvDist = np.dot(self.discVisit(self.TS.genPtrans(policy)), self.stDist)
        aN = len(policy.acSet)
        expandDV = np.kron(dvDist, np.ones((aN,)))
        Qvec = self.Q(policy)
        return (1. / (1. - self.gamma)) * np.dot(expandDV, Qvec)
