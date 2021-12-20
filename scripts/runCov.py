from markovGames.examples.setCovGame import *
from markovGames.learning.bruteSearch import *
from markovGames.gameDefs.markovDefs import *

np.set_printoptions(linewidth=np.inf)

basestate = [1, 1, 0]
states = genBinStates(basestate)

acSet1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
acSet2 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


greward = setCovR
rewardList = [greward, greward]
gamma = 2/3


ts = BinarySCTS(basestate)
stDist = np.zeros((len(states),))
stDist[-1] = 1
MG = MultiMDP(greward, rewardList, gamma, ts, stDist)


def W(polList):
    return np.round(MG.J(JointPolicy(polList), -1), 2)

MG.deCentral('G')
prodList = prodPolList(states, [acSet1, acSet2])
WMat = getPayoff(W, prodList)

print(WMat)
print(WMat.shape)
cpnes = bruteFindNash([WMat, WMat])
print(len(cpnes))
print(cpnes)

for cpne, eff in zip(cpnes, getEfficiency(cpnes, WMat)):
        print('NE', cpne, eff)
        [print(prodList[i][ci]._pactions) for i, ci in enumerate(cpne)]


for cpne, eff in zip(cpnes, getEfficiency(cpnes, WMat)):
    if eff == 1:
        print('OPT', cpne)
        [print(prodList[i][ci]._pactions) for i, ci in enumerate(cpne)]

print(states)
# print(stDist)
# [print(prodList[i][ci]._pactions) for i, ci in enumerate((3, 1))]
# print('OPT')
# [print(prodList[i][ci]._pactions) for i, ci in enumerate((8, 0))]
print('PoA', getPoA(cpnes, WMat))

# pL = [prodList[i][ci] for i, ci in enumerate((3, 1))]
# print(ts.genPtrans(JointPolicy(pL)))
# print([prodList[i][ci]._pactions for i, ci in enumerate((3, 1))])
# for s in [(1, 1, 0)]:
#     for snext in states:
#         a = ([0, 1, 0], [0, 0, 1])
#         print(s, snext, ts.ptrans(s, snext, a))