from markovGames.gameDefs.mdpDefs import TS


class PDTS(TS):
    def __init__(self):
        TS.__init__(self, [1, 2])

    def ptrans(self, s, snext, a):
        eps = .1

        if a == ('C', 'C'):
            if snext == 1:
                return 1 - eps
            else:
                return eps
        else:
            if snext == 2:
                return 1 - eps - .1
            else:
                return eps + .1


def ParetoPrisonerReward(s, a):
    r = 0
    if a == ('C', 'C'):
        r = 3
    elif a == ('D', 'D'):
        r = 2
    else:
        r = 1

    if s == 2:
        r += 2

    return r

def ParetoPrisonerReward2(s, a):
    return ParetoPrisonerReward(s, a) + 1