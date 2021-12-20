#!/usr/bin/env python
# Author : Rohit Konda
# Copyright (c) 2020 Rohit Konda. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""
Example Game for 2 player Repeated Game Prisoner Game Scenario.
If both players cooperate, they stay in state 1 otherwise they switch to state 2.
"""


from markovGames.gameDefs.mdpDefs import TS


class PDTS(TS):
    """
    Transition sytem for prisoner dilemma - only action dependent.
    """

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
    # greatest reward when cooperating, subreward when defecting
    # using common interest.
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