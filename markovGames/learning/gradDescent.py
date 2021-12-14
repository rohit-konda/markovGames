import numpy as np


def gradAscent(grad, initState):
    # container function to run gradient descent
    gradAscender = GradAscent(grad)
    finalState = gradAscender.run(initState)
    return finalState


class GradAscent():
    def __init__(self, grad):
        self._grad = grad  # gradient function
        self.history = []  # state history

    def step(self, ascent=True):
        # calculate gradient and step in the direction
        state = self.history[-1]
        if ascent:
            newState = state + self.stepsize() * self.grad(state)
        else:
            newState = state - self.stepsize() * self.grad(state)
        self.history.append(newState)

    def stopping(self, t):
        # stopping rule
        return t > 100

    def grad(self, state, stoch=False, eps=None):
        # container function for calculating the gradient
        if not stoch:
            return self._grad(state)
        else:
            grad = self._grad(state)
            return grad + eps*np.random.rand(np.shape(grad))

    def run(self, initState):
        # run the gradient descent algorithm
        self.history = [initState]
        t = 0
        while not self.stopping(t):
            self.step(t)
            t += 1
        finalState = self.history[-1]
        return finalState