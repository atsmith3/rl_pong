import sys
import random


class RandomPongAgent:

    def __init__(self):
        pass

    def train(self):
        pass

    def eval(self, state) -> int:
        return random.choice([0, 1, 2])
