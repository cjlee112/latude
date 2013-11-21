import random
import numpy
from scipy import stats

import info

class ConditionalPlayer(object):
    def __init__(self, p, id=None, first_move=None):
        """p is a list of conditional probabilities of cooperating based on the last outcomes in dictionary order, e.g.
        p = [1., 0.1, 0.2, 0] for
        d = {'CC': 1., 'CD':0.1, 'DC':0.2 'DD':0}
        identifier should be a unique identifier"""
        self.p = p
        self.p = dict(zip(['CC','CD','DC','DD'],p))
        self.id = id
        self.is_first_move = True
        if first_move:
            self._first_move = first_move
        else:
            self._first_move = 'C'
    
    def first_move(self):
        """Override if other functionality is desired, e.g.
        random.choice(['C', 'D']) """
        return self._first_move
    
    def next_move(self, last_move=None, player_id=None):
        """last_move should be a string 'CC', 'CD', etc."""
        if self.is_first_move:
            self.is_first_move = False
            return self.first_move()
        if not last_move:
            return self.first_move()
        r = self.p[last_move]
        if random.random() < r:
            return 'C'
        return 'D'

def example_play(ep=0.01, moves=10):
    p = [1.-ep, ep, ep, 1.-ep]
    q = [ep, ep, ep, ep]
    p1 = ConditionalPlayer(p)
    p2 = ConditionalPlayer(q, first_move='D')
    history = [p1.next_move() + p2.next_move()]
    for i in range(moves):
        yield history[-1]
        choice1 = p1.next_move(history[-1])
        choice2 = p2.next_move(info.swap_moves(history[-1]))
        history.append(choice1 + choice2)

def inference_vs_conditional(p, moves=100):
    p1 = info.InferencePlayer2()
    p2 = ConditionalPlayer(p)
    history = [p1.next_move() + p2.next_move()]
    for i in range(moves):
        yield history[-1]
        choice1 = p1.next_move(history[-1])
        choice2 = p2.next_move(info.swap_moves(history[-1]))
        history.append(choice1 + choice2)

def seed_population(a=10, b=20):
    population = []
    next_id = 0
    p = [1.-ep, ep, ep, 1.-ep]
    q = [ep, ep, ep, ep]
    for _ in range(a):
        p1 = ConditionalPlayer(p, id=next_id)
        population.append(p1)
        next_id += 1
    for _ in range(b):
        p1 = ConditionalPlayer(q, id=next_id, first_move='D')
        population.append(p1)
        next_id += 1
    return population

def population_sim(rounds=1000):
    ## seed population
    population = seed_population()
    next_id = len(population)
    for round_number in range(rounds):
        ## compute fitness by play with every over player
        ## fitness proportionate selection to reproduce
        ## randomly select one to die
        pass

## Examples ##
# ep = 0.01
## GTFT
#p = dict(zip(['CC','CD','DC','DD'],[1.-ep, ep, 1.-ep, ep]))
## WSLS
#p = dict(zip(['CC','CD','DC','DD'],[1.-ep, ep, ep, 1.-ep]))
## ALL D
#p = dict(zip(['CC','CD','DC','DD'],[ep, ep, ep, ep]))
## ALL C
#p = dict(zip(['CC','CD','DC','DD'],[1.-ep, 1.-ep, 1.-ep, 1.-ep]))

if __name__ == '__main__':
    ep = 0.05
    # WSLS
    p = [1.-ep, ep, ep, 1.-ep]
    # GTFT
    #ep = 0
    #p = [1.-ep, ep, 1.-ep, ep]
    # ALLC
    #p = [1.-ep, 1.-ep, 1.-ep, 1.-ep]
    # ALLD
    #p = [ep, ep, ep, ep]
    gen = inference_vs_conditional(p)
    for play in gen:
        print play


