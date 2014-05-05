import collections

START = 0
ENACT = 1
EXPECT = 2
EXCLUDE = 3

class MarsStrategy(object):
    def __init__(self, scores=(2, -1, 3, 0), m1=7, m2=5, ap1=2, ap2=5, 
                 pvec=None):
        r,s,t,p = scores
        self.pstar = float(t - s) / (t - s + r - p)
        self.passive = collections.deque(maxlen=m1)
        self.reactive = collections.deque(maxlen=m2)
        self.state = START
        self.step = 0
        self.monitor = False
        self.ap1 = ap1
        self.ap2 = ap2
        self.p_sp = self.p_sr = 0.

    def next_move(self, epsilon=0.05):
        if len(self.passive) < self.ap1: # passive EMPTY
            return 'C'
        if self.state == EXPECT and self.step == 1: # always cooperate 2nd step
            return 'C'
        elif self.p_sp < self.pstar:
            self.state = EXCLUDE
            return 'D'
        elif self.p_sr >= self.pstar and self.last_move == 'DD':
            self.state = EXPECT
            self.step = 0
            return 'C'
        else:
            self.state = ENACT
            return self.last_move[1]

    def print_state(self,
                    states=('init', 'enact', 'expect', 'exclude')):
        print 'state: %s, %s, %s, %0.3f, %0.3f, %d' \
            % (states[self.state], 
               ''.join([str(i) for i in self.passive]), 
               ''.join([str(i) for i in self.reactive]), 
               self.p_sp, self.p_sr, self.step)

    def save_outcome(self, last_move):
        self.last_move = last_move
        if self.state != EXPECT or self.step != 0:
            if last_move[0] == last_move[1]:
                self.passive.append(1)
            else:
                self.passive.append(0)
            self.p_sp = float(sum(self.passive)) / len(self.passive)
        if self.state == EXPECT and self.step == 1:
            if last_move[1] == 'C':
                self.reactive.append(1)
                self.monitor = True
            else:
                self.reactive.append(0)
        elif self.monitor and last_move[1] == 'D':
            self.reactive.append(-1)
            self.monitor = False
        if len(self.reactive) >= self.ap2:
            self.p_sr = float(sum(self.reactive)) / len(self.reactive)
        else: # reactive EMPTY
            self.p_sr = 1.
        self.step += 1


