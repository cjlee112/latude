import info

def recognize_self(ncycle=10, epsilon=0.05, klass=info.InfogainStrategy):
    ip1 = klass()
    ip2 = klass()
    p1 = p2 = 1.
    for i in range(ncycle):
        m1 = info.move_with_error(ip1.next_move(epsilon), epsilon)
        m2 = info.move_with_error(ip2.next_move(epsilon), epsilon)
        p12, p11 = ip1.save_outcome(m1 + m2, epsilon)
        p21, p22 = ip2.save_outcome(m2 + m1, epsilon)
        assert p12 == p22 and p21 == p11
        p2 *= p12
        p1 *= p21
    return p1, p2

def state_from_history(history):
    n = len(history)
    d = {}
    for ctx in ('CC', 'CD', 'DC', 'DD'):
        moves = [history[i + 1][1] for i in range(n - 1)
                 if history[i] == ctx]
        d[ctx] = (len([m for m in moves if m == 'C']), len(moves))
    return d

def check_infogain_moves(history):
    state = state_from_history(history)
    m11 = info.simple_infogain_move(history[-1], state, history)
    m12 = info.opponent_infogain_move(history[-1], state, history)
    history2 = [info.swap_moves(g) for g in history]
    state2 = state_from_history(history2)
    m22 = info.simple_infogain_move(history2[-1], state2, history2)
    m21 = info.opponent_infogain_move(history2[-1], state2, history2)
    assert m11 == m21 and m12 == m22

def test_infogain_moves(history=['CC', 'CC', 'DC', 'DD', 'DD', 'CD', 'CD']):
    check_infogain_moves(history)

def test_self_recognition(ntrial=100, epsilon=0.05):
    for i in range(ntrial):
        p1, p2 = recognize_self(epsilon=epsilon)
        assert p1 >= 1. and p2 >= 1.
