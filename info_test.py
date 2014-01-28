import info

def recognize_self(ncycle=10, epsilon=0.05, klass=info.InfogainStrategy,
                   epsilonMin=1e-6):
    'compute odds and pvals for IP pair to recognize each other'
    if epsilon < epsilonMin:
        epsilonIP = epsilonMin
    else:
        epsilonIP = epsilon
    ip1 = klass()
    ip2 = klass()
    p1 = p2 = 1.
    for i in range(ncycle):
        m1 = info.move_with_error(ip1.next_move(epsilonIP), epsilon)
        m2 = info.move_with_error(ip2.next_move(epsilonIP), epsilon)
        p12, p11 = ip1.save_outcome(m1 + m2, epsilon)
        p21, p22 = ip2.save_outcome(m2 + m1, epsilon)
        assert p12 == p22 and p21 == p11
        p2 *= p12
        p1 *= p21
    #print 'mismatches:', ip2.mismatches, ip1.mismatches
    return p1, ip2.match_pval(epsilonIP), p2, ip1.match_pval(epsilonIP)

def recognize_other(pvec=None, ncycle=10, epsilon=0.05, 
                    klass=info.InfogainStrategy,
                    klass2=info.MarkovStrategy,
                    epsilonMin=1e-6):
    'compute odds and pval for IP to recognize Markov player'
    if epsilon < epsilonMin:
        epsilonIP = epsilonMin
    else:
        epsilonIP = epsilon
    ip1 = klass()
    mp2 = klass2(pvec)
    p2 = 1.
    for i in range(ncycle):
        m1 = info.move_with_error(ip1.next_move(epsilonIP), epsilon)
        m2 = info.move_with_error(mp2.next_move(epsilon), epsilon)
        p12, p11 = ip1.save_outcome(m1 + m2, epsilon)
        mp2.save_outcome(m2 + m1)
        p2 *= p12
    #print 'mismatches:', ip1.mismatches
    return p2, ip1.match_pval(epsilonIP)

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
    history2 = [info.swap_moves(g) for g in history]
    state2 = state_from_history(history2)
    m11 = info.infogain_move(state, history)
    m12 = info.opponent_infogain_move(state2, history)
    m22 = info.infogain_move(state2, history2)
    m21 = info.opponent_infogain_move(state, history2)
    assert m11 == m21 and m12 == m22

def test_infogain_moves(history=['CC', 'CC', 'DC', 'DD', 'DD', 'CD', 'CD']):
    'test for old opponent_infogain_move() bug'
    check_infogain_moves(history)

def test_self_recognition(ntrial=1000, epsilon=0.05, minP=0.01):
    'check that false negative rate is below expected threshold'
    fn = 0
    for i in range(ntrial):
        p1, pval1, p2, pval2 = recognize_self(epsilon=epsilon)
        #print 'odds: %0.2f, %0.2f:  %0.2f, %0.2f' % (p1,p2, pval1, pval2)
        if p1 < minP:
            fn += 1
        if p2 < minP:
            fn += 1
    p = fn / (2. * ntrial)
    assert p <= minP * 2.

