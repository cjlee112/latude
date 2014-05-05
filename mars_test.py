import mars

def check_game_sequence(gameSequence, scores=(3., 0., 5., 1.), 
                        ndecimal=2):
    marsPlayer = mars.MarsStrategy(scores)
    for moves, state, passive, reactive, p_sp, p_sr in gameSequence:
        marsMove = marsPlayer.next_move()
        assert marsMove == moves[0]
        assert marsPlayer.state == state
        marsPlayer.save_outcome(moves)
        reg = ''.join([str(i) for i in marsPlayer.passive])
        assert reg == passive
        reg = ''.join([str(i) for i in marsPlayer.reactive])
        assert reg == reactive
        assert round(marsPlayer.p_sp - p_sp, ndecimal) == 0.
        assert round(marsPlayer.p_sr - p_sr, ndecimal) == 0.

def test_random():
    'test MaRS game sequence against random player from Table S1'
    gameSequence = (
        ('CC', mars.START, '1', '', 1., 1.),
        ('CC', mars.START, '11', '', 1., 1.),
        ('CD', mars.ENACT, '110', '', .667, 1.),
        ('DD', mars.EXCLUDE, '1101', '', .750, 1.),
        ('CC', mars.EXPECT, '1101', '', .750, 1.),
        ('CD', mars.EXPECT, '11010', '0', .600, 1.),
        ('DC', mars.EXCLUDE, '110100', '0', .500, 1.),
        ('DD', mars.EXCLUDE, '1101001', '0', .571, 1.),
        ('DD', mars.EXCLUDE, '1010011', '0', .571, 1.),
        ('DC', mars.EXCLUDE, '0100110', '0', .428, 1.),
        ('DC', mars.EXCLUDE, '1001100', '0', .428, 1.),
        ('DD', mars.EXCLUDE, '0011001', '0', .428, 1.),
        ('DC', mars.EXCLUDE, '0110010', '0', .428, 1.),
        ('DD', mars.EXCLUDE, '1100101', '0', .571, 1.),
        ('DD', mars.EXCLUDE, '1001011', '0', .571, 1.),
        ('DD', mars.EXCLUDE, '0010111', '0', .571, 1.),
        ('DD', mars.EXCLUDE, '0101111', '0', .714, 1.),
        ('CD', mars.EXPECT, '0101111', '0', .714, 1.),
        ('CC', mars.EXPECT, '1011111', '01', .857, 1.),
        ('CC', mars.ENACT, '0111111', '01', .857, 1.),
        ('CC', mars.ENACT, '1111111', '01', 1., 1.),
        ('CD', mars.ENACT, '1111110', '01-1', .857, 1.),
        ('DD', mars.ENACT, '1111101', '01-1', .857, 1.),
        ('DC', mars.ENACT, '1111010', '01-1', .714, 1.),
        ('CC', mars.ENACT, '1110101', '01-1', .714, 1.),
        ('CC', mars.ENACT, '1101011', '01-1', .714, 1.),
        ('CD', mars.ENACT, '1010110', '01-1', .571, 1.),
        ('DD', mars.EXCLUDE, '0101101', '01-1', .571, 1.),
        ('DD', mars.EXCLUDE, '1011011', '01-1', .714, 1.),
        ('CC', mars.EXPECT, '1011011', '01-1', .714, 1.),
    )
    check_game_sequence(gameSequence)
