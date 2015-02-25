import pstats


p = pstats.Stats('profiles/profile')

p.sort_stats('time').print_stats(5)

