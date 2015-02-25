import pstats
p = pstats.Stats('profile/profile')
p.sort_stats('time').print_stats(10)

