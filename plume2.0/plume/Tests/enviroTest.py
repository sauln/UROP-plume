import plumeSimWlcm
import sys

print "se la vie"

#DU, DU_p, V0, D2U0, U0 = plumeSimWlcm.calcConGradDiv(0.75,0.75, 0.30, 0.15, 0.90, -1, 0)
DU, DU_p, V0, D2U0, U0 = plumeSimWlcm.calcConGradDiv(0.60,0.45, 0.45, 0.45, 0.60, -1, 0) 


#print "DU: %s DU_p: %s\nV0: %s D2U0: %s U0: %s" \
#		%(DU, DU_p, V0, D2U0, U0)
