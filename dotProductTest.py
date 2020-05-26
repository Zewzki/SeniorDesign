import numpy as np
import time

#l1 = [18.034285, 22.724102, 15.124005, 12.804266, 22.285585, -81.51105]
#l2 = [25.769325, 23.096819, 23.794834, 27.170263, 30.332172, 37.93327]

#dot = np.dot(l1[0:4], l2[0:4])

#print(dot)


p1 = (254, 110)
p2 = (11, 66)

t1 = time.time()
slope = (p1[1] - p2[1]) / (p1[0] - p2[0])
deltaT1 = time.time() - t1

t2 = time.time()
deg = np.arctan((p1[1] - p2[1]) / (p1[0] - p2[0]))
deltaT2 = time.time() - t2

print(deltaT1)
print(deltaT2)
