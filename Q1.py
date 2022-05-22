import exam_utilities as ul
import math
import numpy as np

u=[3,572,16381,10000]#seed,a,m,N
a,b=[0,1]

def func(x):
    return np.exp(-x**2)

def gfunc(x):
    return -math.log(1-(math.exp(1)-1)/math.exp(1)*x/16381)

print("Integration without sampling",ul.montecarlo_int(u,10000,func,a,b))

print("Integration with important sampling",ul.with_sampling(u,10000,gfunc))

#solution
#Integration without sampling 0.7461492957151006
#Integration with important sampling 0.7469775831410292
