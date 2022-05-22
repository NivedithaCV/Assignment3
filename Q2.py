import numpy as np
import exam_utilities as ul
import matplotlib.pyplot as plt

def Equation(y, r, V, E):
    psi, phi = y
    df = [phi, (V-E)*psi]
    return np.array(df)


E_ipw = [1.0, 100.0]
k=[0,2]

Energy, eigen, x_ = ul.qiw_shoot(E_ipw, k[0],Equation)
print('Ground state energy:', Energy)
Energy1, eigen1, x_1 = ul.qiw_shoot(E_ipw, k[1],Equation)
print('1st excited state energy:  ', Energy1)


plt.plot(x_, eigen,label='Eigenstate : % s' % (0, ))
plt.plot(x_1, eigen1,label='Eigenstate : % s' % (1, ))
plt.legend()
plt.xlabel('x')
plt.ylabel(r'$\psi$')
plt.show()

#Output
 #Ground state energy: 9.86960456126392
#1st excited state energy:   39.47842784472754
#plot Q2.png
