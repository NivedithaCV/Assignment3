import exam_utilities as ul
import numpy as np
import matplotlib.pyplot as plt
#boundary
potential = np.zeros((100, 100))
potential[:, 0] = 1

potential = ul.laplace(potential)

edge= np.linspace(0, 1, 100)
xv,yv= np.meshgrid(edge,edge)
plt.contour(xv,yv,potential,400)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# solution
#Q3_3.png
