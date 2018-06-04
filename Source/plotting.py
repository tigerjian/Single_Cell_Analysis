import matplotlib.pyplot as plt
import numpy as np
from numpy import ma

c1 = 4.27766925e-21
c2 = 2.96138659e-19
c3 = -1.47075642e-15
c4 = -4.44992382e-14
c5 = 1.78785413e-10
c6 = 2.61024506e-09
c7 = -8.12092127e-06
c8 = -7.46017497e-05
c9 = 1.01274890e+00
c10 = 1.03168232e+00
c11 = 9.99999999e-01
c12 = 1.80580291e-20
c13 = 2.02015814e-19
c14 = -7.59361267e-15
c15 = -7.59361267e-15
c16 = 1.18964802e-09
c17 = 1.68866609e-09
c18 = -8.19203097e-05
c19 = -4.75759918e-05
c20 = 2.04222861e+00
c21 = 1.65638944e+00
c22 = 9.99999925e-01



x, y = np.meshgrid(np.linspace(95000, 105000, 10), np.linspace(18000, 43000, 25))


U = y - (c1 * x**5 + c2 * y**5 + c3 * x**4 + c4 * y**4 + c5 * x**3 + c6 * y**3 + c7 * x**2 + c8 * y**2 + c9 * x + c10 * y + c11)
V = x - (c12 * x**5 + c13 * y**5 + c14 * x**4 + c15 * y**4 + c16 * x**3 + c17 * y**3 + c18 * x**2 + c19 * y**2 + c20 * x + c21 * y + c22)

plt.figure(figsize = (11,15))

#plt.title('Arrows scale with plot width, not view')
Q = plt.quiver(x, y, U, V, units='width')
qk = plt.quiverkey(Q, 0.76, 0.83, 100000, r'$250 um$', labelpos='E',
                   coordinates='figure')
qk.text.set_size(25)

plt.axis('equal')
plt.show()