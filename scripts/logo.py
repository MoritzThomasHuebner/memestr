import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy.misc

f = imread(fname='ozgrav_memory/original.png')

for i in range(2400, 4766):
    f[:, i, :] = np.roll(f[:, i, :], shift=-600)

for i in range(1800, 2400):
    f[:, i, :] = np.roll(f[:, i, :], shift=(1800-i - (1800-i) % 4))
    # f[:, i, :] = np.roll(f[:, i, :], shift=-400)


scipy.misc.imsave('ozgrav_memory/new.png', f)

plt.imshow(f)
plt.show()
# plt.savefig('ozgrav_memory/new.png')
plt.clf()
