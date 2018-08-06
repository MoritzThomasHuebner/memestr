import matplotlib.pyplot as plt
import gwmemory
import numpy as np

fig = plt.figure(figsize=(12, 4))

q = 1
S1 = [0, 0, 0]
S2 = [0, 0, 0]

colours = ['r', 'b', 'g', 'k']
model = 'IMRPhenomD'

gamma = gwmemory.angles.load_gamma()

wave = gwmemory.waveforms.Approximant(model, q, MTot=60, times=times)
total = dict()
oscillatory, _ = wave.time_domain_oscillatory(inc=np.pi / 2, pol=0.6)
memory, test = wave.time_domain_memory(inc=np.pi / 2, pol=0.6, gamma_lmlm=gamma)

for key in oscillatory.keys():
    total[key] = oscillatory[key] + memory[key]
    plot(test, total[key])
plt.show()
plt.close()