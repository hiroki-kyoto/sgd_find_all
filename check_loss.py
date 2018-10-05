import numpy as np
import matplotlib.pyplot as plt

log_file = open('./log.txt', 'rt')
line = log_file.readline()
loss = []
while len(line)>8:
    loss.append(float(line[-8:]))
    line = log_file.readline()
loss = np.array(loss)
loss = np.reshape(loss, [-1, 10])
loss = np.mean(loss, axis=1)
plt.plot(loss)
plt.show()
