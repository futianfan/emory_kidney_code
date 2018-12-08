
import numpy as np
import matplotlib.pyplot as plt

## 4 -> 29
## depth = 5 -> 49 
num_node = [15, 29, 49, 77, 125]
DT = [0.777, 0.801, 0.804, 0.812, 0.807]
DTNN = [0.827, 0.832, 0.828, 0.82804, 0.8276]
NN = [0.8276 for i in range(len(DT))]
LR = [0.8223 for i in range(len(DT))]

plt.plot(num_node, DT, 'r-', label = 'DT')
plt.plot(num_node, DTNN, 'b-', label = 'DTNN')
plt.plot(num_node, NN, 'g-.', label = 'NN')
plt.plot(num_node, LR, 'k:', label = 'LR')

#_, ax = plt.subplots()
#ax.set_ylabel('AUC on test set')
plt.xlabel('Number of DT Nodes')
plt.ylabel('AUC')
plt.legend()
plt.savefig('./data/curve.png')

#plt.show()
