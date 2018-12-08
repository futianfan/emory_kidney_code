
import numpy as np
import matplotlib.pyplot as plt

# Create Arrays for the plot
materials = ['LR', 'DT', 'NN', 'NN+DT']
x_pos = np.arange(len(materials))

CTEs = [32.8, 1.93, 49.3, 135.8]
error = [4.3, 0.12, 3.4, 6.8]
### Heart Failure

fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Run time (seconds)')
ax.set_xticks(x_pos)
ax.set_xticklabels(materials)
ax.set_title('')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('./data/bar_runtime.png')
plt.show()


