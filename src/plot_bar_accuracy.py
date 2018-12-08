
import numpy as np
import matplotlib.pyplot as plt

# Create Arrays for the plot
materials = ['LR', 'DT', 'NN', 'NN+DT']
x_pos = np.arange(len(materials))


CTEs = [0.8223, 0.8055, 0.8282, 0.8324]
error = [0.0009, 0.0000, 0.0083, 0.0093]
### Heart Failure



fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('AUC on test set')
ax.set_xticks(x_pos)
ax.set_xticklabels(materials)
ax.set_title('')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('./data/bar_accuracy.png')
plt.show()


