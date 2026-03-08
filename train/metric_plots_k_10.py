import matplotlib.pyplot as plt
from train.metric_plots_k_20 import read_metrics

# first value of metrics tells us how many principal components were used
# 2nd value is the different threshold values
# 3rd and 4th values are the TPR and FPR rates associated to that given threshold
metrics = read_metrics('train/results/k_10_metrics.csv')
fpr = metrics[3]
tpr = metrics[2]
thresholds = metrics[1]
k = metrics[0]

plt.figure(figsize=(10, 7))

sc = plt.scatter(fpr, tpr, c=thresholds, cmap='viridis', s=20, edgecolors='none')

for i in range(0, len(thresholds)): 
    plt.annotate(
        f't = {thresholds[i]:.2f}',
        (fpr[i], tpr[i]),
        textcoords="offset points",
        xytext=(5, 5),
        ha='left',
        fontsize=6,
        alpha=0.8
    )

plt.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title(f'ROC Curve Colored by Threshold (k = {k})')
plt.grid(True, alpha=0.2)
plt.savefig('train/plots/ROC_Curve_k10.png', dpi=300, bbox_inches='tight')
plt.show()