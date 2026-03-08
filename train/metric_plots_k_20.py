import os
import matplotlib.pyplot as plt

def read_metrics(filepath):
    tprs = []
    fprs = []
    thresholds = []
    k = 0
    
    with open(filepath, 'r') as f:
        for line in f:
            if 'k =' in line:
                k = int(line.split()[-1])
            else:
                clean_line = line.strip().rstrip(',')
                clean_line = clean_line.split(', ')
                t_data = clean_line[0]
                tpr_data = clean_line[1]
                fpr_data = clean_line[2]
                
                t_parts = t_data.split()
                tpr_parts = tpr_data.split()
                fpr_parts = fpr_data.split()
                thresholds.append(float(t_parts[-1]))
                tprs.append(float(tpr_parts[-1]))
                fprs.append(float(fpr_parts[-1]))
                
    return (k, thresholds, tprs, fprs)

# first value of metrics tells us how many principal components were used
# 2nd value is the different threshold values
# 3rd and 4th values are the TPR and FPR rates associated to that given threshold
metrics = read_metrics('train/results/k_20_metrics.csv')
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

plt.savefig('train/plots/ROC_Curve_k20.png', dpi=300, bbox_inches='tight')
plt.show()