import numpy as np
thresholds = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]

def read_into_2d(filepath):
    matrix_rows = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if 'k =' in line or not line.strip():
                continue
            
            clean_line = line.strip().rstrip(',')
            
            row_values = [float(x) for x in clean_line.split(',')]

            matrix_rows.append(row_values)
            
    return np.array(matrix_rows)

dist_mat = read_into_2d('distances/k_20.csv')
print(f"Matrix Shape: {dist_mat.shape}")

rows = len(dist_mat)
cols = len(dist_mat[0])

with open(f"results/k_20_metrics.csv", "w") as f:
    f.write(f"k = 20\n")
    for t in thresholds:
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(rows):
            for j in range(cols):
                dist = dist_mat[i][j]
                # checks if they should be given access, T = Yes, F = No
                if (dist <= t):
                    # checks if they have access, T = TP (same user), F = FP (different user)
                    if (i == j):
                        tp += 1
                    else:
                        fp += 1
                # if we get here then the user should not be given access
                else:
                    # means they weren't give access when they really should've been
                    if (i == j):
                        fn += 1
                    # means we didn't give them access and that was the correct decision
                    else:
                        tn += 1
        tpr = tp/(tp+fn)
        fpr = fp/(fp+tn)
        if t == thresholds[len(thresholds) - 1]:    
            f.write(f"t = {t}, TPR = {tpr}, FPR = {fpr}")
        else:
            f.write(f"t = {t}, TPR = {tpr}, FPR = {fpr}\n")