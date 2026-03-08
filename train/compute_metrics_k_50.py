from compute_metrics_k_20 import read_into_2d

thresholds = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]

dist_mat = read_into_2d('distances/k_50.csv')
print(f"Matrix Shape: {dist_mat.shape}")

rows = len(dist_mat)
cols = len(dist_mat[0])

with open(f"results/k_50_metrics.csv", "w") as f:
    f.write(f"k = 150\n")
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