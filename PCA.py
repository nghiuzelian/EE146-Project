import numpy as np

def PCA(dataset_matrix, k):
    red_cov_mat, X_centered, mean_face = cov_matrix(dataset_matrix)
    eigen_vals, eigen_vects = eigen_decomp(red_cov_mat, k)
    return red_cov_mat, X_centered, mean_face, eigen_vals, eigen_vects

# assumes each row of dataset_matrix is a flattened vector of the grayscale image of a specific face
def cov_matrix(dataset_matrix):
    # computes the mean face over the rows, each row pertains to a face
    mean_face = np.mean(dataset_matrix, axis=0)
    
    # delete later
    # reshapes mean face to 128x128
    # mean_face_img128 = mean_face.reshape((128, 128))
    # mean_face_img512 = cv2.resize(mean_face_img128, (512, 512))
    # cv2.imshow("Mean Face", mean_face_img512.astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # normalizes each face
    X_centered = dataset_matrix - mean_face

    # computes the reduced covariance matrix
    red_cov_mat = X_centered @ X_centered.T
    return red_cov_mat, X_centered, mean_face

# red_cov_mat has dimensions (N x N) where N is the number of faces in our dataset
def eigen_decomp(red_cov_mat, k):
    # calculates the eigen values and vectors
    eigen_vals, eigen_vects = np.linalg.eigh(red_cov_mat)

    # sort them such that they are now in descending order (highest eigen value, and corresponding vector, first)
    idx = np.argsort(eigen_vals)[::-1]
    eigen_vals = eigen_vals[idx]
    eigen_vects = eigen_vects[:, idx]

    # get top k eigen values and vectors
    eigen_vals = eigen_vals[:k]
    eigen_vects = eigen_vects[:, :k]

    return eigen_vals, eigen_vects