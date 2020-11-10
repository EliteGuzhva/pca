import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from util import *
from data_loader import DataLoader

DATASET_IDX = 3
CORR_THRESHOLD = 0.9

if DATASET_IDX == 1:
    DATASET = 'breast_cancer'
    STONE_N_COMPONENTS = 8
    PCA_N_COMPONENTS = 7
elif DATASET_IDX == 2:
    DATASET = '60'
    STONE_N_COMPONENTS = 40
    PCA_N_COMPONENTS = 30
elif DATASET_IDX == 3:
    DATASET = 'chinese_mnist'
    STONE_N_COMPONENTS = 34
    PCA_N_COMPONENTS = 35
else:
    exit()

# utility function
def sep():
    print("="*100)

# load data
dl = DataLoader(DATASET, verbose=1)
X, y = dl.load()

sep()
print("DATASET:", DATASET)

# pd.plotting.scatter_matrix(X)

# Create correlation matrix
corr_matrix = X.corr()
sep()
print("Correlation matrix")
print(corr_matrix.head())

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(abs(upper[column]) > CORR_THRESHOLD)]

# Drop features
X.drop(to_drop, axis=1, inplace=True)
sep()
print("Dropped:", len(to_drop), "highly correlated features")

# Scale data
X_scaled = scale_features(X)

# Get the number of components
pca = PCA(n_components=None)
pca.fit(X_scaled)

eigenvalues = pca.explained_variance_
n_eigenvalues = len(eigenvalues)
sum_eigenvalues = sum(eigenvalues)
eigenvalues_count = np.arange(1, n_eigenvalues + 1)
normed_eigenvalues = [ev/sum_eigenvalues for ev in eigenvalues]

sep()
print("Eigenvalues:", eigenvalues)
print("Normalized eigenvalues:", normed_eigenvalues)

sep()
# Kaiser
kaiser_n_components = sum(i > (sum_eigenvalues / n_eigenvalues) for i in eigenvalues)
print("Kaiser:", kaiser_n_components)

# Каменистая осыпь
plt.figure(1)
plt.title("Каменистая осыпь")
plt.plot(eigenvalues_count, eigenvalues, '-o')
stone_n_components = STONE_N_COMPONENTS
print("Stone:", stone_n_components)

# Broken stick
l = []
for i in range(n_eigenvalues):
    li = 0
    for j in range(i, n_eigenvalues):
        li += 1/(j+1)
    li /= n_eigenvalues
    l.append(li)

stick_n_components = 0
for i in range(n_eigenvalues):
    if normed_eigenvalues[i] <= l[i]:
        break
    else:
        stick_n_components += 1

print("Broken stick:", stick_n_components)

var = pca.explained_variance_ratio_
var_percantage = np.cumsum(var * 100)
plt.figure(2)
plt.title("Variance percentage")
plt.plot(eigenvalues_count, var_percantage, '-o')
plt.text(eigenvalues_count[kaiser_n_components - 1],
         var_percantage[kaiser_n_components - 1] + 1,
         "Kaiser")
plt.text(eigenvalues_count[stone_n_components - 1],
         var_percantage[stone_n_components - 1],
         "Stone")
if stick_n_components != 0:
    plt.text(eigenvalues_count[stick_n_components - 1],
             var_percantage[stick_n_components - 1] - 1,
             "Stick")

# actual pca
pca = PCA(n_components=PCA_N_COMPONENTS)
X_transformed = pca.fit_transform(X_scaled)
sep()
print("Scores")
print(X_transformed[:5])
print("...")

sep()
print("Loadings")
print(pca.components_.T)

E = X_scaled - X_transformed.dot(pca.components_)
sep()
print("Residuals")
print(E.head())
print(E.describe())

plt.show()
