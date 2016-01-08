import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import decomposition
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing


res_dict = pd.read_csv("compare_data.csv")

np_points = res_dict.select_dtypes(exclude=["object", "int64"])

np_points = np_points.as_matrix()
np_points = np.nan_to_num(np_points)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(np_points)


pca = decomposition.PCA()
pca.fit(x_scaled)

plt.figure(1, figsize=(4, 3))
plt.clf()

plt.plot(pca.explained_variance_[:50], linewidth=2)
plt.xlabel('components')
plt.ylabel('variance')

###############################################################################
# Prediction
'''
n_components = [20, 40, 64]
Cs = np.logspace(-4, 4, 3)

#Parameters of pipelines can be set using  separated parameter names:

estimator = GridSearchCV(pipe,
                         dict(pca__n_components=n_components,
                              logistic__C=Cs))
estimator.fit(X_digits, y_digits)

plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
'''
plt.show()