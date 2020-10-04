import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestRegressor

'''
#feture importance using RFR


feat = pd.read_csv('../test_std.csv')
X = feat[feat.columns[1:-1]]
y = feat[feat.columns[-1]]

regr = RandomForestRegressor(max_depth=8, random_state=0)
regr.fit(X, y)

importances = regr.feature_importances_
std = np.std([tree.feature_importances_ for tree in regr.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
important_features = pd.DataFrame()
important_features[feat.columns[0]] = feat[feat.columns[0]]
for f in range(X.shape[1]):
    print("%s. feature %d (%f)" % (X.columns[indices[f]], indices[f], importances[indices[f]]))
    if importances[indices[f]]>0:
	    important_features[X.columns[indices[f]]] = X[X.columns[indices[f]]]
important_features[feat.columns[-1]] = feat[feat.columns[-1]]
important_features.to_csv('../important_features.csv', index = False)



# Plot the impurity-based feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(X.shape[1]), importances[indices],
#         color="r", yerr=std[indices], align="center")
# plt.xticks(range(X.shape[1]), indices)
# plt.xlim([-1, X.shape[1]])
# plt.show()

'''
feature = pd.read_csv('../important_features.csv')
feature_seg_map = pd.read_csv('../train_radiomics_feat_seg_map_std.csv')
valid_seg_map = pd.read_csv('../valid_radiomics_feat_seg_map_std.csv')

X = feature_seg_map[feature.columns[1:-1]]
y = feature_seg_map[feature.columns[-1]]
pred = valid_seg_map[feature.columns[1:-1]]
regr = RandomForestRegressor(max_depth=8, random_state=0)
regr.fit(X, y)
prediction = regr.predict(pred)
print(prediction.shape, prediction*1767)
submit_df = pd.DataFrame()
submit_df['Brats20ID'] = valid_seg_map[feature.columns[0]]
submit_df['Survival_days'] = prediction*1767
submit_df.to_csv('../submit_valid_seg_map.csv',header = False,index = False)


