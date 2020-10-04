import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import random


# Read Radiomic features as dataframe
df = pd.read_csv("training_radiomic_features.csv") 
print(df.shape)
df = df[df.columns[1:100:]]
print(df.shape)

# Sort the dataframe according to Patient IDs
df = df.sort_values('CPM_RadPath_2019_ID')
# MinMaxScaler standardizes the data values in range 0 to 1
min_max_scaler = preprocessing.MinMaxScaler()
# Standardize dataframe the first column is index and second is Patient Ids so skip these
df.iloc[:,1:] = min_max_scaler.fit_transform(df.iloc[:,1:])
# Just again sorting to make sure dataframe is sorted with Patient IDs
df = df.sort_values('CPM_RadPath_2019_ID')

# Copy the dataframe to make predictions after SVM training
pred = df.copy(deep=True)
# Convert it to list
pred_ls = pred.values.tolist()

# Read Survival info as dataframe
surv_df = pd.read_csv("survival_info.csv") 
# Sort the dataframe according to Patient IDs
surv_df = surv_df.sort_values('Brats20ID')
# Convert it to list
surv_ls = surv_df.values.tolist()

# Initialize final list
final_ls = []
# Initialize counters and count to calculate total features
i,j,cnt = 0,0,0

# This loop combines values from Radiomic Features and Survival Info
# Only those values present in survival info are used as final features
while i<len(pred_ls):
    # If the Patient IDs match Patient Name from survival and Radiomic features are 
    # placed in final_ls.
    if pred_ls[i][0]==surv_ls[j][0]:
        cnt+=1
        temp = surv_ls[j] + pred_ls[i][2:]
        final_ls.append(temp)
        i+=1
        j+=1
    # Check if next Patient in pred_ls is same as in surv_ls
    elif pred_ls[i+1][0]==surv_ls[j][0]:
        i+=1
    # Check if next Patient in surv_ls is same as in pred_ls
    elif pred_ls[i][0]==surv_ls[j+1][0]:
        j+=1
    # Increment counter while both patient IDs match
    else:
        while pred_ls[i][0]!=surv_ls[j][0]:
            i+=1
            if i==len(pred_ls):
                break
print(len(final_ls))
# Declare Lists for three classes patients
low_surv, mid_surv, high_surv = [], [], []
# If the survival days is less than 300 then in low 
# it is in between 300 to 450 then in mid else high
# The radiomic features and just the PatientId is included in these lists
for i in range(len(final_ls)):
    y = final_ls[i][4:]
    if int(final_ls[i][2])<=300:
        tmp = y + [0]
        low_surv.append(tmp)
    elif int(final_ls[i][2])<=450:
        tmp = y + [1]
        mid_surv.append(tmp)
    else:
        tmp = y + [2]
        high_surv.append(tmp)
print(len(low_surv), len(mid_surv), len(high_surv))


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

# k-fold is not exactly cross validation
# Basically I have just shuffled the data randomly five times
# to check the accuracy
k_fold = 1
# split indicates percentage of datapoints for three classes 
split = 0.7
for i in range(k_fold):
    # Train and Valid declaration lists
    # Split data into training and validation sets
    train_reg, valid_reg = [], []
    low, mid, high = len(low_surv),len(mid_surv),len(high_surv)
    random.shuffle(low_surv)
    random.shuffle(mid_surv)
    random.shuffle(high_surv)

    train_reg.extend(low_surv[:int(split*low)])
    valid_reg.extend(low_surv[int(split*low):])

    train_reg.extend(mid_surv[:int(split*mid)])
    valid_reg.extend(mid_surv[int(split*mid):])

    train_reg.extend(high_surv[:int(split*high)])
    valid_reg.extend(high_surv[int(split*high):])

    # Convert it into numnpy
    train_features = np.array(train_reg)
    test_features = np.array(valid_reg)

    # Split it into features and labels
    X_train = train_features[:,:-1]
    Y_train = train_features[:,-1]
    X_test = test_features[:,:-1]
    Y_test = test_features[:,-1]

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_normalized = scaler.transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    print('Train Data Shape', X_train.shape)
    print('Test Data Shape', X_test.shape)
    # print(Y_train)
    # print(Y_test)
    def my_kernel(X, Y):
        return np.dot(X, Y.T)

    print("_"*60)
    print("SVM")

    svm_model_linear = make_pipeline(SVC(kernel = 'rbf'))   
    svm_model_linear.fit(X_train_normalized, Y_train) 
    predict_test = svm_model_linear.predict(X_test_normalized) 
      
    print(confusion_matrix(Y_test,predict_test))
    print(classification_report(Y_test,predict_test))

    print("_"*60)
    print("XGBClassifier")

    xgb = make_pipeline(XGBClassifier(n_estimators=500))
    xgb.fit(X_train_normalized, Y_train)

    predict_test=xgb.predict(X_test_normalized)
    print(confusion_matrix(Y_test,predict_test))
    print(classification_report(Y_test,predict_test))

    print("_"*60)
    print("RandomForestClassifier")
    #Create a Gaussian Classifier
    clf= make_pipeline(RandomForestClassifier(n_estimators=500))
    clf.fit(X_train_normalized, Y_train)
    predict_test=clf.predict(X_test_normalized)
    print(confusion_matrix(Y_test,predict_test))
    print(classification_report(Y_test,predict_test))

    print("_"*60)
    print("KNN Classifier")
    #Create KNN Classifier
    knn = make_pipeline(KNeighborsClassifier(n_neighbors=5))
    #Train the model using the training sets
    knn.fit(X_train_normalized, Y_train)
    predict_test=knn.predict(X_test_normalized)
    print(confusion_matrix(Y_test,predict_test))
    print(classification_report(Y_test,predict_test))

