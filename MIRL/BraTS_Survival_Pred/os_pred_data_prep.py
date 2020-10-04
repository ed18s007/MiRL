import numpy as np 
import pandas as pd 



rad_feat = pd.read_csv('/media/bmi/Windows/MICCAI CHALLENGE 2020/BraTs 2020/validation_radiomic_features_full.csv')
rad_feat = rad_feat.sort_values(by = ['Brats20ID'])
surv_data = pd.read_csv('/media/bmi/Windows/MICCAI CHALLENGE 2020/BraTs 2020/survival_evaluation.csv')
# print(rad_feat.shape, surv_data.shape)
feat = []
for pat_id in surv_data['BraTS20ID']:

    try:
        feat.append(rad_feat[rad_feat['Brats20ID'] == pat_id+'.nii.gz'].values[0])
    except: 
        print(pat_id)

df = pd.DataFrame(feat, columns = rad_feat.columns )
# df = df.sort_values(by = ['Brats20ID'])
# df['Survival_days'] = surv_data['Survival_days']
print(df.columns)
df.to_csv('../valid_test.csv',index = False)
df_std = pd.DataFrame()
df_std[df.columns[0]] = [i[:-7] for i in df[df.columns[0]]]
for column in df.columns[1:]:
    print(column)
    if column == 'Brats20ID':
        df_std[column] = df[column]
    if column == 'Survival_days':
        print(np.max(df[column].values))
        df_std[column] = (df[column].values)/np.max(df[column].values)
    else:
        # print(df[column].mean)
        # print(np.mean(df[column].values),np.std(df[column].values))
        try:
            df_std[column] = (df[column].values - np.mean(df[column].values))/np.std(df[column].values)
        except:
            pass
# df_std['Survival_days'] = surv_data['Survival_days']
# df_std.dropna()
df_std.to_csv('../valid_radiomics_feat_seg_map_std.csv',index = False)