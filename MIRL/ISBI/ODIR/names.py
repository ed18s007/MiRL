import pandas as pd 
filename = 'ODIR-5K_Training_Annotations.csv'
data = pd.read_csv(filename)
print(data.head())

data_copy = data

normal = data_copy[data_copy.N == 1]
print(normal.head())
normal.to_csv('normal.csv')

diabetes = data_copy[data_copy.D == 1]
print(diabetes.head())
diabetes.to_csv('diabetes.csv')

glaucoma = data_copy[data_copy.G == 1]
print(glaucoma.head())
glaucoma.to_csv('glaucoma.csv')

cataract = data_copy[data_copy.C == 1]
print(cataract.head())
cataract.to_csv('cataract.csv')

# Age-related macular degeneration
AMD = data_copy[data_copy.A == 1]
print(AMD.head())
AMD.to_csv('AMD.csv')

hypertension = data_copy[data_copy.H == 1]
print(hypertension.head())
hypertension.to_csv('hypertension.csv')

myopia = data_copy[data_copy.M == 1]
print(myopia.head())
myopia.to_csv('myopia.csv')

other = data_copy[data_copy.O == 1]
print(other.head())
other.to_csv('other_abnormalities.csv')