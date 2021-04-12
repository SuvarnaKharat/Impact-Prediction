#IMPACT PREDICTION 

import pickle
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix,f1_score

# Loading the data
Incidents_service = pd.read_excel("C:\\Users\\mona\\Desktop\\DATA SCIENCE\\DS Project\\Shruti Project\\Incidents_service.xlsx")
pd.options.display.max_columns = None #Helps to see all column names in the Dataset
Incidents_service.head(10) 
Incidents_service.dtypes
Incidents_service.shape
Incidents_service.isnull().sum()
Incidents_service.describe()

#import the packages 
from pandas_profiling import ProfileReport

# Run the profile report 
profile = ProfileReport(Incidents_service, title='Pandas Profiling Report',explorative=True) 

# Save the report as html file 
profile.to_file(output_file="pandas_profiling_P31.html") 

invalid = lambda x:sum(x=="?")/len(x)
Incidents_service.apply(invalid)
#"problem_id"  and "change request" has highest number of missing values which are represented by '?'
#Hence, we can eliminate those columns.

# Column 'ID_status' has some mis-interpreted values with entry- '-100'
Incidents_service.ID_status.value_counts()

#Hence, we can drop those entries as it won't affect our model
Incidents_service = Incidents_service[Incidents_service.ID_status != -100]
Incidents_service.ID_status.value_counts()

# ID column has multiple(different ID_status) and duplicate(with same ID_status) entries
#Hence, we are removing rows with duplicate entries w.r.t. columns -'ID','ID_status','updated_at'
# After removing duplicates we can drop 'ID' column 
Incidents_service=pd.merge(Incidents_service, Incidents_service.groupby(['ID','ID_status'])['updated_at'].max(),on=['ID','ID_status','updated_at'])
Incidents_service.shape
Incidents_service.head(10)

# Getting list of columns 
list(Incidents_service.columns)

sns.heatmap(Incidents_service.corr(),annot=True)

# Correlation matrix 
Incidents_service.corr()

#Renaming Target Varibale entries
#Incidents_service['impact'].replace({"1 - High": "High", "2 - Medium": "Medium","3 - Low": "Low"}, inplace=True)
#print(Incidents_service['impact'])

X = Incidents_service[['ID','ID_status','count_reassign','count_opening','count_updated','location','category_ID','user_symptom']]
y = Incidents_service[['impact']]

categorical_features_indices = np.where(X.dtypes != np.float)[0]
categorical_features_indices

# Splitting data into train and test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=42)
y_train.impact.value_counts()
X_train.columns

#Model Building - Catboost
kf=StratifiedKFold(n_splits=5)
for train_index ,test_index in kf.split(X_train,y_train):X_train1,x_test1,y_train1,y_test1 =X_train.iloc[train_index],X_train.iloc[test_index],y_train.iloc[train_index],y_train.iloc[test_index]
clf=CatBoostClassifier(n_estimators = 500, verbose=False)
clf.fit(X_train,y_train,cat_features = categorical_features_indices)
 
pred=clf.predict(X_test)
print(classification_report(pred,y_test))

pred1=clf.predict(X_train)
print(classification_report(pred1,y_train))

F1_score=f1_score(pred,y_test,average='micro')
F1_score

confusion_matrix(y_test,pred)

sns.heatmap(confusion_matrix(y_test,pred),annot=True,cmap='Blues',xticklabels=['High', 'Low ','Medium'],yticklabels=['High', 'Low ','Medium'],fmt='g')

#Feature Importance Graph
clf.feature_importances_ 

plt.figure(figsize=(18,7))
for i in range(len(clf.feature_importances_)):print('Feature %d: %f' % (i, clf.feature_importances_[i]))
    
# plot the scores
plt.bar([i for i in X.columns], clf.feature_importances_)
plt.show()

#saving model to pickle
pickle.dump(clf,open('Impact_Prediction.pkl','wb'))
