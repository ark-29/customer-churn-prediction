#READING DATASET

import pandas as pd
data= pd.read_csv("C:\\Users\\hp\\Downloads\\customer_churn\\WA_Fn-UseC_-Telco-Customer-Churn.csv")
data.head()

data.info()

# DATASET_DESCRIBE
data.describe()

# NULL VALUES
print(data.isnull().sum())

#DROP DUPLICATES
data=data.drop_duplicates()

#DATA TYPES OF COLUMNS
data.dtypes

#INITIAL 
data['Churn'].value_counts()

#USELESS COLUMN
data=data.drop('customerID',axis=1)
data

#CONVERTING INTO NUMERIC
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)

data["TotalCharges"].fillna(
    data["TotalCharges"].median(),
    inplace=True
)

#AFTER CHANGING 
data.dtypes

#ONLY UNIQUE COLUMN (WITH OUT NO/YES)
data["gender"].unique()

#ENCODING IT INTO 0 AND 1
data['gender']=data['gender'].map({"Male":0,"Female":1})
data.head()

#ENCODING NO/YES COLUMNS INTO 0 AND 1:
for i in ["Partner","Dependents","PhoneService","PaperlessBilling"
         ,"Churn"]:
    data[i]=data[i].map({"No":0,"Yes":1})

#CREATING DUMMIES FOR VALUE OF A COLUMN
data = pd.get_dummies(data, drop_first=True)

#CONVERTING ALL DUMMIES(BOOLS) INTO 0 AND 1:
bool_cols=data.select_dtypes(include='bool').columns
data[bool_cols]=data[bool_cols].astype(int)

#GROUPING VALUES OF TENURE VALUES:
def tenure_group(t):
    if t <= 12:
        return "New"
    elif t <= 36:
        return "Medium"
    else:
        return "Loyal"
data['tenure_groups']=data['tenure'].apply(tenure_group)


#DUMMIES FOR TENURE_GROUPS:
data=pd.get_dummies(data,columns=['tenure_groups'],drop_first=True)

#CONVERTING TENURE_GROUPS INTO 0 AND 1 FROM DUMMIES:
bool_cols=data.select_dtypes(include='bool').columns
data[bool_cols]=data[bool_cols].astype(int)


#SEPERATING DATA SET INTO INPUT AND OUTPUT VALUES:
X=data.drop("Churn",axis=1) #INPUT-INDEPENDENT
y=data["Churn"] #OUTPUT-DEPENDENT

#DATA SPLITTING: 80-TRAIN 20-TEST
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#MODEL:
from sklearn.ensemble import RandomForestClassifier as rfc

model = rfc(
    n_estimators=300, NO OF TREES
    max_depth=10, LEVELS OF A TREE
    min_samples_split=5, NO OF MIN SPLITS TO GET RESULT
    min_samples_leaf=2, NO OF LEAFS OF A TREE
    random_state=42, RANDOMNESS OF DATA SELECTION
    class_weight='balanced' DATA SHOULD BE BALANCED
)

#FITTING DATA INTO MODEL:
model.fit(X_train,y_train)

#PREDICT FOR EXISTING INPUT VALUES:
y_pred=model.predict(X_test)


#METRICS AND PERFORMANCE OF MODEL:
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))


#CONFUSION MATRIX:
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)





