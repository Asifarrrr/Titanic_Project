import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
#____________________________________ Cleaning Data ____________________________________
df = None

try:
    df = pd.read_csv(r"C:\Users\pc\PycharmProjects\WelcomeScreen\Mini Projects\Titanic_Project\Titanic-Dataset.csv")
except FileNotFoundError:
    pass

df = df.drop(columns = ['Cabin','PassengerId'])

df['Age'] = df['Age'].fillna(df['Age'].mean()).astype(int)

text = ['Name','Sex','Ticket','Embarked']
for col in text:
    df[col] = df[col].str.strip()

df[['Last_Name','Rest']] = df['Name'].str.split(',',n = 1,expand = True)
df['Last_Name'] = df['Last_Name'].str.strip()
df['Rest'] = df['Rest'].str.strip()

df[['Title','First_Name']] = df['Rest'].str.split('.',n = 1,expand = True)
df['Title'] = df['Title'].str.strip()
df['First_Name'] = df['First_Name'].str.strip()

df = df.drop(columns = ['Name','Rest'])

col = df.pop('Last_Name')
df.insert(3,'Last_Name',col)

col = df.pop('First_Name')
df.insert(4,'First_Name',col)

df['Title'] = df['Title'].replace({
    'Miss': 'Mrs',
    'Master': 'Mr',
    'Mlle': 'Mrs',
    'Mme': 'Mrs',
    'Ms': 'Mrs',
    'Lady': 'Mrs',
    'Sir': 'Mr',
})

df['First_Name'] = df['First_Name'].str.replace(r'\([^(]*\)','',regex = True)
df['First_Name'] = df['First_Name'].str.replace(r'"[^"]*"','',regex = True)
df['First_Name'] = df['First_Name'].str.strip()

df['Fare'] = df['Fare'].round(3)

df['Sex'] = df['Sex'].replace({
    'male': '1',
    'female': '0'
})
df['Sex'] = df['Sex'].astype(int)

df = df.dropna(subset = 'Embarked')
df = pd.get_dummies(df,columns = ['Embarked','Title','Pclass'],dtype = int)

df = df.rename(columns = {'Title_the Countess': 'Title_the_Countess'})
#____________________________________ Model Ai (Logistic Regression) ____________________________________
x = df.drop(columns = ['Survived','Last_Name','First_Name','Ticket'])
y = df['Survived']

x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size = 0.22,
    random_state = 1,
)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression(class_weight = 'balanced')
model.fit(x_train,y_train)

joblib.dump(model,'titanic_model.pkl')
joblib.dump(scaler,'scaler.pkl')
joblib.dump(df,'data_frame.pkl')
joblib.dump(list(x.columns),'model_columns.pkl')