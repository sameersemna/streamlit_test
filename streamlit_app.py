import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import joblib
import pickle

# (c) Create a dataframe called df to read the file train.csv.
df=pd.read_csv("train.csv")

# (d) Copy the following code lines to add a title and to create 3 pages called "Exploration", "DataVizualization" and "Modelling" on Streamlit.
st.title("Titanic : binary classification project")
st.sidebar.title("Table of contents")
pages=["Exploration", "DataVizualization", "Modelling"]
page=st.sidebar.radio("Go to", pages)

if page == pages[0] : 
    st.write("### Presentation of data")

    # (c) Display the first 10 lines of df on the web application Streamlit by using the method st.dataframe().
    st.dataframe(df.head(10))
    # (d) Display informations about the dataframe on the Streamlit web application using the st.write() method in the same way as a print and the st.dataframe() method for a dataframe.
    st.write(df.shape)
    st.dataframe(df.describe())
    # (e) Create a checkbox to choose whether to display the number of missing values or not, using the st.checkbox() method.
    if st.checkbox("Show NA") :
        st.dataframe(df.isna().sum())


if page == pages[1] : 
    st.write("### DataVizualization")

    fig = plt.figure()
    sns.countplot(x = 'Survived', data = df)
    st.pyplot(fig)

    fig = plt.figure()
    sns.countplot(x = 'Sex', data = df)
    plt.title("Distribution of the passengers gender")
    st.pyplot(fig)
    fig = plt.figure()
    sns.countplot(x = 'Pclass', data = df)
    plt.title("Distribution of the passengers class")
    st.pyplot(fig)
    fig = sns.displot(x = 'Age', data = df)
    plt.title("Distribution of the passengers age")
    st.pyplot(fig)

    fig = plt.figure()
    sns.countplot(x = 'Survived', hue='Sex', data = df)
    st.pyplot(fig)
    fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
    st.pyplot(fig)
    fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
    st.pyplot(fig)

    # fig, ax = plt.subplots()
    # sns.heatmap(df.corr(), ax=ax)
    # st.write(fig)

if page == pages[2] : 
    st.write("### Modelling")

    # (b) In the Python script streamlit_app.py, remove the irrelevant variables (PassengerID, Name, Ticket, Cabin).
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    # (c) In the Python script, create a variable y containing the target variable. Create a dataframe X_cat containing the categorical explanatory variables and a dataframe X_num containing the numerical explanatory variables.
    y = df['Survived']
    X_cat = df[['Pclass', 'Sex',  'Embarked']]
    X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]

    for col in X_cat.columns:
        X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
    for col in X_num.columns:
        X_num[col] = X_num[col].fillna(X_num[col].median())
    X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
    X = pd.concat([X_cat_scaled, X_num], axis = 1)
    # (g) In the Python script, separate the data into a train set and a test set using the train_test_split function from the Scikit-Learn model_selection package.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    # (h) In the Python script, standardize the numerical values using the StandardScaler function from the Preprocessing package of Scikit-Learn.
    scaler = StandardScaler()
    X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
    X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

    def prediction(classifier):
        if classifier == 'Random Forest':
            clf = RandomForestClassifier()
        elif classifier == 'SVC':
            clf = SVC()
        elif classifier == 'Logistic Regression':
            clf = LogisticRegression()
        clf.fit(X_train, y_train)
        return clf
    # Since the classes are not unbalanced, it is interesting to look at the accuracy of the predictions. Copy the following code into your Python script. It creates a function which returns either the accuracy or the confusion matrix.

    def scores(clf, choice):
        if choice == 'Accuracy':
            return clf.score(X_test, y_test)
        elif choice == 'Confusion matrix':
            return confusion_matrix(y_test, clf.predict(X_test))
        
    choice = ['Random Forest', 'SVC', 'Logistic Regression']
    option = st.selectbox('Choice of the model', choice)
    st.write('The chosen model is :', option)

    clf = prediction(option)
    display = st.radio('What do you want to show ?', ('Accuracy', 'Confusion matrix'))
    if display == 'Accuracy':
        st.write(scores(clf, display))
    elif display == 'Confusion matrix':
        st.dataframe(scores(clf, display))

    joblib.dump(clf, "model.jbl")
    pickle.dump(clf, open("model.pkl", 'wb'))
