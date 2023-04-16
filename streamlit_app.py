# Importer les bibliothèques
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Chargement de train 
@st.cache_resource
def load_data():
    data = pd.read_csv("train_fini.csv")
    return data

def preprocess_data(test):
    #Age
    test['Age'] = test['Age'].fillna(test['Age'].median())
    test['Age'] = (test['Age']-test['Age'].mean())/test['Age'].std()
    
    #Sex
    test.loc[test.Sex=="male",'Sex']='0'
    test.loc[test.Sex=="female",'Sex']='1'
    
    #Embarked
    test.loc[test.Embarked=="C",'Embarked']='1'
    test.loc[test.Embarked=="S",'Embarked']='2'
    test.loc[test.Embarked.isna(),'Embarked']='2'
    test.loc[test.Embarked=="Q",'Embarked']='3'
    
    #Fare
    test['Fare'] = test['Fare'].fillna(test['Fare'].median())
    test['Fare'] = (test['Fare']-test['Fare'].mean())/test['Fare'].std()
    
    return test



# entraîner le modèle
def train_model(data):
    X = data.drop("Survived", axis=1)
    y = data["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
    model = RandomForestClassifier(n_estimators=160,max_depth=13)
    model.fit(X_train,y_train)
    return model

# Créer l'interface utilisateur de l'application
def app_ui():
    st.title("Prédiction de survie d'un passager personnalisé au Titanic")
    st.markdown('Veuillez définir les informations suivantes pour prédire le résultat de survie de ce passager.')

    # Chargement de données
    train = load_data()
    test = pd.read_csv("Data_from_Kaggle/test.csv")
    features_drop=['Name','Ticket','Cabin','PassengerId']
    test=test.drop(features_drop, axis=1)
    
    # entraîner le modèle
    model = train_model(train)

    # définir les données de ce passager
    pclass = st.selectbox('船票等级', [1, 2, 3])
    sex = st.selectbox('性别', ['male', 'female'])
    age = st.slider('年龄', 0, 100, 25)
    sibsp = st.slider('兄弟姐妹/配偶数量', 0, 8, 0)
    parch = st.slider("父母/子女数量", 0, 10, 0)
    fare = st.slider("船票价格", 0.0, 600.0, 50.0)
    embarked = st.selectbox('登船港口', ['C', 'Q', 'S'])

    # créer un tableau de données de ce passager
    features = pd.DataFrame([
        {'Pclass': pclass,
         'Sex': sex,
         'Age': age,
         'SibSp': sibsp,
         'Parch': parch,
         'Fare': fare,
         'Embarked': embarked}
    ])
    
    # concaténation de test avec les données de ce passager
    all_passengers = pd.concat([test, features])
    all_passengers = preprocess_data(all_passengers)
    
    # faire la prédiction avec le Random Forest
    prediction = model.predict(all_passengers)
    last_prediction = prediction[-1]
    
    # Afficher le résultat prévu
    if st.button("Faire la prédiction"):
        st.write("Voici les données de ce passager")
        st.write("Pclass：", features['Pclass'][0])
        st.write("Sex：", features['Sex'][0])
        st.write("Age：", features['Age'][0])
        st.write("Sibsp：", features['SibSp'][0])
        st.write("Parch：", features['Parch'][0])
        st.write("Fare：", features['Fare'][0])
        st.write("Embarked：", features['Embarked'][0])
    
    if last_prediction == 1:
        st.success("Résultat：survivant")
    else:
        st.error("Résultat：non survivant")

# démarrer le programme
if __name__ == '__main__':
    app_ui()
