# Importer les bibliothèques
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 定义英文和法语文本内容
content = {
    "english": {
        "title": "Language Selection in Streamlit",
        "language_selection": "Please select a language:",
        "option_english": "English",
        "option_french": "French",
    },
    "french": {
        "title": "Sélection de la langue dans Streamlit",
        "language_selection": "Veuillez sélectionner une langue:",
        "option_english": "Anglais",
        "option_french": "Français",
    },
}


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
    st.title("Prédiction du résultat de survie d'un passager personnalisé au Titanic")
    st.image("images/100_anniversary_titanic.jpg", caption="Naufrage du Titanic")
    st.markdown('Projet L2I1: Machine Learning from Disaster')
    st.markdown('Veuillez définir les informations suivantes pour prédire le résultat de survie de ce passager.')

    # Chargement de données
    train = load_data()
    test = pd.read_csv("Data_from_Kaggle/test.csv")
    features_drop=['Name','Ticket','Cabin','PassengerId']
    test=test.drop(features_drop, axis=1)
    
    # entraîner le modèle
    model = train_model(train)

    # définir les données de ce passager
    pclass = st.selectbox('Pclass', [1, 2, 3])
    sex = st.selectbox('Sex', ['male', 'female'])
    age = st.slider('Age', 0, 100, 25)
    sibsp = st.slider('Sibsp', 0, 8, 0)
    parch = st.slider("Parch", 0, 10, 0)
    fare = st.slider("Fare", 0.0, 600.0, 50.0)
    embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])

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
    
    if last_prediction == 1:
        st.success("Résultat：survivant")
    else:
        st.error("Résultat：non survivant")

# démarrer le programme
if __name__ == '__main__':
    # 用户选择语言
    selected_language = st.sidebar.selectbox(
        "Select a language / Sélectionnez une langue",
        options=["english", "french"],
        format_func=lambda x: "English" if x == "english" else "Français",
    )
    # 根据所选语言显示标题
    st.title(content[selected_language]["title"])

    # 根据所选语言显示其他文本内容
    st.write(content[selected_language]["language_selection"])
    st.write(content[selected_language]["option_english"])
    st.write(content[selected_language]["option_french"])
    app_ui()
