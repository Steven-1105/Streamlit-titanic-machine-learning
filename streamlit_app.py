# Importer les bibliothèques
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 定义英文和法语文本内容
content = {
    "english": {
        "title": "Predicting the survival results of personalised Titanic passenger",
        "question": "Please define the following information to predict the survival result of this passenger:",
        "image": "Sinking of the Titanic",
        "result_1": "Result：surviving",
        "result_2": "Result：non-surviving",      
#         "sidebar_1": """
#                         **Pclass (Passenger Class)** is the socio-economic status of the passenger and it is a categorical ordinal feature which has 3 unique values (1, 2 or 3):

#                         - **1 = Upper Class**
#                         - **2 = Middle Class**
#                         - **3 = Lower Class** """,
#         "sidebar_2": "Sibsp: the total number of the passenger's siblings and spouse",
#         "sidebar_3": "Parch: the total number of the passenger's parents and children",
#         "sidebar_4": "C = Cherbourg",
#         "sidebar_5": "Q = Queenstown",
#         "sidebar_6": "S = Southampton",
        "Name":"Passenger's Name",
        "Pclass":"Passenger Class",
        "Sibsp":"Total number of the passenger's siblings and spouse",
        "Parch":"Total number of the passenger's parents and children",
        "Fare":"Price of ticket",
        "Embarked":"Port of embarkation",
        "Choix":"Choose a window",
        "Prediction":"Would you survive?",
        "title_wiki":"Wiki of our project: Machine Learning from Disaster",
        "About":"About us",
        "About_1":"to be countinue :)",
    },
    "french": {
        "title": "Prédiction du résultat de survie d'un passager personnalisé au Titanic",
        "question": "Veuillez définir les informations suivantes pour prédire le résultat de survie de ce passager:",
        "image": "Naufrage du Titanic",
        "result_1": "Résultat：survivant",
        "result_2": "Résultat：non survivant",
#         "sidebar_1" : """ 
#                         **Pclass (Passenger Class)** est le statut socio-économique du passager et il s'agit d'une caractéristique ordinale catégorique qui a 3 valeurs uniques (1, 2 ou 3) :
                        
#                         - **1 = classe supérieure**
#                         - **2 = classe moyenne**
#                         - **3 = classe inférieure** """,
#         "sidebar_2": "Sibsp: le nombre total des frères et sœurs et du conjoint du passager",
#         "sidebar_3": "Parch: le nombre total de parents et d'enfants du passager\n",
#         "sidebar_4": "C = Cherbourg",
#         "sidebar_5": "Q = Queenstown",
#         "sidebar_6": "S = Southampton",
        "Name" : "Prenom du passager",
        "Pclass":"Passenger Class",
        "Sibsp":"Nombre total de frères et soeurs et de conjoint du passager",
        "Parch":"Nombre total de parents et d'enfants du passager",
        "Fare":"Prix du billet",
        "Embarked":"Port d'embarquement",
        "Choix":"Choisir une fenêtre",
        "Prediction":"Survivrez-vous?",
        "title_wiki":"Wiki de notre projet : Machine Learning à partir d'une catastrophe",
        "About":"A propos de nous",
        "About_1" : "à compléter :)",
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

# Créer le wiki du projet
def show_wiki():
    st.title(content[selected_language]["title_wiki"])
    st.image("images/100_anniversary_titanic.jpg", caption=content[selected_language]["image"])
    st.markdown('Project L2I1: Machine Learning from Disaster')
    st.markdown(content[selected_language]["question"])

# Créer le wiki du projet
def show_about():
    st.markdown(content[selected_language]["About_1"])   
    
# Créer l'interface utilisateur de l'application
def show_prediction():
#     st.sidebar.markdown(content[selected_language]["sidebar_1"])
#     st.sidebar.title(content[selected_language]["sidebar_2"])
#     st.sidebar.title(content[selected_language]["sidebar_3"])
#     st.sidebar.title(content[selected_language]["sidebar_4"])
#     st.sidebar.title(content[selected_language]["sidebar_5"])
#     st.sidebar.title(content[selected_language]["sidebar_6"])
    st.title(content[selected_language]["title"])
    st.image("images/100_anniversary_titanic.jpg", caption=content[selected_language]["image"])
    st.markdown('Project L2I1: Machine Learning from Disaster')
    st.markdown(content[selected_language]["question"])

    # Chargement de données
    train = load_data()
    test = pd.read_csv("Data_from_Kaggle/test.csv")
    features_drop=['Name','Ticket','Cabin','PassengerId']
    test=test.drop(features_drop, axis=1)
    
    # entraîner le modèle
    model = train_model(train)

    # définir les données de ce passager
    name = st.text_input(content[selected_language]["Name"])
    pclass = st.selectbox(content[selected_language]["Pclass"], [1, 2, 3])
    sex = st.selectbox('Sex', ['male', 'female'])
    age = st.slider('Age', 0, 100, 25)
    sibsp = st.slider(content[selected_language]["Sibsp"], 0, 8, 0)
    parch = st.slider(content[selected_language]["Parch"], 0, 10, 0)
    fare = st.slider(content[selected_language]["Fare"], 0.0, 600.0, 50.0)
    embarked = st.selectbox(content[selected_language]["Embarked"], ['C', 'Q', 'S'])

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
        st.success(content[selected_language]["result_1"])
    else:
        st.error(content[selected_language]["result_2"])

# démarrer le programme
if __name__ == '__main__':
    # 用户选择语言
    selected_language = st.sidebar.selectbox(
        "Select a language / Sélectionnez une langue",
        options=["english", "french"],
        format_func=lambda x: "English" if x == "english" else "Français",
    )
    # 在侧边栏创建页面选择器
    pages = ["Wiki",content[selected_language]["About"], content[selected_language]["Prediction"]]
    selected_page = st.sidebar.selectbox(content[selected_language]["Choix"], pages)
    # 根据所选页面显示内容
    if selected_page == "Wiki":
        show_wiki()
    elif selected_page == content[selected_language]["About"]:
        show_about()
    elif selected_page == content[selected_language]["Prediction"]:
        show_prediction()
