# Importer les bibliothèques
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pandas import DataFrame

# 定义英文和法语文本内容 
# Définir le contenu du texte en anglais et en français
content = {
    "english": {
        "title": "Predicting the survival results of personalised Titanic passenger",
        "question": "Please define the following information to predict the survival result of this passenger:",
        "image_1": "Sinking of the Titanic",
        "image_2": "The cabins of the Titanic",
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
        "wiki_intro":"""
                            ## Introduction
                            The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew. While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.\n
                            In this challenge, we have to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc)."
                     """,
        "wiki_objective":"""
                            ## Objective

                            In this project, we will have access to two similar datasets that include passenger information like name, age, gender, socio-economic class, etc. One dataset is titled 'train.csv' and the other one is titled 'test.csv'.

                            - train.csv will contain the details of a subset of the passengers on board (891 to be exact) and importantly, will reveal whether they survived or not, also known as the “ground truth”. This information will help us to establish a link between passenger information and their survival.
                            - test.csv dataset contains similar information but does not disclose the “ground truth” for each passenger. We will have to find the best models to predict these outcomes with a maximum success rate.

                            We will use all the data available in the 'train.csv' file to predict whether the other 418 passengers on board (found in test.csv) survived.

                            You may find more information about the Kaggle challenge here: [Kaggle Titanic Challenge](https://www.kaggle.com/competitions/titanic/overview)
                         """,
        "About":"About us",
        "About_1":"to be countinue :)",
        "About_2":  """
                        ## Project Members
                        We are 4 students in our second year of Computer Science/Mathematics and Computer Science degree at the University Paris Descartes:
                        
                        - Hongxiang LIN
                        - Matthieu Antonopoulos
                        - Melissa Merabet
                        - Timothé Miel

                        This project is supervised by Mr. Paul Boniol.
                    """,
        "Analyse" : "Data analysis",
        "Analyse_1": """We have 2 files: train.csv and test.csv.\n 
We will analyse the different data available to us in order to determine the main variables affecting the survival of a passenger.
## Available data:

| Variable | Definition | Key |
|:---------:|:----------:|:----:|
| survival |  Survival  |  0 = No, 1 = Yes |
| pclass |  Ticket class |  1 = 1st, 2 = 2nd, 3 = 3rd |
| sex |  Sex | |
| Age |  Age in years |  |
| sibsp |  Number of siblings / spouses aboard the Titanic | |
| parch |  Number of parents / children aboard the Titanic | |
| ticket |  Ticket number | |
| fare |  Passenger fare | |
| cabin |  Cabin number | |
| embarked |  Port of Embarkation |  C = Cherbourg, Q = Queenstown, S = Southampton |

**pclass**: Social status:
- 1st = high
- 2nd = medium
- 3rd = low

**sibsp**: Number of family members (siblings, spouse)

**parch**: Number of parents/children.
""", 
        "Analyse_Embarked": """Through this histogram, we realize that:

- For passengers who embarked from the port of Cherbourg, there were more survivors than deaths (about 75 deaths and 90 survivors)
- For those who embarked from the port of Southampton, there are more deaths than survivors (over 400 deaths and about 230 survivors)
- For those who embarked from the port of Queenstown, there are more deaths than survivors (about 50 deaths and 40 survivors)

Therefore, people who embarked from the port of Cherbourg have survived the most. And the percentage of deaths is higher for those who embarked from the port of Southampton. This may be related to the class of passengers embarking from each port (hypothesis).
""",
        "Analyse_Sex" : "By this histogram, we notice that about 100 men survived against more than 450 deaths; And about a hundred women died against approximately 250 survivors. More than 2/3 of the survivors are women.",
        "Analyse_Age" :"""According to the analysis graph, the following conclusions can be drawn:

- Most of the passengers were between 20 and 40 years old
- Infants and young children (0 to 4 years old) had relatively high survival rates
- Elderly people (60 years old and above) had relatively low survival rates
- Passengers between the ages of 15 and 35 generally had low survival rates

These conclusions indicate that age has a certain influence on the survival rate on the Titanic, especially for infants and young children, who had higher survival rates. The relatively low survival rate for passengers between the ages of 15 and 35 may be related to factors such as their position on the ship, gender, and cabin class. Therefore, passengers of different ages may need to adopt different survival strategies to improve their chances of survival on the Titanic.
        """,
        "Analyse_Cabin": """According to the analysis graph, we can draw the following conclusions:

- Many passengers have lost their cabin information and the survival rate of these passengers is relatively low
- Cabin information is represented by letters, which can represent different positions or levels
- Passengers in categories B, C, D, E and F have a relatively high survival rate of their cabin information
- Category A and G passengers have relatively low survival rates
- Due to the large amount of missing cabin information, the conclusions of this analysis may not be accurate

In summary, cabin information may have a certain influence on passengers' survival rate, and specific cabin categories may be related to the survival rate. However, due to the lack of cabin information, the conclusions may be subject to errors and some imprecision.
        """,

    },
    "french": {
        "title": "Prédiction du résultat de survie d'un passager personnalisé au Titanic",
        "question": "Veuillez définir les informations suivantes pour prédire le résultat de survie de ce passager:",
        "image_1": "Naufrage du Titanic",
        "image_2": "Les cabines du Titanic",
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
        "wiki_intro":"""
                            ## Introduction
                            Le naufrage du Titanic est l'un des naufrages les plus tristement célèbres de l'histoire. Le 15 avril 1912, lors de son voyage inaugural, le RMS Titanic, largement considéré comme "insubmersible", a coulé après avoir heurté un iceberg. Malheureusement, il n'y avait pas assez de canots de sauvetage pour tout le monde à bord, ce qui a entraîné la mort de 1502 des 2224 passagers et membres d'équipage. Bien qu'il y ait eu une part de chance dans la survie, il semble que certains groupes de personnes aient eu plus de chances de survivre que d'autres.\n 
                            Dans ce défi, notre équipe va construire un modèle prédictif qui répond à la question suivante : "Quelles sont les personnes qui ont le plus de chances de survivre ?" en utilisant les données des passagers (nom, âge, sexe, classe socio-économique, etc.).
                     """,
       
        "wiki_objective":"""
                            ## Objectif

                            Afin de réaliser ce projet, nous aurons accès à deux sets de donnés incluant des informations propres aux passagers telles que le nom, l'âge, le sex, classe sociale, etc... Le premier dataset est intitulé 'train.csv' et le second 'test.csv'

                            - train.csv contient les informations d'une partie des passagers à bord du Titanic au moment du naufrage (891 passagers pour être exact). De plus ce dataset contient l'information sur la survie ou non de chaque passagers. On appel cette information la "vérité terrain", elle nous permettra d'entrainer notre modèle et d'établir des liens entre informations des passagers et la survie.
                            - test.csv contient des informations similaires au premier dataset à l'exception près que nous n'aurons aucune informations concernant la survie des passagers ("vérité terrain"). Ainsi nous devrons utiliser différents modèles de Machine Learning pour prédire au mieux la survie de ces 418 passagers du fichier 'test.csv'

                            Pour plus d'informations concernant les modalités du concours Kaggle : https://www.kaggle.com/competitions/titanic/overview
                         """,
        "About":"A propos de nous",
        "About_1" : "à compléter :)",
        "About_2" : """
                        ## Les membres du projet
                        Nous sommes 4 étudiants en 2ème année de licence informatique/mathématique et informatique à l'Unversité Paris Descartes:

                        - Hongxiang LIN
                        - Matthieu Antonopoulos
                        - Melissa Merabet
                        - Timothé Miel

                        Ce projet est encadré par M.Paul Boniol
                    """,
        "Analyse" : "Analyse de données",
        "Analyse_1" :"""On dispose de 2 fichiers : train.csv et test.csv.\n
Nous allons analyser les différentes données à notre disposition afin de déterminer les principales variables jouant sur la survie d'un passager. 

## Données disponibles : 

| Variable  | Définition |  Key | 
|:---------:|:----------:|:----:|
|survival | Survie |  0 = Non, 1 = Oui|
|pclass | Classe du billet |  1 = 1ère, 2 = 2ème, 3 = 3ème|
|sex |  Sexe ||
|Age |  Âge en années|   |
|sibsp |  Nombre de frères/soeurs/époux à bord du Titanic  ||
|parch |  Nombre de parents / enfants à bord du Titanic  ||
|ticket |  Numéro du billet  ||
|fare | Tarif du passager  ||
|cabin |  Numéro de la cabine||  
|embarked |  Port d'embarquement|  C = Cherbourg, Q = Queenstown, S = Southampton|

**pclass**: Statut social :
- 1st = haute
- 2nd = moyenne 
- 3rd = basse

**sibsp**: Nombre de membres de la famille (Frère/soeur, époux/femme)

**parch**: Nombre de parents / enfants.
""",
        "Analyse_Embarked" : """Par cet histogramme, nous réalisons que:

- Pour les passagers ayant embarqué depuis le port de Cherbourg, il y a eu plus de survivants que de morts (environ 75 morts et 90 survivants)
- Pour ceux ayant embarqué depuis le port de Southampton il y a plus de morts que de survivants (plus de 400 morts et environ 230 survivants)
- Pour ceux ayant embarqué depuis le port de Queenstown il y a plus de morts que de survivants (environ 50 morts et 40 survivants)

Ce sont donc les personnes qui ont embarqué depuis le port de Cherbourg qui ont le plus survécu. Et le pourcentage de décès est plus élévé pour ceux qui ont embarqué depuis le port de Southampton. Cela peut être lié à la classe des passagers embarcant dans chaque ports (hypothèse)

""",
        "Analyse_Sex" : "Nous remarquons par cet histograme, qu'environ 100 hommes ont survécu contre plus de 450 morts; Et qu'une centaine de femmes sont décédées contre environ 250 survivantes. Plus de 2/3 des survivants sont des femmes.",
        "Analyse_Age" :"""Selon le graphique d'analyse, les conclusions suivantes peuvent être tirées :

- La plupart des passagers étaient âgés entre 20 et 40 ans
- Les nourrissons et les jeunes enfants (de 0 à 4 ans) ont eu des taux de survie relativement élevés
- Les personnes âgées (de 60 ans et plus) ont eu des taux de survie relativement faibles
- Les passagers âgés de 15 à 35 ans ont généralement eu des taux de survie faibles

Ces conclusions indiquent que l'âge a une certaine influence sur le taux de survie sur le Titanic, en particulier pour les nourrissons et les jeunes enfants, qui ont eu des taux de survie plus élevés. Le taux de survie relativement faible pour les passagers âgés de 15 à 35 ans peut être lié à des facteurs tels que leur position sur le navire, leur sexe et leur classe de cabine. Par conséquent, les passagers de différents âges peuvent avoir besoin d'adopter différentes stratégies de survie pour améliorer leurs chances de survie sur le Titanic.
""",
        "Analyse_Cabin" : """Selon le graphique d'analyse, nous pouvons tirer les conclusions suivantes :

- De nombreux passagers ont perdu leurs informations de cabine et le taux de survie de ces passagers est relativement faible
- Les informations de cabine sont représentées par des lettres, ce qui peut représenter différentes positions ou niveaux
- les passagers des catégories B, C, D, E et F ont un taux de survie relativement élevé de leurs informations de cabine
- les taux de survie des passagers des catégories A et G sont relativement faibles
- En raison du grand nombre d'informations manquantes sur les cabines, les conclusions de cette analyse peuvent ne pas être précises

En résumé, les informations de cabine peuvent avoir une certaine influence sur le taux de survie des passagers, et des catégories de cabine spécifiques peuvent être liées au taux de survie. Cependant, en raison du manque d'informations sur les cabines, les conclusions peuvent être sujettes à des erreurs et à une certaine imprécision.
        """,
    },
}


# Chargement de train 
@st.cache_resource
def load_data():
    data = pd.read_csv("train_fini.csv")
    return data

# Preprocessing de données
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
    st.image("images/titanic_photo.jpg", caption=content[selected_language]["image_1"])
    st.markdown('Project L2I1: Machine Learning from Disaster')
    st.markdown(content[selected_language]["wiki_intro"])
    st.markdown(content[selected_language]["wiki_objective"])
    
# Créer la présentation des membres du projet
def show_about():
    st.markdown(content[selected_language]["About_2"])

# Créer l'Analyse de données
def show_Analyse():
    st.markdown(content[selected_language]["Analyse_1"])
    train = pd.read_csv("Data_from_Kaggle/train.csv")
    # sélectionner le style de seaborn
    sns.set(style="darkgrid")

    # 绘制性别和生还情况的计数图
    # résultat des chiffres relatifs au sexe et à la survie
    fig_sex, ax = plt.subplots()
    sns.countplot(data=train, x='Sex', hue='Survived')

    # 绘制年龄和生还情况的直方图
    # Histogrammes de l'âge et de la survie
    fig_age, ax = plt.subplots()
    sns.histplot(data=train, x='Age', hue='Survived', element='step', kde=True, ax=ax)

    # 绘制登船码头和生还情况的计数图
    # résultat des chiffres relatifs au port d'embarquement et à la survie
    fig_embarked, ax = plt.subplots()
    sns.countplot(data=train, x='Embarked', hue='Survived' )

    # 绘制乘客船舱和生还情况的计数图
    # résultat des chiffres relatifs au cabines et à la survie
    # Fill the missing values in the 'Cabin' column with 'Unknown'
    train['Cabin'].fillna('Unknown', inplace=True)

    # Extract the first letter from each value in the 'Cabin' column
    train['Cabin'] = train['Cabin'].apply(lambda x: x[0])

    # Create a countplot of 'Cabin' vs 'Survived'
    fig_cabin = sns.catplot(x='Cabin', hue='Survived', data=train, kind='count', order=['A','B','C','D','E','F','G'], palette='winter', aspect=2)

    # afficher le plot sur Streamlit
    st.pyplot(fig_sex)
    st.markdown(content[selected_language]["Analyse_Sex"])
    st.pyplot(fig_age)
    st.markdown(content[selected_language]["Analyse_Age"])
    st.pyplot(fig_embarked)
    st.markdown(content[selected_language]["Analyse_Embarked"])
    st.image("images/Cabin_Titanic.webp", caption=content[selected_language]["image_2"])
    st.pyplot(fig_cabin)
    st.markdown(content[selected_language]["Analyse_Cabin"])
    
# Créer l'interface utilisateur de l'application
def show_prediction():
    st.title(content[selected_language]["title"])
    st.image("images/100_anniversary_titanic.jpg", caption=content[selected_language]["image_1"])
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
    # Langues sélectionnées par l'utilisateur
    selected_language = st.sidebar.selectbox(
        "Select a language / Sélectionnez une langue",
        options=["english", "french"],
        format_func=lambda x: "English" if x == "english" else "Français",
    )
    # 在页面内创建页面选择器
    # Créer un sélecteur de page 
    tab_wiki, tab_About, tab_Prediction, tab_Analyse = st.tabs(["Wiki",content[selected_language]["About"], content[selected_language]["Prediction"],content[selected_language]["Analyse"]])
    # 根据所选页面显示内容
    # Afficher le contenu en fonction de la page sélectionnée
    with tab_wiki:
        show_wiki()
    with tab_About:
        show_about()
    with tab_Prediction:
        show_prediction()
    with tab_Analyse:
        show_Analyse()

