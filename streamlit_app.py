# 导入必要的库和数据集
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 加载数据集
@st.cache_resource
def load_data():
    data = pd.read_csv("train_fini.csv")
    return data

def preprocess_data(data):
    
    # 缺失值处理
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    
    # 类别特征编码
    le = LabelEncoder()
    data['Sex'] = le.fit_transform(data['Sex'])
    data['Embarked'] = le.fit_transform(data['Embarked'])
    
    #Age
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Age'] = (data['Age']-data['Age'].mean())/data['Age'].std()
    
    #Sex
    data.loc[data.Sex=="male",'Sex']='0'
    data.loc[data.Sex=="female",'Sex']='1'
    data.loc[data.Sex.isna(),'Sex']='0'
    
    #Embarked
    data.loc[data.Embarked=="C",'Embarked']='1'
    data.loc[data.Embarked=="S",'Embarked']='2'
    data.loc[data.Embarked.isna(),'Embarked']='2'
    data.loc[data.Embarked=="Q",'Embarked']='3'
    
    #Fare
    data.loc[data.Fare.isna(), 'Fare'] = 50.0
    data['Fare'] = (data['Fare']-data['Fare'].mean())/data['Fare'].std()
    
    return data



# 训练模型
def train_model(data):
    X = data.drop("Survived", axis=1)
    y = data["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
    model = RandomForestClassifier(n_estimators=160,max_depth=13)
    model.fit(X_train,y_train)
    return model

# 创建应用程序UI
def app_ui():
    st.title('Titanic 生还预测')
    st.markdown('请输入以下信息以预测您的生还结果。')

    # 加载数据集
    data = load_data()

    # 训练模型
    model = train_model(data)

    # 添加输入组件
    pclass = st.selectbox('船票等级', [1, 2, 3])
    sex = st.selectbox('性别', ['male', 'female'])
    age = st.slider('年龄', 0, 100, 25)
    sibsp = st.slider('兄弟姐妹/配偶数量', 0, 8, 0)
    parch = st.slider("父母/子女数量", 0, 10, 0)
    fare = st.slider("船票价格", 0.0, 600.0, 50.0)
    embarked = st.selectbox('登船港口', ['C', 'Q', 'S'])

    # 创建一个特征向量
    features = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    })
    
    features = preprocess_data(features)
    
    # 进行预测
    prediction = model.predict(features)

    # 显示预测结果
    if st.button("预测"):
        if prediction[0] == 1:
            st.success("预测结果：乘客生还")
        else:
            st.error("预测结果：乘客未生还")

# 运行应用程序
if __name__ == '__main__':
    app_ui()
