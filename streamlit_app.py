# 导入必要的库和数据集
from collections import namedtuple
import altair as alt
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据集
@st.cache
def load_data():
    data = pd.read_csv('train.csv')
    return data

# 训练模型
@st.cache
def train_model(data):
    X = data.drop(['Survived'], axis=1)
    y = data['Survived']
    model = RandomForestClassifier()
    model.fit(X, y)
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
    fare = st.slider('船票费用', 0, 100, 10)

    # 创建一个特征向量
    features = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'Fare': [fare]
    })

    # 进行预测
    prediction = model.predict(features)[0]

    # 显示预测结果
    if prediction == 0:
        st.error('很抱歉，您可能不会生还。')
    else:
        st.success('恭喜您，您有可能生还！')

# 运行应用程序
if __name__ == '__main__':
    app_ui()
