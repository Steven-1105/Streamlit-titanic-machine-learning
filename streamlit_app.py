# 导入必要的库和数据集
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# 加载数据集
@st.cache_resource
def load_data():
    data = pd.read_csv('train_fini.csv')
    # 处理缺失值
    imputer = SimpleImputer(strategy='median')
    imputer.fit(data)
    data = pd.DataFrame(imputer.transform(data), columns=data.columns)
    return data

# 训练模型
@st.cache_resource
def train_model(data):
    X = data.drop(['Survived'], axis=1)
    y = data['Survived']
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
    parch = st.slider('父母/子女数量', 0, 6, 0)
    fare = st.slider('船票费用', 0, 100, 10)
    embarked = st.radio('登船港口', ['C', 'Q', 'S'])

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
    
    
    #Sex
    features.loc[features.Sex=="male",'Sex']='0'
    features.loc[features.Sex=="female",'Sex']='1'
    features.loc[features.Sex.isna(),'Sex']='0'
    
    # 进行数据清洗，把内容转换成与train一样的格式
    imputer = SimpleImputer(strategy='median')
    imputer.fit(features)
    features = pd.DataFrame(imputer.transform(features), columns=features.columns)
    
    # Age
    features.loc[features.Age.isna(),'Age']= 20.0
    #features['Age'] = features['Age'].fillna(features['Age'].median())
    features['Age']=(features['Age']-features['Age'].mean())/features['Age'].std()

    #Embarked
    features.loc[features.Embarked=="C",'Embarked']='1'
    features.loc[features.Embarked=="S",'Embarked']='2'
    features.loc[features.Embarked.isna(),'Embarked']='2'
    features.loc[features.Embarked=="Q",'Embarked']='3'

    #Fare
    features.loc[features.Fare.isna(), 'Fare'] = 50.0
    features['Fare'] = features['Fare'].fillna(features['Fare'].median())
    features['Fare'] = (features['Fare']-features['Fare'].mean())/features['Fare'].std()

    
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
