import joblib
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
#____________________________________ Streamlit UI (WebPage) ____________________________________
st.set_page_config(page_title = 'Titanic Model',layout = 'wide')
st.title('Titanic Survival Prediction Project')
st.caption("Powered by Logistic Regression Model that predicts the survival of titanic passengers")

df_preview = None

try:
    df_preview = joblib.load('data_frame.pkl')
    with st.expander('First 100 Rows'):
        st.dataframe(df_preview.head(100))
except FileNotFoundError:
    pass
st.divider()

col_left,col_right = st.columns(2)

with col_left:
    st.subheader('Passenger Features')
    age = st.number_input('Age',min_value = 0,max_value = 80,step = 1)
    fare = st.number_input('Fare',min_value = 0,max_value = 513)
    sibsp = st.number_input('Siblings', min_value=0, max_value=8, step=1)
    parch = st.number_input('Parent/Child',min_value = 0,max_value = 6,step = 1)
    sex = st.selectbox('Sex', options=['Male', 'Female'])
    Pclass = st.selectbox('Pclass', options=[1, 2, 3])
    embarked = st.selectbox('Embarked', options=['S', 'C', 'Q'])
    title = st.selectbox('Title',['Mr','Mrs','Don','Col'
                                    ,'Capt','Dr','Major','Rev','Jonkheer'
                                    ,'The Countess'])

with col_right:
    st.subheader('Ai Prediction')
    st.write('Click predict to see the probability of survival')

    if st.button('Predict'):
        sex_val = 1 if sex == 'Male' else 0
        Pclass_1 = 1 if Pclass == 1 else 0
        Pclass_2 = 1 if Pclass == 2 else 0
        Pclass_3 = 1 if Pclass == 3 else 0
        Embarked_S = 1 if embarked == 'S' else 0
        Embarked_C = 1 if embarked == 'C' else 0
        Embarked_Q = 1 if embarked == 'Q' else 0
        mr = 1 if title == 'Mr' else 0
        mrs = 1 if title == 'Mrs' else 0
        capt = 1 if title == 'Capt' else 0
        don = 1 if title == 'Don' else 0
        col = 1 if title == 'Col' else 0
        major = 1 if title == 'Major' else 0
        dr = 1 if title == 'Dr' else 0
        rev = 1 if title == 'Rev' else 0
        jonkheer = 1 if title == 'Jonkheer' else 0
        countess = 1 if title == 'The Countess' else 0

        payload = {
            'Sex': sex_val,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Embarked_C': Embarked_C,
            'Embarked_Q': Embarked_Q,
            'Embarked_S': Embarked_S,
            'Title_Dr': 0,
            'Title_Mr': 0 if sex == 'Female' else 1,
            'Title_Mrs': 0 if sex == 'Male' else 1,
            'Title_Don': don,
            'Title_Col': col,
            'Title_Capt': capt,
            'Title_Jonkheer': jonkheer,
            'Title_the_Countess': countess,
            'Title_Major': major,
            'Title_Rev': rev,
            'Fare': fare,
            'Pclass_1': Pclass_1,
            'Pclass_2': Pclass_2,
            'Pclass_3': Pclass_3
        }
        try:
            response = requests.post('http://127.0.0.1:8000/predict',json = payload)
            if response.status_code == 200:
                result = response.json()
                prediction = result['Prediction']
                probability = result['Probability']
                if prediction == 1:
                    st.success(f'Result: **Survived!**, Probability: {round(probability * 100,2)}%')
                else:
                    st.error(f'Result: **Did Not Survive!**, Probability: {round(probability * 100,2)}%')
        except requests.exceptions.ConnectionError:
            st.error('Could not connect to FastAPI server!')
    else:
        st.info('Fill the features in the left before clicking (Predict)')
st.divider()

with st.sidebar:
    st.subheader('Features Filtering')

    min_value , max_value = df_preview['Fare'].min() , df_preview['Fare'].max()

    pclass_list = ['Pclass_1','Pclass_2','Pclass_3']
    embarked_list = ['Embarked_C','Embarked_Q','Embarked_S']

    fare = st.slider('Fare',min_value,max_value,(min_value,max_value))
    pclass = st.multiselect('Pclass',options = pclass_list,default = pclass_list)
    embarked = st.multiselect('Embarked',options = embarked_list,default = embarked_list)

    filtered_data = df_preview[
        (df_preview['Fare'] >= fare[0]) &
        (df_preview['Fare'] <= fare[1]) &

        (   ((df_preview['Pclass_1'] == 1) & ('Pclass_1' in pclass)) |
            ((df_preview['Pclass_2'] == 1) & ('Pclass_2' in pclass)) |
            ((df_preview['Pclass_3'] == 1) & ('Pclass_3' in pclass))
        ) &
        (   ((df_preview['Embarked_C'] == 1) & ('Embarked_C' in embarked)) |
            ((df_preview['Embarked_Q'] == 1) & ('Embarked_Q' in embarked)) |
            ((df_preview['Embarked_S'] == 1) & ('Embarked_S' in embarked))
        )
    ]

st.subheader('Data Visualization')

if not filtered_data.empty:
    col1,col2 = st.columns(2)

    with col1:
        fig1 = px.scatter(
            filtered_data,
            x = 'Age',
            y = 'Fare',
            color = 'Survived',
            hover_data = ['Sex','SibSp','Parch'],
            title = 'Age vs Fare (Colored By Survival Status)'
        )
        fig1.update_layout(template = 'plotly_dark')
        st.plotly_chart(fig1, width = 'stretch')

    with col2:
        filtered_data['port'] = 'S'
        filtered_data.loc[filtered_data['Embarked_C'] == 1,'port'] = 'C'
        filtered_data.loc[filtered_data['Embarked_Q'] == 1,'port'] = 'Q'

        fig2 = px.pie(
            filtered_data,
            names = 'port',
            title = 'Proportion Of Survival By Passenger Embarked'
        )
        fig2.update_layout(template = 'plotly_dark')
        st.plotly_chart(fig2, width = 'stretch')

else:
    st.warning('No passengers match the selected filters. Try broadening your criteria!')