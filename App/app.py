# Import Packages
from numpy.core.fromnumeric import mean
import streamlit as st
import altair as at
from streamlit.elements import layouts
import plotly.express as px
import pandas as pd
import numpy as np
import joblib

# Models

models = ['LogisticRegression', 'AdaBoostClassifier', 'SVM',
          'KNeighborsClassifier', 'RandomForestClassifier', 'XGBClassifier','ExtraTreesClassifier',
          'KMeans','DecisionTreeClassifier','RidgeClassifier']

# Load the models
pipeline_lgr = joblib.load(open("../Models/LGR.pkl", 'rb'))
pipeline_ada = joblib.load(open("../Models/ADA.pkl", 'rb'))
pipeline_svm = joblib.load(open("../Models/SVM.pkl", 'rb'))
pipeline_knn = joblib.load(open("../Models/KNN.pkl", 'rb'))
pipeline_rfc = joblib.load(open("../Models/RFC.pkl", 'rb'))
pipeline_xgb = joblib.load(open("../Models/XGB.pkl", 'rb'))
pipeline_etc = joblib.load(open("../Models/ETC.pkl", 'rb'))
pipeline_kmc = joblib.load(open("../Models/KMC.pkl", 'rb'))
pipeline_dtc = joblib.load(open("../Models/DTC.pkl", 'rb'))
pipeline_rdc = joblib.load(open("../Models/RDC.pkl", 'rb'))

# Functions


def predict_emotions(text):
    results = [pipeline_lgr.predict([text]), pipeline_ada.predict([text]), pipeline_svm.predict(
        [text]), pipeline_knn.predict([text]), pipeline_rfc.predict([text]), pipeline_xgb.predict([text])]
    return results


def predict_probability(text):
    results = [pipeline_lgr.predict_proba([text]), pipeline_ada.predict_proba([text]), pipeline_svm.predict_proba(
        [text]), pipeline_knn.predict_proba([text]), pipeline_rfc.predict_proba([text]), pipeline_xgb.predict_proba([text])]
    return results


st.set_page_config(layout='wide')
st.title('Emotion Guesser')
menu = ['Home', 'About']
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Home':
    st.subheader('Home Page')
    with st.form(key='emotion_clf_form'):
        raw_text = st.text_area('Enter you sentence here')
        submit_text = st.form_submit_button('Submit')
        if submit_text:
            prediction = predict_emotions(raw_text)
            probability = predict_probability(raw_text)
            st.success('Orginal Text: ' + raw_text)
            for i in range(len(prediction)):
                with st.beta_container():
                    st.header(models[i])
                    col1, col2 = st.beta_columns(2)
                    with col1:
                        st.success('Emotion Prediction')
                        st.write('Predicted Emotion: ' + prediction[i][0])
                        st.write('Confidence: ', np.max([probability[i]]))

                    with col2:
                        st.success('Prediction Probability')
                        prob_df = pd.DataFrame(
                            probability[i], columns=pipeline_lgr.classes_)
                        prob_df_clean = prob_df.T.reset_index()
                        prob_df_clean.columns = ['emotions', 'probability']
                        fig = at.Chart(prob_df_clean).mark_bar().encode(
                            x='emotions', y='probability', color='emotions')
                        st.altair_chart(fig, use_container_width=True)

elif choice == 'About':
    st.subheader("Made by Anand")
    st.write(
        'A simple emotion detection app made using a various models')