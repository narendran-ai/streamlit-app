import streamlit as st
import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

st.set_page_config('Mushrooms prediction')
st.title('Mushroom')
st.markdown('lets find out your mushrooms are edbible or not')


upload_file=st.file_uploader('upload your raw.csv file')
if upload_file:
    df=pd.read_csv(upload_file)
    st.dataframe(df)

    st.header('your file encoding is done')

    encoder=LabelEncoder()
    for columns in range(len(df.columns)):
        df[df.columns[columns]]=encoder.fit_transform(df[df.columns[columns]])
    st.dataframe(df)


    model=joblib.load('mushroom,logistic.pkl')
    predict=pd.DataFrame(model.predict(df))
    predict.columns=['result']
    result=predict.replace({1:'edible',0:'posion'})
    st.header('your prediction is done')
    st.dataframe(result)


    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name="large_df.csv",
    mime="text/csv",
)