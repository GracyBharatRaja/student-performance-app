import streamlit as st 
import pandas as pd 
import numpy as np 
import pickle 
from sklearn.preprocessing import StandardScaler,LabelBinarizer 
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://bharatgracy:8081sunny@cluster0.sa3qvy0.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db=client['student']
collection=db["student_pred"]



def load_model():
    with open("student_lr_final_model.pkl","rb") as file:
        model,scaler,le=pickle.load(file)
        return model,scaler,le


def preprocessing_input_data(data,scaler,le):

    data['Extracurricular Activities']=le.transform([data['Extracurricular Activities']]) [0]
    df=pd.DataFrame([data])
    df_transformed=scaler.transform(df)
    return df_transformed

def predict_data(data):

    model,scaler,le=load_model()

    processed_data=preprocessing_input_data(data,scaler,le)
    prediction=model.predict(processed_data)

    return prediction

def main():
    st.title("Student Performance Prediction")
    st.write("Enter your data to get a prediction for your performance")

    hours_studied=st.number_input("Hours Studied",min_value=1,max_value=10,value=5)
    previous_score=st.number_input("Previous Score",min_value=40,max_value=100,value=70)
    extra=st.selectbox("Extra Currucullar activitie",['Yes',"No"])
    sleeping_hour=st.number_input("Sleeping Hours",min_value=4,max_value=10,value=7)
    papers_solved=st.number_input("Number of question papers solved",min_value=0,max_value=10,value=5)

    if st.button("Predict your Score"):
        user_data={

              "Hours Studied":hours_studied,
              "Previous Scores":previous_score,
              "Extracurricular Activities":extra,
              "Sleep Hours":sleeping_hour,
              "Sample Question Papers Practiced":papers_solved }
              



        prediction= predict_data(user_data)
        st.success(f"Your Prediction result is {prediction}")
        user_data["prediction"]=float(prediction)
        user_data = {key: int(value) if isinstance(value, np.integer) else float(value) if isinstance(value, np.floating) else value for key, value in user_data.items()}

        collection.insert_one((user_data))
        
          


             

if __name__=="__main__":

     main()