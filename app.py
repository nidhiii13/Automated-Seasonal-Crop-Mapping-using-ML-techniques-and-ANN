import streamlit as st
import pandas as pd
import pickle 
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import os
path = os.getcwd()

model=pickle.load(open(path+'\Models\\randomForest.pkl','rb'))
model2 = load_model(path+'\Models\\ann.h5')

def predict_random_forest(data):
    print(data)
    X = data.iloc[:, 0:]
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X)
    input=np.array(X_test).astype(np.float64)
    print(X_test)
    
    prediction=model.predict(input)
    pred = model2.predict(input)
    for i in range(len(pred)):
        for j in range(7):
            if pred[i][j] == pred[i].max():
                pred[i][j] = 1
                p = j+1
        else:
            pred[i][j] = 0
    y_pred_ann = pred.astype(int)
    return [prediction[0] ,p]


def main():
    html_temp = """
    <div style="background-color:#025246 ;">
    <h2 style="color:white;text-align:center;">Automated seasonal crop mapping ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)

    crop =["Corn","Pea","Canola","Soy","Oat","Wheat","Broadleaf"]

    if st.button("Predict"):
        output=predict_random_forest(dataframe)
        st.success('The crop classified according to random forest - {} '.format(crop[output[0]-1]))
        st.success('The crop classified according to Artificial neural network - {} '.format(crop[output[1]-1]))


if __name__ == '__main__':
    main()