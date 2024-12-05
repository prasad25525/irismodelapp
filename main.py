import streamlit as st 
import pandas as pd
import pickle
import numpy as np

st.title('IRIS Spieces Prediction')

spl = st.number_input('sepal length (cm):')

spw = st.number_input('sepal width (cm):')


ptl = st.number_input('petal length (cm):')

ptw = st.number_input('petal width (cm):')

with open('model.pkl','rb') as f:
    model = pickle.load(f)
    f.close()
    
    
if st.button('Predict'):
    
    data = np.array([[spl,spw,ptl,ptw]])
    prediction = model.predict(data)[0]
    
    # 'setosa', 'versicolor', 'virginica']
    
    if prediction ==0:
        
        st.write('setosa')
    elif prediction ==1:
        st.write('versicolor')
    elif prediction ==2:
        st.write('virginica')

