import streamlit as st 
import pandas as pd
import pickle
import numpy as np

st.title('IRIS Spieces Prediction')

# ['sepal length (cm)',
#  'sepal width (cm)',
#  'petal length (cm)',
#  'petal width (cm)']


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
    

    
# names = ['prasad','ram']
# age = [29,30]

# for name,a in zip(names,age):
#     print(name,a)
    







# st.header('i am Header')
# st.subheader('i am subheader')

# st.text_input('Enter your name:')

# st.selectbox('Choose your color:',['red','green','blue'])

# st.slider('select your age:',min_value=1,max_value=100,step=5)

# data = {
#     'names':['sneha','raju'],
#     'age':[28,29]
# }

# df = pd.DataFrame(data)
# st.dataframe(df)

# st.line_chart(df['age'])
# st.

# if st.button('Submit'):
#     st.write('Submitted')

# st.file_uploader("upload file")
# st.checkbox('Are you agree..?')

import pandas as pd
