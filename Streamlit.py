import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
regressor=pickle.load(open('Models/linearReg(weight).pkl','rb'))
scaler=pickle.load(open('Models/scaler_weight.pkl','rb'))
def pred(weight):
    weight_scaled=scaler.transform([[weight]])
    result=regressor.predict(weight_scaled)
    return result[0]
def main():
    st.title('Welcome to my height prediction app!')
    wt=st.text_input("Enter your weight in kgs", 'Type here!')
    result=''
    if st.button('predict'):
        result=pred(wt)
    st.success("The predicted height is {} cm".format(result))

if __name__=='__main__':
    main()