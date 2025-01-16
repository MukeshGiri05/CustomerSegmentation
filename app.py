import streamlit as st
from streamlit_option_menu import option_menu as om
from App_Files import ChurnPrediction, RFMAnalysis, Home
from PIL import Image as ig

st.set_page_config(page_title="Customer And Product Analysis Tool", page_icon=ig.open("App_Data\\fevicon.jpg"), layout="centered", initial_sidebar_state="auto", menu_items=None)

with st.sidebar:
    selected = om("Main Menu", ['Home','RFM Segmentation','Churn Prediction'], 
        icons=['house','bi-layers-half','bi-bar-chart-fill','bi-book-half','bi-hypnotize','bi-basket-fill','bi-boxes','gear'], menu_icon="cast", default_index=0)


if selected == "Home":
    st.header("Home Screen")
    Home.Home_Page()
    pass
elif selected == "RFM Segmentation":
    st.header("RFM Segmentation")
    RFMAnalysis.run()
elif selected == "Churn Prediction":
    st.header("Churn Prediction")
    ChurnPrediction.run()
    pass

