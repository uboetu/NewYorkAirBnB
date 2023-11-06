import pandas as pd
import requests
import folium
import numpy as np
import math
import missingno as msno
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt
from folium.plugins import MarkerCluster
import streamlit as st
from streamlit_folium import folium_static
from folium import Element
from ipywidgets import interact
import json
import html



st.set_option('deprecation.showPyplotGlobalUse', False)



st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Home", "Data", "Data1", "data2", "data3"])

# Main content
if selection == "Home":
    st.title("Basic data")
    st.subheader("hier vertel ik een mooi verhaaltje")
    st.markdown(

    """
    
   

    
            """
    )     
   
  


elif selection == "Data":
    st.title("gg")
    st.subheader("ggggg")
