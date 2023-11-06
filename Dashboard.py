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
   
    dataset = pd.read_csv('AB_NYC_2019.csv')
    dataset.head()
    
    dataset.info()

    # Create a Streamlit app
    st.title("Distribution Analysis")
    
    # Display the code in the Streamlit app
    st.code("""
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    log_transform_cols = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count']
    numerical_cols = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
    summary_stats = dataset[numerical_cols].describe()
    
    for i, col in enumerate(numerical_cols):
        ax = axes[i]
        data = dataset[col]
        if col in log_transform_cols:
            data = np.log1p(data)
            ax.set_title(f'Log-transformed Distribution of {col}')
        else:
            ax.set_title(f'Distribution of {col}')
        sns.histplot(data, bins=30, kde=True, ax=ax)
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    summary_stats
    """, language="python")
    
    # Display the plot in the Streamlit app
    st.write("### Distribution of Numerical Columns")
    st.pyplot(fig)    
    # Display the summary statistics if needed
    st.write(summary_stats)

elif selection == "Data":
    st.title("gg")
    st.subheader("ggggg")
