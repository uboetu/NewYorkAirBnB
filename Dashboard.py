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

data_start = pd.read_csv('AB_NYC_2019.csv') #Originele data van Kaggle
data_subway = pd.read_csv('Subway_Location.csv') #Extra data subway stations
dataset = pd.read_csv('NY_AirBnB_Feature_2.csv') #Dataset soort gemerged/klaar gemaakt om model mee te maken


st.set_page_config(layout="wide")

# Function to create the map with Marker Clusters for unique AirBnBs
def create_map(data):
    # Create a Map centered around New York with a wider view and a zoomed-out starting point
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, width='100%')
    marker_cluster = MarkerCluster().add_to(m)
    unique_listings = data.drop_duplicates(subset=['latitude', 'longitude'])
    for idx, row in unique_listings.iterrows():
        folium.Marker(
            [row['latitude'], row['longitude']],
            popup=row['name']
        ).add_to(marker_cluster)
    return m

def load_data():
    data = pd.read_csv('AB_NYC_2019.csv')
    data['last_review'] = pd.to_datetime(data['last_review'])
    return data

# Load the data
airbnb_data = load_data()

# Calculate key metrics for the dashboard
average_price = airbnb_data['price'].mean()
median_price = airbnb_data['price'].median()
total_listings = len(airbnb_data)
average_minimum_nights = airbnb_data['minimum_nights'].mean()
most_recent_review = airbnb_data['last_review'].max()
top_neighborhoods = airbnb_data['neighbourhood'].value_counts().head(5)
average_availability_365 = airbnb_data['availability_365'].mean()
price_quartiles = np.percentile(airbnb_data['price'], [25, 50, 75])
price_zero_listings = airbnb_data[airbnb_data['price'] == 0].shape[0]

# Set Streamlit options
st.set_option('deprecation.showPyplotGlobalUse', False)

# Sidebar navigation
st.sidebar.title("Navigation")
page_names = [
    "Home: Overview of NYC AirBnB Data",
    "Explore: Dive Into Data Analysis",
    "Prepare: Data Cleaning and Feature Engineering",
    "Predict: Machine Learning Models for Price Forecasting",
    "Wrap-Up: Summary and Key Takeaways"
]

selection = st.sidebar.radio("Navigate to", page_names)

if selection == "Home: Overview of NYC AirBnB Data":
    st.title("AirBnB Data Overview - New York 2019")
    st.markdown("""
        Welcome to the AirBnB New York 2019 Data Dashboard. 
        This interactive tool allows you to explore the landscape of AirBnB listings across New York City. 
        Discover key insights, delve into pricing trends, and understand the dynamics of the rental market.
    """)

    st.subheader("Map of Listings")
    sample_size = st.slider("Select the amount of AirBnB listings to display on the map", 100, 10000, 2500)
    map_data = airbnb_data.sample(n=sample_size).drop_duplicates(subset=['latitude', 'longitude'])
    map_fig = create_map(map_data)
    # Set width as a percentage of the page width to make the map wider
    col1, col2, col3 = st.columns([1,6,1])  # The middle column takes up the majority of the space

    with col2:  # This places the map in the middle column
        folium_static(map_fig, width=950)



    # Data Overview Section
    st.subheader("Data Overview")
    st.json({
        'Number of Listings': len(airbnb_data),
        'Number of Features': airbnb_data.shape[1],
        'Missing Values': airbnb_data.isnull().sum().sum(),  # Total number of missing values
        'Date Range': f"{airbnb_data['last_review'].min().date()} to {airbnb_data['last_review'].max().date()}"
    })

    # Key Metrics
    st.subheader("Key Metrics Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Price", f"${airbnb_data['price'].mean():.2f}")
    col2.metric("Median Price", f"${airbnb_data['price'].median():.2f}")
    col3.metric("Total Listings", f"{len(airbnb_data)}")
    col4.metric("Average Minimum Nights", f"{airbnb_data['minimum_nights'].mean():.2f}")

    # Additional Insights
    st.subheader("Recent Activity & Popular Neighborhoods")
    st.write("Most Recent Review Date:", airbnb_data['last_review'].max().date())
    st.table(airbnb_data['neighbourhood'].value_counts().head(5))

    st.subheader("Price Distribution")
    price_quartiles = np.percentile(airbnb_data['price'], [25, 50, 75])
    st.write(f"25th Percentile: ${price_quartiles[0]:.2f}")
    st.write(f"Median Price: ${price_quartiles[1]:.2f}")
    st.write(f"75th Percentile: ${price_quartiles[2]:.2f}")

    price_zero_listings = airbnb_data[airbnb_data['price'] == 0].shape[0]
    if price_zero_listings > 0:
        st.warning(f"There are {price_zero_listings} listings with a price of $0 which may require further investigation.")

# if selection == "Introduction to AirBnB Data":
#     st.title("AirBnB data New York 2019")
#     st.subheader("Exploring the Heartbeat of New York Through AirBnB: A Journey into the City's Living Spaces")

#     # Display key metrics
#     st.write("## Key Metrics Summary")
#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric("Average Price", f"${average_price:.2f}")
#     col2.metric("Median Price", f"${median_price:.2f}")
#     col3.metric("Total Listings", f"{total_listings}")
#     col4.metric("Average Minimum Nights", f"{average_minimum_nights:.2f}")

#     st.write("## Recent Activity")
#     st.write("Most Recent Review Date:", most_recent_review.date())

#     st.write("## Popular Neighborhoods")
#     st.table(top_neighborhoods)

#     st.write("## Price Distribution")
#     st.write(f"25th percentile: ${price_quartiles[0]:.2f}")
#     st.write(f"Median Price: ${price_quartiles[1]:.2f}")
#     st.write(f"75th percentile: ${price_quartiles[2]:.2f}")

#     if price_zero_listings > 0:
#         st.warning(f"There are {price_zero_listings} listings with a price of $0 which may require further investigation.")

#     # Map visualization with unique data points
#     st.write("## Map of Listings")
#     map_fig = create_map(airbnb_data.sample(2500).drop_duplicates(subset=['latitude', 'longitude']))
#     folium_static(map_fig)
    
#     # Summary statistics
#     st.write("## Key Metrics Summary")
#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric("Average Price", f"${average_price:.2f}")
#     col2.metric("Median Price", f"${median_price:.2f}")
#     col3.metric("Total Listings", f"{total_listings}")
#     col4.metric("Average Minimum Nights", f"{average_minimum_nights:.2f}")

#     # Recent activity
#     st.write("## Recent Activity")
#     st.write("Most Recent Review Date:", most_recent_review.date())

#     # Popular neighborhoods
#     st.write("## Popular Neighborhoods")
#     st.table(top_neighborhoods)

#     # Price distribution
#     st.write("## Price Distribution")
#     st.write(f"25th percentile: ${price_quartiles[0]:.2f}")
#     st.write(f"Median Price: ${price_quartiles[1]:.2f}")
#     st.write(f"75th percentile: ${price_quartiles[2]:.2f}")

#     # Note about special cases
#     if price_zero_listings > 0:
#         st.warning(f"There are {price_zero_listings} listings with a price of $0 which may require further investigation.")

#     # Use the navigation bar to explore different sections of the dashboard
#     st.sidebar.header('Navigation')
#     st.sidebar.write("Use the navigation bar to explore different sections of the dashboard. Each section provides deeper analysis and interactive visualizations.")
   
#     data_start.head()
    
#     data_start.info()

#     # Create a Streamlit app
#     st.title("Distribution Analysis")
    
#     # Display the code in the Streamlit app
          
#     fig, axes = plt.subplots(3, 2, figsize=(14, 10))
#     axes = axes.flatten()

#     log_transform_cols = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count']
#     numerical_cols = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
#     summary_stats = dataset[numerical_cols].describe()
    
#     for i, col in enumerate(numerical_cols):
#         ax = axes[i]
#         data = dataset[col]
#         if col in log_transform_cols:
#             data = np.log1p(data)
#             ax.set_title(f'Log-transformed Distribution of {col}')
#         else:
#             ax.set_title(f'Distribution of {col}')
#         sns.histplot(data, bins=30, kde=True, ax=ax)
#         ax.set_xlabel(col)
#         ax.set_ylabel('Frequency')
    
#     plt.tight_layout()
#     plt.show()
        
        
#     # Display the plot in the Streamlit app
#     st.write("### Distribution of Numerical Columns")     
#     st.pyplot(fig)    
#     # Display the summary statistics if needed
#     st.write(summary_stats)

elif selection == "Data":
    st.title("gg")
    st.subheader("ggggg")



elif selection == "data1":
    st.title("gg")
    st.subheader("ggggg")



elif selection == "data1":
    st.title("gg")
    st.subheader("ggggg")



elif selection == "data3":
    st.title("gg")
    st.subheader("ggggg")
