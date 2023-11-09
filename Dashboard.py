import pandas as pd
import folium
import numpy as np
import missingno as msno
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt
from folium.plugins import MarkerCluster
from folium.plugins import HeatMap
import streamlit as st
from streamlit_folium import folium_static
from folium import Element
from ipywidgets import interact
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import urllib
import requests
from PIL import Image, ImageDraw, ImageFont
import plotly.express as px



data_start = pd.read_csv('AB_NYC_2019.csv') #Originele data van Kaggle
data_subway = pd.read_csv('Subway_Location.csv') #Extra data subway stations
dataset = pd.read_csv('NY_AirBnB_Feature_2.csv') #Dataset soort gemerged/klaar gemaakt om model mee te maken
coordinaten= pd.read_csv('coordinaten.csv',delimeter=',') 

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
    "Extra Data: Subway Station Data",
    "Predict: Machine Learning Models for Price Forecasting",
    "Summary and Key Takeaways First Semester"
]

selection = st.sidebar.radio("Navigate to", page_names)

if selection == "Home: Overview of NYC AirBnB Data":
    st.markdown("""
    # Welcome to the AirBnB New York 2019 Data Dashboard
    This interactive tool allows you to explore the landscape of AirBnB listings across New York City. 
    Discover key insights, delve into pricing trends, and understand the dynamics of the rental market.

    The data for this dashboard is sourced from the [New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) on Kaggle. 
    It includes detailed listings activity and metrics in NYC for 2019, such as location, pricing, and more, to provide a comprehensive view of the Airbnb ecosystem.
    """)

    st.subheader("Map of Listings")
    sample_size = st.slider("Select the amount of AirBnB listings to display on the map", 100, 10000, 2500)
    map_data = airbnb_data.sample(n=sample_size).drop_duplicates(subset=['latitude', 'longitude'])
    map_fig = create_map(map_data)
    # Set width as a percentage of the page width to make the map wider
    col1, col2, col3 = st.columns([1,6,1])  # The middle column takes up the majority of the space

    with col2:  # This places the map in the middle column
        folium_static(map_fig, width=950)

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
#     col3.metric("Total Listings", f"{total_lgiistings}")
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

if selection == "Explore: Dive Into Data Analysis":

    def load_data():
        data = pd.read_csv('Subway_Location.csv')
        return data

    st.title("Data Exploration AirBnB New York City")

    st.markdown("""
                The New York City dataset, sourced from Kaggle. 
                This dataset serves as the foundation for our project, where our primary objective is to create an interactive dashboard for exploring and visualizing key insights from the AirBnB market in New York City.
                """)

    st.header('Data Exploring')
    st.markdown("""
               The New York City AirBnB dataset contains all the data needed information to find out more about hosts, geographical availability, necessary metrics to make predictions and draw conclusions.
                For reverence the first 5 rows of the dataset has been displayed.
                """ )
    
    df = pd.read_csv('AB_NYC_2019.csv')
    st.write(df.head())

    st.subheader('Missing values Analysis')
    st.write(df.head())
    #calculate missing values
    missing_values = df.isna().sum()

    #display missing values
    st.write('Missing Values:')
    st.table(missing_values)

    st.markdown("""
                In the Airbnb dataset, missing values are notable in the 'name' (16), 'host_name' (21), 'last_review' (10,052), and 'reviews_per_month' (10,052) columns.
                The absence of names suggests some properties lack designated titles, possibly influencing identification and branding analyses.
                The missing host names (21) may reflect listings without specified hosts. Notably, the 'last_review' and 'reviews_per_month' columns share identical counts for missing values, indicating properties without reviews.
                This absence could impact time-dependent analyses and metrics related to booking frequency.
                """)
    
    st.subheader('Descriptive Statistics')
    descriptive_stats = df.describe()
    st.table(descriptive_stats)

    st.markdown("### Breakdown of Key Statistics:")
    st. markdown("1. **ID and Host ID:** The ID columns have a count of 48,895 entries with unique identifiers."
            "   - The mean and standard deviation reveal the distribution and spread of these identifier values.")
    st.markdown("2. **Geographical Coordinates (Latitude and Longitude):**"
            "   - Latitude ranges around 40.73, with a small standard deviation, indicating relatively clustered locations."
            "   - Longitude is approximately -73.95, with a similarly low standard deviation, suggesting a concentrated geographic area.")
    st.markdown("3. **Price:**"
            "   - Prices vary widely, with a mean of $152.72 and a significant standard deviation of $240.15."
            "   - The minimum price is $0, indicating some listings are free, and the maximum is $10,000.")
    st.markdown("4. **Minimum Nights:**"
            "   - The mean minimum nights stay is around 7, with a diverse range from 1 to 1,250 nights.")
    st.markdown("5. **Number of Reviews and Reviews per Month:**"
            "   - The average number of reviews is approximately 23, with a substantial standard deviation of 44.55."
            "   - Reviews per month have a mean of 1.37, indicating an average frequency of reviews.")
    st.markdown("6. **Calculated Host Listings Count and Availability:**"
            "   - Hosts, on average, have around 7 listings, with a maximum of 327."
            "   - Availability ranges from 0 to 365 days, with a mean of 112.78.")

    st.subheader('What are the most popular neighborhoods for Airbnb listings in NYC?')
    st.markdown("""
                We are going to examine different neighborhood groups and their frequency to understand the datasets geographical diversity and make informed decisions in the analysis and modeling process.
                """)

    st.subheader('What are the most popular neighborhoods for Airbnb listings in NYC?')
    st.markdown("""
                We are going to examine different neighborhood groups and their frequency to understand the datasets geographical diversity and make informed decisions in the analysis and modeling process.
                """)

    #first plot
    df= pd.read_csv('AB_NYC_2019.csv')

    plt.figure(figsize=(8,6))
    sns.histplot(df['neighbourhood_group'], bins=20, kde=True)
    plt.title('Distribution of Listings by Neighbourhood Group')
    plt.xlabel('Neighbourhood Group')
    plt.ylabel('Count')

    st.pyplot(plt)

    st.write("""
            The graph shows that there is a significant amount of listings in Manhattan and Brooklyn.
            Manhattan has the highest number of Airbnb listings in NYC, making it the most popular neighborhood group.
            Brooklyn follows as the second most popular neighborhood group and Queens in third and Bronx with the fewest number of Airbnb listings.
            If you look at the proximity to attractions, manhattan for example, is the home to iconic sites like Times Square, Central Park, and Broadway, making it a top choice for tourists.
              """)
    
    st.subheader('What is the distribution of different room types?')
    st.markdown("""
                 This will show us what type of rooms people are listing on Airbnb in NYC.
                """)
    
    #second plot
    df = pd.read_csv('AB_NYC_2019.csv')

    # Calculate room type counts
    room_type_counts = df['room_type'].value_counts()

    # Create a pie chart using Matplotlib
    fig, ax = plt.subplots()
    ax.pie(room_type_counts, labels=room_type_counts.index, autopct='%1.1f%%',
       shadow=True, startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that the pie chart is circular

    # Set a title for the pie chart
    ax.set_title('Distribution of Room Types')
    st.pyplot(fig)

    

    st.write("""
            In conclusion, the dataset shows 3 types of rooms; private rooms, shared rooms and entire homes/apartments.
             Entire homes/apartments, representing 52% of listings, and private rooms 45.7% and lastly shared rooms with 2.37%.
             This could be because of guest preferences, the level of privacy and control over their property.
             Even though the price is different for each type of room. Lets now look at the prices for each room type. 
             """)

    st.subheader('How does the listing price vary across different neighborhoods and room types?')
    st.markdown("""
                Comparing the mean prices of different towns with the help of a scatter graph provides a better understanding of pricing variations in various areas, along with the number of bookings:
                - *Brooklyn*: Mean Price = 124.38, Number of Reviews = 2400
                - *Manhattan*: Mean Price = 196.88, Number of Reviews = 20000
                - *Quuens*: Mean Price = 99.52, Number of Reviews = 2700
                - *Staten Island*: Mean Price = 114.81, Number or Reviews = 3000
                - *Bronx*: Mean Price = 87.50, Number of Reviews = 2600)
                """)
    towns = ['Town1', 'Town2', 'Town3', 'Town4', 'Town5']
    mean_price = [124.383207, 196.875814, 99.517649, 114.812332, 87.496792]
    number_of_reviews = [2400, 2000, 2700, 3000, 2600]
    c = np.random.randint(10, size=5)

    # Create a Streamlit app
    st.markdown('*Relationship between Mean Price and Number of Reviews*')

    # Create the scatter plot
    fig, ax = plt.subplots()
    scatter = ax.scatter(towns, mean_price, s=number_of_reviews, alpha=0.7, c=c)
    ax.set_title('Relationship between Mean Price and Number of Reviews', size=8)
    ax.set_ylabel('Mean Price --->')

    # Add an explanation
    st.write('Number of reviews is represented by the size of the bubble.')
    st.pyplot(fig)


    NY_map = folium.Map([40.730610,-73.935242],zoom_start=10)
    HeatMap(dataset[['latitude','longitude']],radius=10).add_to(NY_map)
    folium_static(NY_map)

    nyc_img = mpimg.imread('New_York_City_.png')
    fig,(ax1, ax2) = plt.subplots(1,2,figsize=(32,8))

    ax1.imshow(nyc_img, extent=[-74.258, -73.7, 40.49, 40.92])
    ax2.imshow(nyc_img, extent=[-74.258, -73.7, 40.49, 40.92])
    plot1 = ax1.scatter(dataset['longitude'][dataset['price']<119], dataset['latitude'][dataset['price']<119], c=dataset['price'][dataset['price']<119], cmap='viridis', linewidth=1, alpha=0.6)
    plot2 = ax2.scatter(dataset['longitude'][dataset['price']>119], dataset['latitude'][dataset['price']>119], c=dataset['price'][dataset['price']>119], cmap='viridis', linewidth=1, alpha=0.6)
    cbar1 = fig.colorbar(plot1,ax=ax1)
    cbar2 = fig.colorbar(plot2,ax=ax2)
    cbar1.set_label('Price in USD',size=20)
    cbar2.set_label('Price in USD',size=20)

    ax1.set_title('Listings with price below mean.', size=20, pad=10)
    ax2.set_title('Listings with price above mean.', size=20, pad=10)
    st.pyplot(fig)


    nyc_img = mpimg.imread('New_York_City_.png')

    # Define the extent of your background image based on the longitude and latitude scales
    extent = [-74.258, -73.7, 40.49, 40.92]

    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))  # Adjust the size as needed

    # Plot the first scatter plot
    axes[0].imshow(nyc_img, extent=extent, aspect='auto')
    sns.scatterplot(ax=axes[0], x='longitude', y='latitude', hue='neighbourhood_group', data=dataset)
    axes[0].set_title('Airbnb Listings by Neighbourhood Group')

    # Plot the second scatter plot
    axes[1].imshow(nyc_img, extent=extent, aspect='auto')
    sns.scatterplot(ax=axes[1], x='longitude', y='latitude', hue='room_type', data=dataset)
    axes[1].set_title('Airbnb Listings by Room Type')

    # Plot the third scatter plot
    axes[2].imshow(nyc_img, extent=extent, aspect='auto')
    sns.scatterplot(ax=axes[2], x='longitude', y='latitude', hue='availability_365', data=dataset)
    axes[2].set_title('Airbnb Listings by Availability')

    # Adjust the layout
    plt.tight_layout()

    # Display the plots in Streamlit
    st.pyplot(fig)

elif selection == "Extra Data: Subway Station Data":
    st.title('NYC Airbnb Proximity to Subway Stations')
    st.markdown("""
    This dashboard presents an overview of the proximity of Airbnb listings to the nearest subway stations in New York City. 
    The dataset contains information about the locations of Airbnb properties and subway stations, which can be used to analyze accessibility and convenience for travelers.

    You can access the dataset used for this analysis on [Kaggle](https://www.kaggle.com/code/kalilurrahman/new-york-city-subway-system-map-visualization).
    """)
    # Display the head of the dataset
    st.header('Dataset Head')
    st.write(data_subway.head())

    nyc_map_img = mpimg.imread('New_York_City_.png')
    def plot_nyc_map(dataset, subway_data, map_path):
        try:
            # Filter out outliers in the subway dataset based on latitude and longitude
            filtered_subway_data = subway_data[(subway_data['Entrance Latitude'] >= 40.5) & 
                                            (subway_data['Entrance Latitude'] <= 41.0) & 
                                            (subway_data['Entrance Longitude'] >= -74.5) & 
                                            (subway_data['Entrance Longitude'] <= -73.5)]

            # Read the New York City map image
            

            # Set the extent for better alignment
            extent = [-74.258, -73.7, 40.49, 40.92]

            # Initialize the plot
            plt.figure(figsize=(12, 12))

            # Plot the map image
            plt.imshow(nyc_map_img, extent=extent, aspect='auto', alpha=0.5)

            # Plot Airbnb listings as blue points
            plt.scatter(dataset['longitude'], dataset['latitude'], c='blue', label='Airbnb Listings', alpha=0.1)

            # Plot subway stations as red points
            plt.scatter(filtered_subway_data['Entrance Longitude'], filtered_subway_data['Entrance Latitude'], c='red', label='Subway Stations', alpha=0.5)

            # Add title and labels
            plt.title('Airbnb Listings and Subway Stations in New York City')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.legend()

            # Show the plot
            plt.show()

        except FileNotFoundError:
            print("File not found. Please check the file path.")
        except SyntaxError:
            print("Not a valid image file. Please check the file format.")
    plot_nyc_map(dataset, data_subway, nyc_map_img)
    st.pyplot()

    st.header('Dataset Head')
    st.write(coordinaten.head())

    st.image('Top10_afbeelding.png')

    fig1=px.histogram(data_start, x="neighbourhood_group", nbins=20)
    st.plotly_chart(fig1)

    fig2 = px.histogram(data_start, x="neighbourhood", nbins=20)
    st.plotly_chart(fig2)

    st.image('Buurten_afbeelding.png')

    
elif selection == "Prepare: Data Cleaning and Feature Engineering":

    st.title('Data Cleaning and Feature Engineering')
    

   
    st.markdown("### Feature Engineering: Date and Ratio Calculations")
    st.markdown("""
    Enhancing the dataset with new features can provide deeper insights and improve model performance. 
    Here's what we've added:
    
    - **Days Since Last Review**: This is calculated by finding the difference between the most recent review date in the dataset and the 'last_review' date for each listing.
    
    - **Potential Superhost Identification**: Listings are flagged as potential Superhosts if they are in the 75th percentile or higher for both the number of reviews and review frequency.
    
    - **Review-to-Availability Ratio**: This ratio is computed by dividing the number of reviews by the availability over 365 days to understand how often listings are reviewed relative to their availability.
    """)
    
    # Convert 'last_review' to datetime
    dataset['last_review'] = pd.to_datetime(dataset['last_review'])
    
    # Calculate 'review_frequency'
    dataset['review_frequency'] = dataset['number_of_reviews'] / dataset['availability_365']
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Calculate 'days_since_last_review'
    reference_date = dataset['last_review'].max()
    dataset['days_since_last_review'] = (reference_date - dataset['last_review']).dt.days
    
    # Identify potential 'is_superhost'
    high_review_count_threshold = dataset['number_of_reviews'].quantile(0.75)
    high_review_frequency_threshold = dataset['review_frequency'].quantile(0.75)
    dataset['is_superhost'] = ((dataset['number_of_reviews'] >= high_review_count_threshold) & 
                               (dataset['review_frequency'] >= high_review_frequency_threshold)).astype(int)
    
    # Calculate 'review_to_availability_ratio'
    dataset['review_to_availability_ratio'] = dataset['number_of_reviews'] / dataset['availability_365']
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Display the new features
    st.markdown("### New Features Preview")
    st.dataframe(dataset[['days_since_last_review', 'is_superhost', 'review_to_availability_ratio']].head())


    # Outlier Detection and Removal
    st.markdown("## Outlier Detection and Removal")
    st.markdown("""
    Outlier detection is a crucial step in data preprocessing, particularly for price-related data. 
    Outliers can significantly skew our analysis, leading to inaccurate models or misinformed decisions. 
    Below is a summary of the dataset before and after the removal of outliers from the 'price' variable.
    """)
    
    # Calculate IQR
    Q1 = dataset['price'].quantile(0.25)
    Q3 = dataset['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    dataset_no_outliers = dataset[(dataset['price'] >= lower_bound) & (dataset['price'] <= upper_bound)]
    
    # Summary of Outliers Removed
    summary_no_outliers = {
        'Initial data size': dataset.shape,
        'New data size': dataset_no_outliers.shape,
        'Number of outliers removed': dataset.shape[0] - dataset_no_outliers.shape[0]
    }
    summary_df = pd.DataFrame(list(summary_no_outliers.items()), columns=['Metric', 'Value'])
    st.subheader('Outlier Removal Summary')
    st.table(summary_df)
    
    # Visualizations for Price Distribution
    st.markdown("### Price Distribution Before and After Outlier Removal")
    st.markdown("""
    The following plots show the distribution of the 'price' variable before and after the removal of outliers.
    This visual comparison helps to understand the effect of outlier removal on the data distribution.
    """)
    
    # Set up the figure layout
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sns.boxplot(y=dataset['price'], ax=ax[0])
    ax[0].set_title('Original Price Distribution')
    sns.boxplot(y=dataset_no_outliers['price'], ax=ax[1])
    ax[1].set_title('Price Distribution Without Outliers')
    for a in ax:
        a.set_ylabel('Price')
    st.pyplot(fig)
    

    fill_columns = ['reviews_per_month', 'review_frequency', 'days_since_last_review', 'review_to_availability_ratio']
    for col in fill_columns:
        dataset_no_outliers[col].fillna(0, inplace=True)
    
    cols_to_drop = ['id', 'name', 'host_id', 'host_name', 'last_review']
    data_cleaned = dataset_no_outliers.drop(columns=cols_to_drop)
    
  
    categorical_cols = ['neighbourhood_group', 'neighbourhood', 'room_type']
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    data_encoded = pd.DataFrame(one_hot_encoder.fit_transform(data_cleaned[categorical_cols]))
    data_encoded.index = data_cleaned.index
    data_encoded.columns = one_hot_encoder.get_feature_names_out(input_features=categorical_cols)
    num_data = data_cleaned.drop(columns=categorical_cols)
    data_prepared = pd.concat([num_data, data_encoded], axis=1)
    st.markdown("""
    ### Data Preparation Steps

    In order to ensure our dataset is primed for the modeling process, we undertake the following steps:

    - **Handling Missing Values**: 
        - Missing values, particularly in review-related features, are set to zero. This assumes that no reviews equate to a lack of data.

    - **Dropping Unnecessary Columns**: 
        - We remove columns that only serve as identifiers (such as IDs and names). These are not predictive and are therefore unnecessary for modeling.

    - **Encoding Categorical Variables**: 
        - Categorical features are transformed using One-Hot Encoding, turning them into a machine-readable format that enhances the predictive quality of our models.

    - **Feature Scaling**: 
        - Numerical features are scaled to a mean of zero and a standard deviation of one. This step is crucial for algorithms that are sensitive to the scale of the data.

    - **Data Splitting**: 
        - The dataset is divided into training and testing sets, allowing us to train our models and then test their performance on unseen data.
    """)
    # Feature Scaling
    numerical_cols = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 
                      'reviews_per_month', 'calculated_host_listings_count', 
                      'availability_365', 'review_frequency', 
                      'days_since_last_review', 'distance_to_nearest_subway']
    scaler = StandardScaler()
    data_prepared[numerical_cols] = scaler.fit_transform(data_prepared[numerical_cols])
    
    # Data Splitting

    y = data_prepared['price']
    X = data_prepared.drop('price', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Display the first few rows of the scaled and prepared dataset
    st.markdown("### Scaled and Prepared Dataset")
    st.markdown("""
    Here's how the prepared dataset looks after scaling the numerical features:
    """)
    st.write(data_prepared.head())

    # Removing any unnamed columns that might have been added during the data preparation
    if 'Unnamed: 0' in data_prepared.columns:
        data_prepared.drop('Unnamed: 0', axis=1, inplace=True)
    
    # Display the cleaned and final dataset
    st.markdown("### Final Dataset for Modeling")
    st.write(data_prepared.head())






if selection == "Predict: Machine Learning Models for Price Forecasting":
    st.title('Predictive Modeling for AirBnB Prices in New York')
    
    st.markdown("""
        In this section, we dive into the machine learning models used to predict AirBnB listing prices. 
        We'll walk through the preprocessing steps, model selection, hyperparameter tuning, and evaluation metrics.
        """)

    st.header('Data Preprocessing')
    st.markdown("""
    - **Outlier Removal**: Outliers in the price column were identified using the IQR method and removed to improve model accuracy.
    - **Missing Value Imputation**: Missing values in review-related features were filled with zeros.
    - **Feature Encoding**: Categorical features like neighbourhood groups and room types were one-hot encoded.
    - **Feature Scaling**: Numerical features were scaled to ensure equal importance during model training.
    - **Data Splitting**: The data was split into training and testing sets with an 80-20 ratio.
    """)
    
    st.header('Model Selection & Cross-Validation')
    st.markdown("""
    A variety of models were evaluated using cross-validation to select the best performing model based on RMSE:
    
    - **Linear Models**: Linear Regression, Ridge, Lasso
    - **Tree-Based Models**: Decision Tree, Random Forest, Gradient Boosting
    """)
    model_results = pd.DataFrame({
    'Model': ['LinearRegression', 'Ridge', 'Lasso', 'DecisionTree', 'RandomForest', 'GradientBoosting'],
    'Average RMSE': [21588379748677.76, 46.87, 47.53, 62.10, 44.37, 44.97]
    })

    st.table(model_results)
    
    st.write("""
    ## Hyperparameter Tuning Results
    The best performing models were RandomForest and GradientBoosting after hyperparameter tuning. Below are the best parameters and performance metrics for these models.
    """)
    tuning_results = pd.DataFrame({
    'Model': ['RandomForest', 'GradientBoosting'],
    'Best Parameters': [
        "{'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_depth': 10}",
        "{'n_estimators': 100, 'min_samples_split': 6, 'min_samples_leaf': 2, 'max_depth': 7, 'learning_rate': 0.1}"
    ],
    'Mean Squared Error': [1962.94, 1922.92],
    'Root Mean Squared Error': [44.31, 43.85],
    'R^2 Score': [0.58, 0.59]
    })

    # Display the tuning results as a table
    st.table(tuning_results)

    st.header('Model Evaluation')
    st.markdown("""
    The final models were evaluated using the test set. Key metrics were:
    
    - **Mean Squared Error (MSE)**
    - **Root Mean Squared Error (RMSE)**
    - **R-squared (R²)**
    """)

    st.header('Results Summary')
    st.markdown("""
    Here's a summary of the model performance after hyperparameter tuning:
    """)

    # Summary table for model results
    models_results = {
        'Model': ['Random Forest', 'Gradient Boosting'],
        'MSE': [1962.94, 1922.92],
        'RMSE': [44.31, 43.85],
        'R²': [0.58, 0.59]
    }

    results_df = pd.DataFrame(models_results)
    st.table(results_df)

    st.header('Feature Importances from Random Forest')
    st.markdown("Understanding which features most influence the price can provide insights into the market dynamics.")
    

    #fix later please
    # image_path = 'path_to_your_image.png'  # Replace with your image path
    # image = Image.open(image_path)

    # # Display the image in Streamlit
    # st.image(image, caption='Top 10 Feature Importances')

    st.header('Conclusions & Next Steps')
    st.markdown("""
    - The Random Forest and Gradient Boosting models performed similarly, with Gradient Boosting having a slight edge.
    - Location and room type were among the most important features affecting price.
    - Further refinement and model ensemble techniques could potentially improve the predictions.
    """)


elif selection == "Summary and Key Takeaways First Semester":
  
    # Insert your summary and key takeaways here
    st.markdown("""
    # Final Summary and Key Takeaways

    In wrapping up our project, we present a comprehensive overview of the diverse skill set harnessed and the methodologies applied. From data collection to predictive analytics, our project stands as a testament to a well-rounded analytical endeavor.

    ## Core Competencies and Methodologies:
    - **Data Acquisition**: Engaged with robust data sources to build a solid analytical foundation.
    - **Cleaning & Preprocessing**: Employed meticulous data cleaning to ensure the integrity of our analyses.
    - **Exploratory Data Analysis (EDA)**: Performed in-depth EDA to uncover trends, detect outliers, and identify influential variables.
    - **Visualization**: Developed interactive visualizations to bring data narratives to life.
    - **Feature Engineering**: Crafted meaningful features to enrich the dataset and enhance model efficacy.
    - **Predictive Modeling**: Applied sophisticated machine learning algorithms to forecast trends and pricing.
    - **Interactive Dashboarding**: Utilized Streamlit to create a dynamic and user-friendly dashboard.
    - **Collaboration & Version Control**: Embraced GitHub for seamless project management and version control, showcasing our proficiency in contemporary collaborative development.

    The skills and insights gained from this project not only underscore the technical aspects of data science but also highlight the importance of collaborative and iterative processes in producing actionable insights.
    """)
    
    col1, col2, col3 = st.columns([1,6,1])
    with col2:
        st.image('github.png')

    # Highlight the GitHub skills learned
    st.markdown("## The last significant achievement this semester was learning to collaborate and manage projects using GitHub, an essential skill for software development and version control.")
