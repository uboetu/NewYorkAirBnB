import pandas as pd
import folium
import numpy as np
import missingno as msno
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt
from folium.plugins import MarkerCluster
import streamlit as st
from streamlit_folium import folium_static
from folium import Element
from ipywidgets import interact
from sklearn.preprocessing import OneHotEncoder

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
    st.title("Data Exploration AirBnB New York City")

    st.markdown("""
                The New York City dataset, sourced from Kaggle. 
                This dataset serves as the foundation for our project, where our primary objective is to create an interactive dashboard for exploring and visualizing key insights from the AirBnB market in New York City.
                """)

    st.header('Data exploring')
    st.markdown("""
               The New York City AirBnB dataset contains all the data needed information to find out more about hosts, geographical availability, necessary metrics to make predictions and draw conclusions.
                For reverence the first 5 rows of the dataset has been displayed.
                """ )
    
    df = pd.read_csv('AB_NYC_2019.csv')
    st.write(df.head())

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




    
elif selection == "Prepare: Data Cleaning and Feature Engineering":
    
    Q1 = dataset['price'].quantile(0.25)
    Q3 = dataset['price'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset_no_outliers = dataset[(dataset['price'] >= lower_bound) & (dataset['price'] <= upper_bound)]

    summary_no_outliers = {
    'Initial data size': dataset.shape,
    'New data size': dataset_no_outliers.shape,
    'Number of outliers removed': dataset.shape[0] - dataset_no_outliers.shape[0]}
    summary_df = pd.DataFrame(list(summary_no_outliers.items()), columns=['Metric', 'Value'])

    st.markdown("""
    ## Outlier Detection and Removal
    Outlier detection is a crucial step in data preprocessing, particularly for price-related data. 
    Outliers can significantly skew our analysis, leading to inaccurate models or misinformed decisions. 
    Below is a summary of the dataset before and after the removal of outliers from the 'price' variable.
    """)

    st.subheader('Outlier Removal Summary')
    st.table(summary_df)


    # Visualizations
    st.markdown("""
    ### Price Distribution Before and After Outlier Removal
    The following plots show the distribution of the 'price' variable before and after the removal of outliers.
    This visual comparison helps to understand the effect of outlier removal on the data distribution.
    """)

    # Set up the figure layout
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Original Price Distribution
    sns.boxplot(y=dataset['price'], ax=ax[0])
    ax[0].set_title('Original Price Distribution')
    ax[0].set_ylabel('Price')

    # Price Distribution Without Outliers
    sns.boxplot(y=dataset_no_outliers['price'], ax=ax[1])
    ax[1].set_title('Price Distribution Without Outliers')
    ax[1].set_ylabel('Price')

    # Display the plots
    st.pyplot(fig)

    st.markdown("""
    ### Handling Missing Values
    Missing values can introduce bias and affect the model's performance. 
    Here, we fill missing values for reviews-related features with zero, 
    assuming no reviews have been made yet.
    """)
    dataset_no_outliers['reviews_per_month'].fillna(0, inplace=True)
    dataset_no_outliers['review_frequency'].fillna(0, inplace=True)
    dataset_no_outliers['days_since_last_review'].fillna(0, inplace=True)
    dataset_no_outliers['review_to_availability_ratio'].fillna(0, inplace=True)

    # Dropping unnecessary identifier columns
    st.markdown("""
    ### Dropping Unnecessary Columns
    Columns that serve as identifiers, which are not useful for modeling, are removed to streamline the dataset.
    """)
    cols_to_drop = ['id', 'name', 'host_id', 'host_name', 'last_review']
    data_cleaned = dataset_no_outliers.drop(columns=cols_to_drop)

    # Encoding categorical variables using One-Hot Encoding
    st.markdown("""
    ### Encoding Categorical Variables
    Categorical variables are transformed using One-Hot Encoding to convert them into a format 
    that can be provided to machine learning algorithms to do a better job in prediction.
    """)
    categorical_cols = ['neighbourhood_group', 'neighbourhood', 'room_type']
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    # Applying OneHotEncoder
    data_encoded = pd.DataFrame(one_hot_encoder.fit_transform(data_cleaned[categorical_cols]))

    # OneHotEncoder removes index; put it back
    data_encoded.index = data_cleaned.index

    # Remove categorical columns (will replace with one-hot encoding)
    num_data = data_cleaned.drop(columns=categorical_cols)

    # Add one-hot encoded columns to numerical features
    data_prepared = pd.concat([num_data, data_encoded], axis=1)

    # Display the first few rows of the prepared dataset
    st.markdown("""
    ### Final Dataset for Modeling
    The dataset is now clean, with all categorical variables encoded and ready for modeling. 
    Here's how the prepared dataset looks:
    """)


    # Assigning meaningful names to the encoded categorical features
    st.markdown("""
    ### Renaming Encoded Categories
    The one-hot encoded categorical variables are renamed for clarity.
    """)
    categories = one_hot_encoder.get_feature_names_out(input_features=categorical_cols)
    data_encoded.columns = categories

    # Combine the numerical and encoded categorical columns
    data_prepared = pd.concat([num_data, data_encoded], axis=1)

    # Scaling the numerical features
    st.markdown("""
    ### Feature Scaling
    Numerical features are scaled to have a mean of zero and a standard deviation of one. 
    This standardization of ranges is a common requirement for many machine learning estimators.
    """)

    # Define numerical columns to scale (excluding the price since it's the target variable)
    numerical_cols = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 
                    'reviews_per_month', 'calculated_host_listings_count', 
                    'availability_365', 'review_frequency', 
                    'days_since_last_review', 'distance_to_nearest_subway']

    # Initialize a scaler
    scaler = StandardScaler()

    # Scale the numerical columns
    data_prepared[numerical_cols] = scaler.fit_transform(data_prepared[numerical_cols])

    # Splitting the data into training and testing sets
    st.markdown("""
    ### Data Splitting
    The data is split into training and testing sets to evaluate the performance of machine learning models.
    """)

    # Define the target variable (dependent variable) as y
    y = data_prepared['price']
    X = data_prepared.drop('price', axis=1)

    # Split the data - 80% for training and 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Display the first few rows of the scaled and prepared dataset
    st.markdown("""
    ### Scaled and Prepared Dataset
    Here's how the prepared dataset looks after scaling the numerical features:
    """)
    st.write(data_prepared.head())

    # Removing any unnamed columns that might have been added during the data preparation
    if 'Unnamed: 0' in data_prepared.columns:
        data_prepared.drop('Unnamed: 0', axis=1, inplace=True)



    st.write(data_prepared.head())

    st.dataframe(dataset[['days_since_last_review', 'is_superhost', 'review_to_availability_ratio']].head())




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


elif selection == "data3":
    st.title("gg")
    st.subheader("ggggg")
