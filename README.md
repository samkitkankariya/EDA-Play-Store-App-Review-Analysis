# EDA - Play Store App Review Analysis
This project conducts an exploratory data analysis on the Google Play Store apps dataset.  It covers data loading, cleaning, and preprocessing, followed by univariate, bivariate and multivariate analysis.

# **Project Summary -**



The project aims to analyze Play Store app data and customer reviews to derive actionable insights that drive app engagement and success on the Android platform. It involves two main datasets: Play Store Apps Data and Customer Reviews Data.

The Play Store Apps Data comprises information such as app category, ratings, reviews, size, installs, pricing, content rating, genres, and more. On the other hand, the Customer Reviews Data contains translated customer reviews of Android apps along with sentiment analysis metrics like sentiment polarity and subjectivity.

The analysis process includes several steps to make the data analysis-ready. Initially, the dataset is checked for rows and columns count, and information about the dataset is obtained. Data cleaning techniques are then applied to handle missing values, convert data types, and remove duplicates. For example, missing values in the 'Rating' column are filled with the median value, rows with missing 'Type' values are removed, and duplicates are dropped based on the 'App' column.

Insights derived from the analysis include:

1. **Trending Categories:** Categories like GAME, COMMUNICATION, and TOOL are dominating the trend charts in terms of user installs, showcasing high user engagement and satisfaction.

2. **Quality Over Quantity:** The emphasis on app quality is evident, especially among apps with high ratings. User engagement and positive feedback drive app popularity.

3. **User Engagement:** Apps with high review counts are primarily from categories like SOCIAL, COMMUNICATION, and GAME, indicating strong user interaction and satisfaction.

4. **Size and Pricing Dynamics:** While some outliers exist, most top-rated apps do not necessarily have large sizes or high prices, emphasizing that quality apps can be resource-efficient.

These insights guide the development of targeted strategies for app performance enhancement and user satisfaction:

- **Category Optimization for Maximum Impact:** Allocate resources and enhancements to top-performing categories like Communication and Social to capitalize on high engagement and positive feedback.
- **Review and Ratings Enhancement:** Implement strategies to boost ratings and reviews, address negative feedback promptly, and encourage user engagement.
- **Pricing Strategy Refinement:** Optimize pricing strategies based on user sentiments and engagement metrics to attract price-sensitive users without compromising value.

The project's success lies in its ability to translate data insights into actionable strategies that improve app performance, user satisfaction, and overall success on the Google Play Store platform.

# **Problem Statement**

The goal of this project is to analyze Play Store app data along with customer reviews to derive actionable insights that drive app engagement and success on the Android platform. The dataset comprises two main components:

1. **Play Store Apps Data**: This dataset includes information such as app category, ratings, reviews, size, installs, pricing, content rating, genres, and more.

2. **Customer Reviews Data**: The second dataset contains translated customer reviews of the Android apps, along with sentiment analysis metrics like sentiment polarity and subjectivity.

Explore and analyse the data to discover key factors responsible for app engagement and success.

#### **Define Your Business Objective?**

The business objective here is to gain actionable insights and make data-driven decisions to improve app performance and user satisfaction in the Google Play Store. This includes understanding user sentiments, analyzing app ratings, exploring category-wise trends, and identifying factors influencing app installs and reviews.


### Data Loading 

1. Load the `playstore.csv` file containing app details.
2. Load the `user_reviews.csv` file containing user reviews.

```python
from google.colab import drive
drive.mount('/content/drive')

# Load Dataset
data_df = pd.read_csv('your-drive-path/Play Store Data.csv')
reviews_df = pd.read_csv('your-drive-path/User Reviews.csv')
```

### Understanding the Data

Let's take a look at the data, which consists of two files:

- playstore data.csv: contains all the details of the applications on Google Play. There are 13 features that describe a given app.

- user_reviews.csv: contains 100 reviews for each app, most helpful first. The text in each review has been pre-processed and attributed with three new features: Sentiment (Positive, Negative or Neutral), Sentiment Polarity and Sentiment Subjectivity.


1. **Dataframe Information:**
   - Use `data_df.info()` to view data types and missing value counts for the main dataset (`data_df`).
   - Use `reviews_df.info()` to view data types and missing value counts for the user reviews dataset (`reviews_df`).

```
]
# Dataset Info
data_df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10841 entries, 0 to 10840
Data columns (total 13 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   App             10841 non-null  object 
 1   Category        10841 non-null  object 
 2   Rating          9367 non-null   float64
 3   Reviews         10841 non-null  object 
 4   Size            10841 non-null  object 
 5   Installs        10841 non-null  object 
 6   Type            10840 non-null  object 
 7   Price           10841 non-null  object 
 8   Content Rating  10840 non-null  object 
 9   Genres          10841 non-null  object 
 10  Last Updated    10841 non-null  object 
 11  Current Ver     10833 non-null  object 
 12  Android Ver     10838 non-null  object 
dtypes: float64(1), object(12)
memory usage: 1.1+ MB


reviews_df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 64295 entries, 0 to 64294
Data columns (total 5 columns):
 #   Column                  Non-Null Count  Dtype  
---  ------                  --------------  -----  
 0   App                     64295 non-null  object 
 1   Translated_Review       37427 non-null  object 
 2   Sentiment               37432 non-null  object 
 3   Sentiment_Polarity      37432 non-null  float64
 4   Sentiment_Subjectivity  37432 non-null  float64
dtypes: float64(2), object(3)
memory usage: 2.5+ MB

```

*By diagnosing the data frame, we know that:*

- There are 13 columns of properties with 10841 rows of data.
- Column 'Reviews', 'Size', 'Installs' and 'Price' are in the type of 'object'
- Values of column 'Size' are strings representing size in 'M' as Megabytes, 'k' as kilobytes and also 'Varies with devices'.
- Values of column 'Installs' are strings representing install amount with symbols such as ',' and '+'.
- Values of column 'Price' are strings representing price with symbol '$'.
Hence, we will need to do some data cleaning.

2. **Columns and Description:**
   - Explore the column names using `data_df.columns` and `reviews_df.columns`.

```

# Dataset Columns
data_df.columns
Index(['App', 'Category', 'Rating', 'Reviews', 'Size', 'Installs', 'Type',
       'Price', 'Content Rating', 'Genres', 'Last Updated', 'Current Ver',
       'Android Ver'],
      dtype='object')
```

3. **Unique Values:**
   - Identify unique values within each column (excluding `App`, `Reviews`, `Rating`, `Size`, `Price`, and `Installs`) using a loop to iterate through columns and displaying unique values.

### Data Cleaning

1. **Cleaning 'Reviews', 'Size', 'Installs', and 'Price' Columns (Data Type Conversion):**
   - **'Reviews'**:
     - Handle entries with 'M' (denoting Megabytes) by converting them to float values (e.g., '3.0M' becomes 3000000).
     - Convert the entire column to float using `.astype(float)`.
   - **'Size'**:
     - Identify and remove rows with the value '1,000+' due to data inconsistency.
     - Create a function `clean_sizes` to handle size representations ('M', 'k', and 'Varies with device').
     - Convert 'M' and 'k' to float values representing Megabytes.
     - Assign the cleaned sizes to the `data_df['Size']` column.
     - Convert the entire column to float using `.astype(float)`.
   - **'Installs'**:
     - Use regular expressions to replace '+' and ',' symbols with empty strings, making the data suitable for conversion to float.
     - Convert the `data_df['Installs']` column to float using `.astype(float)`.
   - **'Price'**:
     - Remove the dollar sign ('$') symbol using `.str.replace('$', '')`.
     - Convert the `data_df['Price']` column to float using `.astype(float)`.

2. **Cleaning User Reviews Data (Data Type Conversion):**
   - Convert the `reviews_df['Sentiment_Polarity']` column to float using `.astype(float)`.

3. **Handling Missing Values:**
   - Identify missing value counts in both datasets using `.isna().sum()`.
   - Analyze if imputation or removal is appropriate for missing values in each column.

4. **Handling Duplicate Values:**
   - Check for duplicate rows in `data_df` and `reviews_df` using `.duplicated()`.
   - Count the number of duplicate rows in each dataset.
   - Decide on an appropriate strategy to address duplicates (e.g., dropping or merging).

### Variables Description

**Description of Main - Dataset Columns**

1. App: Name of the mobile application.
2. Category: Category or genre of the application.
3. Rating: Average user rating of the application (on a scale of 1 to 5).
4. Reviews: Number of user reviews/ratings received for the application.
5. Size: Size of the application (in terms of storage space).
6. Installs: Number of times the application has been installed/downloaded.
7. Type: Type of the application (e.g., Free or Paid).
8. Price: Price of the application (if it's a paid app).
9. Content Rating: Content rating or maturity level of the application (e.g., Everyone, Teen, etc.).
10. Genres: Specific genres or sub-categories of the application.
11. Last Updated: Date when the application was last updated.
12. Current Ver: Current version of the application.
13. Android Ver: Minimum required Android version for the application to run.


**Description of User Reviews - Dataset Columns**

1. App: Name of the mobile application.
2. Translated_Review: Translated version of the user review for the application.
3. Sentiment: Sentiment analysis result of the review (e.g., Positive, Negative, Neutral).
4. Sentiment_Polarity: Numerical value indicating the sentiment polarity of the review.
5. Sentiment_Subjectivity: Numerical value indicating the subjectivity of the review (how subjective or objective it is).


### Data Wrangling Summary

- **Missing Values:**
   - Describe how missing values were handled in each column (e.g., imputation or removal).
   - Explain the rationale behind the chosen approach.
- **Duplicates:**
   - Describe how duplicate rows were addressed in each dataset.
   - Explain the chosen approach for handling duplicates.
- **Insights:**
   - Summarize the key findings from the data cleaning process.
   - How does the cleaning process improve data quality and preparation for analysis?

This data wrangling process prepares the Play Store app information and user review datasets for further analysis by ensuring data quality, consistency, and completeness. The cleaned datasets are now ready for exploration and insights generation.
        

