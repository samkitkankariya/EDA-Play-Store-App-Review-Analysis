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

# **GitHub Link -**

https://github.com/samkitkankariya/EDA-Play-Store-App-Review-Analysis

# **Problem Statement**


**Problem Statement: Play Store App Analysis for Enhanced Engagement and Success**

The goal of this project is to analyze Play Store app data along with customer reviews to derive actionable insights that drive app engagement and success on the Android platform. The dataset comprises two main components:

1. **Play Store Apps Data**: This dataset includes information such as app category, ratings, reviews, size, installs, pricing, content rating, genres, and more.

2. **Customer Reviews Data**: The second dataset contains translated customer reviews of the Android apps, along with sentiment analysis metrics like sentiment polarity and subjectivity.

Explore and analyse the data to discover key factors responsible for app engagement and success.

#### **Define Your Business Objective?**

The business objective here is to gain actionable insights and make data-driven decisions to improve app performance and user satisfaction in the Google Play Store. This includes understanding user sentiments, analyzing app ratings, exploring category-wise trends, and identifying factors influencing app installs and reviews.


## ***1. Know Your Data***

Let's take a look at the data, which consists of two files:

- playstore data.csv: contains all the details of the applications on Google Play. There are 13 features that describe a given app.

- user_reviews.csv: contains 100 reviews for each app, most helpful first. The text in each review has been pre-processed and attributed with three new features: Sentiment (Positive, Negative or Neutral), Sentiment Polarity and Sentiment Subjectivity.

## Data Wrangling for Play Store Analysis

This is a README file for data wrangling processes applied to a dataset containing information on mobile applications from Google Play and corresponding user reviews. The wrangling steps aim to prepare the data for further analysis.

### Data Loading (**Replace with your code for Colab**)

**Note:** These instructions assume you've mounted your drive and have the data downloaded. Specific code for Colab mounting would need to be replaced here.

1. Load the `playstore.csv` file containing app details.
2. Load the `user_reviews.csv` file containing user reviews.

### Understanding the Data

1. **Dataframe Information:**
   - Use `data_df.info()` to view data types and missing value counts for the main dataset (`data_df`).
   - Use `reviews_df.info()` to view data types and missing value counts for the user reviews dataset (`reviews_df`).

2. **Columns and Description:**
   - Explore the column names using `data_df.columns` and `reviews_df.columns`.

3. **Unique Values:**
   - Identify unique values within each column (excluding `App`, `Reviews`, `Rating`, `Size`, `Price`, and `Installs`) using a loop to iterate through columns and displaying unique values.

### Data Cleaning

1. **Cleaning 'Reviews', 'Size', 'Installs', and 'Price' Columns (Data Type Conversion):**
   - **'Reviews'**:
     - Handle entries with 'M' (denoting Megabytes) by converting them to float values (e.g., '3.0M' becomes 3000000).
     - Update the `data_df['Reviews']` column directly using `.at`.
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

### Code (**Replace with your actual code**)

**Note:** Replace the placeholders below with your actual code for each data cleaning step.

```python
# ... your data cleaning code here ...
```

### Outputs (**Replace with your actual outputs**)

**Note:** Replace the placeholders below with the actual outputs from your code.

```
# ... your data cleaning outputs here ...
```

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
        

