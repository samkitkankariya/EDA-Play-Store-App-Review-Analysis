# EDA - Play Store App Review Analysis
This project conducts an exploratory data analysis on the Google Play Store apps dataset.  It covers data loading, cleaning, and preprocessing, followed by univariate, bivariate and multivariate analysis.

# **Problem Statement**

The goal of this project is to analyze Play Store app data along with customer reviews to derive actionable insights that drive app engagement and success on the Android platform. The dataset comprises two main components:

1. **Play Store Apps Data**: This dataset includes information such as app category, ratings, reviews, size, installs, pricing, content rating, genres, and more.

2. **Customer Reviews Data**: The second dataset contains translated customer reviews of the Android apps, along with sentiment analysis metrics like sentiment polarity and subjectivity.

Explore and analyse the data to discover key factors responsible for app engagement and success.

#### **Define Your Business Objective?**

The business objective here is to gain actionable insights and make data-driven decisions to improve app performance and user satisfaction in the Google Play Store. This includes understanding user sentiments, analyzing app ratings, exploring category-wise trends, and identifying factors influencing app installs and reviews.

---

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

---

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
(data_df.columns)
Index(['App', 'Category', 'Rating', 'Reviews', 'Size', 'Installs', 'Type',
       'Price', 'Content Rating', 'Genres', 'Last Updated', 'Current Ver',
       'Android Ver'],
      dtype='object')

(reviews_df.columns)
Index(['App', 'Translated_Review', 'Sentiment', 'Sentiment_Polarity',
       'Sentiment_Subjectivity'],
      dtype='object')
```

3. **Unique Values:**
   - Identify unique values within each column (excluding `App`, `Reviews`, `Rating`, `Size`, `Price`, and `Installs`) using a loop to iterate through columns and displaying unique values.

---

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

3. **Missing Values:**
   - Identify missing value counts in both datasets using `.isna().sum()`.

   ```
   Main Data: 
   Rating         1474
   Type              1
   Current Ver       8
   Android Ver       2
   dtype: int64
   
   User Reviews: 
   Translated_Review         26868
   Sentiment                 26863
   Sentiment_Polarity        26863
   Sentiment_Subjectivity    26863
   dtype: int64
   ```

4. **Duplicate Values:**
   - Check for duplicate rows in `data_df` and `reviews_df` using `.duplicated()`.
   - Count the number of duplicate rows in each dataset.
  
   ``` python
   # Check for duplicate rows in data_df
   duplicate_rows_data_df = data_df[data_df.duplicated()]
   
   # Check for duplicate rows in reviews_df
   duplicate_rows_reviews_df = reviews_df[reviews_df.duplicated()]
   
   # Get the count of duplicate rows in each dataset
   num_duplicates_data_df = len(duplicate_rows_data_df)
   num_duplicates_reviews_df = len(duplicate_rows_reviews_df)
   ```
   ```
   Number of duplicate rows in data_df: 483
   Number of duplicate rows in reviews_df: 33616
   ```
---

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
---

### Data Wrangling

- **Handling Missing Values:**
   - The Rating column contains 1470 NaN values in the entire dataset. It is not practical to drop these rows because by doing so, we will loose a large amount of data, which may impact the final quality of the analysis.

   - The NaN values in this case can be imputed by the aggregate (mean or median) of the remaining values in the Rating column.

```python
# Finding mean and median in the Rating column excluding the NaN values.

mean_rating = round(data_df['Rating'].mean(),3)
median_rating = data_df['Rating'].median()
[mean_rating, median_rating]
```
```
[4.192, 4.3]
```
*Visualization of distribution of rating using displot and detecting the outliers through boxplot.*

```python
fig, ax = plt.subplots(2,1, figsize=(12,7))
sns.histplot(data=data_df, x = 'Rating', color = 'firebrick', kde=True, ax=ax[0])
sns.boxplot(x='Rating', data=data_df, ax=ax[1])
```

 ![image](https://github.com/samkitkankariya/EDA-Play-Store-App-Review-Analysis/assets/31250827/1319b410-1dc0-4e6a-8b86-1e969c4d7b62)

*Findings : *
* The mean of the average ratings (excluding the NaN values) comes to be 4.2.

* The median of the entries (excluding the NaN values) in the 'Rating' column comes to be 4.3. From this we can say that 50% of the apps have an average rating of above 4.3, and the rest below 4.3.
* From the distplot visualizations, it is clear that the ratings are left skewed.
* We know that if the variable is skewed, the mean is biased by the values at the far end of the distribution. Therefore, the median is a better representation of the majority of the values in the variable.
* Hence we will impute the NaN values in the Rating column with its median.

```python
# Replace missing values with the median
data_df['Rating'].fillna(median_rating, inplace=True)

#One Type value is null so remove it
data_df.dropna(subset = 'Type', axis = 0, inplace = True)

'''
Since the NaN values in the Current Ver & Android Ver column cannot be replaced by any particular value, and, since there are only 8 and 2 rows respectively which contain NaN values in this column it can be be dropped.
'''
data_df.dropna(subset = ['Current Ver', 'Android Ver'], axis = 0, inplace = True)

#In User Reviews data we drop the rows which dont have any reviews
reviews_df.dropna(subset = 'Translated_Review', axis = 0, inplace = True)
```
  
- **Duplicates:**
```python
data_df['App'].value_counts()
```
```
App
ROBLOX                                                9
CBS Sports App - Scores, News, Stats & Watch Live     8
Candy Crush Saga                                      7
8 Ball Pool                                           7
ESPN                                                  7
                                                     ..
Meet U - Get Friends for Snapchat, Kik & Instagram    1
U-Report                                              1
U of I Community Credit Union                         1
Waiting For U Launcher Theme                          1
iHoroscope - 2018 Daily Horoscope & Astrology         1
Name: count, Length: 9648, dtype: int64
```

```python
# dropping duplicates from the 'App' column.
data_df.drop_duplicates(subset = 'App', inplace = True)
data_df.shape
```

*Handling duplicates of User Reviews Dataset:*
```python
# Print the shape of the cleaned dataset before and after removing duplicates
print("Shape of reviews_df before removing duplicates:", reviews_df.shape)

# Check for duplicate rows in reviews_df
duplicate_rows_reviews_df = reviews_df[reviews_df.duplicated()]
num_duplicates_reviews_df = len(duplicate_rows_reviews_df)
print(f"Number of duplicate rows in reviews_df: {num_duplicates_reviews_df}")

# Remove duplicate rows from reviews_df based on all columns
reviews_df = reviews_df.drop_duplicates()

print("Shape of cleaned_reviews_df after removing duplicates:", reviews_df.shape)
```
```
Shape of reviews_df before removing duplicates: (37427, 5)
Number of duplicate rows in reviews_df: 7735
Shape of cleaned_reviews_df after removing duplicates: (29692, 5)
```
#### Summary
1. **Handling Missing Values**:
   - Filled missing values in the 'Rating' column with the median value.
   - Removed rows with missing values in the 'Type' column from 'data_df'.
   - Removed rows with missing values in the 'Translated_Review' column from `reviews_df`.

2. **Removing Duplicates**:
   - Grouped `data_df` by various columns and calculated the mean of 'Installs'.
   - Sorted `data_df` by 'Reviews' in descending order and dropped duplicate rows based on the 'App' column.
   - Removed duplicate rows from `reviews_df` based on all columns.

3. **Insights**:
   - The median value was used to fill missing values in the 'Rating' column, ensuring no data loss in an important variable.
   - Removing rows with missing 'Type' values in `data_df` ensures data integrity and completeness.
   - The cleaning process in `reviews_df` removes redundant data, ensuring each review entry is unique.
   - After cleaning, the datasets are ready for analysis without missing values or duplicate entries, providing accurate insights.

These manipulations and insights contribute to data quality and prepare the datasets for meaningful analysis, ensuring reliable results and actionable conclusions.

### Data Vizualization, Storytelling & Experimenting with charts : Understand the relationships between variables

**Summary of Analysis Charts:**

1. Number of Apps per Category
   - Plot: Bar chart
   - Insight: Identified the distribution of apps across different categories, highlighting popular and less popular categories.

2. Number of Installs per App Category
   - Plot: Bar chart
   - Insight: Revealed which app categories have the highest number of installations, indicating user preferences and market demand.

3. Rating Distribution
   - Plot: Histogram
   - Insight: Showed the spread of app ratings, including the concentration of ratings around certain values and any unusual patterns.

4. Rating Groups
   - Plot: Bar chart
   - Insight: Grouped ratings into categories, providing a clearer understanding of the distribution of ratings across different groups.

5. Reviews, Size, Installs, Price vs Rating
   - Plot: Scatter plot
   - Insight: Explored relationships between app features like reviews, size, installs, price, and ratings, highlighting any correlations or trends.

6. Exploring App Ratings by Size and Category
   - Plot: Box plot
   - Insight: Investigated how app ratings vary based on their size and category, identifying potential factors influencing user ratings.

7. Distribution of Paid and Free Apps
   - Plot: Pie chart
   - Insight: Showed the proportion of paid and free apps, indicating the market dynamics between paid and free offerings.

8. Top Apps of Paid Type
   - Plot: Bar chart
   - Insight: Identified the top apps in the paid category, offering insights into popular paid apps and their characteristics.

9. Content Rating Distribution
   - Plot: Pie chart
   - Insight: Illustrated the distribution of content ratings among apps, highlighting the diversity of content available.

10. Distribution of Apps by Rating, Size, and Type
    - Plot: Scatter plot
    - Insight: Explored how apps are distributed based on their ratings, size, and type (paid or free), revealing patterns in app characteristics.

11. Correlation Heatmap
    - Plot: Heatmap
    - Insight: Visualized correlations between different app features, such as reviews, installs, price, and ratings, indicating potential relationships.

12. Different Distributions in User Review Data
    - Plot: Multiple histograms
    - Insight: Examined various distributions within user review data, including sentiment polarity, subjectivity, and other aspects.

13. Correlation between 2 datasets
    - Plot: Scatter plot
    - Insight: Investigated correlations between data from two datasets, providing insights into potential connections or dependencies.

14. Percentage of Review Sentiments
    - Plot: Pie chart
    - Insight: Analyzed the distribution of review sentiments (positive, neutral, negative), offering an overview of user sentiments towards apps.

15. Distribution of Subjectivity
    - Plot: Histogram
    - Insight: Explored the distribution of subjectivity in app reviews, indicating the extent of personal opinions and emotions expressed.

16. Is sentiment_subjectivity proportional to sentiment_polarity?
    - Plot: Scatter plot
    - Insight: Explored the relationship between sentiment subjectivity and polarity, investigating whether more subjective reviews tend to have more extreme polarities.
---

### Business Solution for App Performance and User Satisfaction

1. **Category Optimization for Maximum Impact**
   - Focus resources and marketing efforts on top-performing categories like Communication and Social, which exhibit high user engagement, positive sentiment, and significant installs.
   - Tailor app features, updates, and promotional campaigns to resonate with user preferences within these dominant categories, ensuring maximum impact and user satisfaction.

2. **Review and Ratings Enhancement**
   - Develop targeted strategies to boost app ratings, leveraging the observed positive correlation (0.6) between ratings and installs.
   - Implement in-app prompts and incentives for satisfied users to leave reviews, while promptly addressing negative feedback to demonstrate responsiveness and commitment to user satisfaction.

3. **Pricing Strategy Refinement**
   - Optimize pricing strategies based on the slight negative correlation (-0.09) between app prices and ratings/reviews.
   - Conduct A/B testing and introduce flexible pricing tiers or promotional offers to attract price-sensitive users without compromising perceived value, thereby enhancing user acquisition and retention.

4. **Sentiment-Driven Feature Enhancements**
   - Prioritize feature enhancements based on sentiment analysis insights to address user satisfaction and pain points effectively.
   - Capitalize on positive sentiment themes to strengthen app features that resonate with users, while addressing negative sentiments to improve overall user experience and mitigate churn.

5. **Marketing Messaging Alignment**
   - Align marketing messaging with sentiment trends to reinforce positive perceptions and attract new users.
   - Ensure consistency between marketing campaigns and user sentiments, maintaining authenticity and trust to drive user engagement and loyalty.

6. **Continuous Monitoring and Agile Iterations**
   - Implement a robust feedback loop and agile development approach to continuously monitor user feedback, sentiment trends, and app performance metrics.
   - Iterate and adapt app features, marketing strategies, and pricing models based on real-time insights, ensuring ongoing improvement, relevance, and competitiveness in the Google Play Store ecosystem.

By implementing these targeted strategies informed by data-driven insights from the combined analysis, the client can optimize app performance, enhance user satisfaction, and drive sustainable business growth and competitive advantage in the dynamic and competitive Google Play Store environment.

---
### Conclusion

In conclusion, the analysis underscores the importance of understanding user preferences, focusing on app quality, and aligning with market trends to succeed in the competitive landscape of the Google Play Store. By leveraging these insights, developers and businesses can make informed decisions to optimize app performance, enhance user satisfaction, and drive sustainable growth in the Android app market.
