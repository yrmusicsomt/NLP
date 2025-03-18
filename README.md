# Natural Language Processing Project

This project focuses on classifying Yelp reviews as either 1 or 5-star categories based on the textual content of the reviews. The classification task leverages the Yelp Review Dataset from Kaggle. Below, you'll find an overview of the dataset, the exploratory data analysis (EDA), and the modeling pipeline used to achieve this goal.

---

## Dataset Overview
The dataset contains Yelp reviews for various businesses, with the following key columns:

- **stars**: Number of stars (1 to 5) assigned by the reviewer to the business.
- **text**: Textual content of the review.
- **cool**: Number of "cool" votes received by the review.
- **useful**: Number of "useful" votes received by the review.
- **funny**: Number of "funny" votes received by the review.

### Additional Notes:
- The "stars" column serves as the target variable for classification.
- Reviews receiving 1-star and 5-star ratings are used for the classification task.
- A new column, **text length**, was created to measure the number of words in each review.

---

## Exploratory Data Analysis (EDA)

### Steps:
1. **Text Length Analysis**:
   - Histograms of text length were plotted for each star category.
   - Boxplots revealed text length distribution for different star ratings.

2. **Star Rating Distribution**:
   - Countplot to visualize occurrences of each star rating.

3. **Correlation Analysis**:
   - Grouped data by "stars" and calculated the mean of numeric columns.
   - Generated a heatmap to visualize correlations among features (e.g., cool, useful, funny, and text length).

---

## NLP Classification Task

### Data Preparation:
1. Extracted reviews with 1-star and 5-star ratings into a new dataframe.
2. Created features (**X**) and target labels (**y**).
3. Utilized `CountVectorizer` for text vectorization.

### Modeling Pipeline:
1. **Naive Bayes Model**:
   - Split data into training and test sets (70/30 split).
   - Trained a Multinomial Naive Bayes model and evaluated its performance using a confusion matrix and classification report.

2. **Pipeline with TF-IDF**:
   - Built a pipeline with `CountVectorizer`, `TfidfTransformer`, and `MultinomialNB`.
   - Although adding TF-IDF resulted in reduced model performance, it provided insights for further experimentation.

---

## Results

### Naive Bayes Model (Without TF-IDF):
- **Confusion Matrix**:
  ```
  [[159  69]
   [ 22 976]]
  ```
- **Classification Report**:
  ```
                precision    recall  f1-score   support

             1       0.88      0.70      0.78       228
             5       0.93      0.98      0.96       998

      accuracy                           0.93      1226
     macro avg       0.91      0.84      0.87      1226
  weighted avg       0.92      0.93      0.92      1226
  ```

### Pipeline with TF-IDF:
- **Confusion Matrix**:
  ```
  [[  0 228]
   [  0 998]]
  ```
- **Classification Report**:
  ```
                precision    recall  f1-score   support

             1       0.00      0.00      0.00       228
             5       0.81      1.00      0.90       998

      accuracy                           0.81      1226
     macro avg       0.41      0.50      0.45      1226
  weighted avg       0.66      0.81      0.73      1226
  ```

---

## Key Insights
1. The Naive Bayes model performed well without TF-IDF, achieving 93% accuracy.
2. Incorporating TF-IDF negatively impacted model performance, likely due to overemphasis on rare terms.

---

## Technologies and Libraries Used
- **Languages**: Python
- **Libraries**: pandas, numpy, matplotlib, seaborn, nltk, sklearn

---

## Future improvements:

- Experiment with alternative classifiers like SVM or Random Forest.
- Optimize hyperparameters for the pipeline components.
- Apply additional preprocessing steps (e.g., stemming or lemmatization).
- Use advanced vectorization techniques like word embeddings.

---

## Acknowledgments
Special thanks to [Kaggle](https://www.kaggle.com) for providing the Yelp Review Dataset.
