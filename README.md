# Toxic-Tweets-Dataset-NLP-Problem
This dataset has a collection of Tweets. Its labelled as Toxic - 1, Non toxic - 0. Apply the NLP  methods to predict the toxicity of the tweets.

# Libraries Used:

pandas: Used for data manipulation and analysis.

scikit-learn: Used for implementing machine learning models, preprocessing and metrics evaluation.

matplotlib: Used for data visualization.

seaborn: Used for enhanced data visualization.

nltk: Used for natural language processing tasks like tokenization, lemmatization and stopwords removal.

# Data:

The dataset was downloaded from the following Kaggle, https://www.kaggle.com/datasets/ashwiniyer176/toxic-tweets-dataset.

The script assumes that the tweet data is available in a pandas DataFrame named df with two columns:

'tweet': Contains the raw tweet text.
'Toxicity': Contains the binary label (0 for non-toxic and 1 for toxic) indicating the toxicity of each tweet.

# A step-by-step explanation of the code using Toxic Tweets dataset and generating various visualizations
1. The code performs text classification using various machine learning models for toxicity detection in tweets.
2. Function Description:
`def clean_text(text):`
- The function takes a text input as a parameter and performs various preprocessing steps to clean the text.
- It converts the text to lowercase to ensure uniformity in the data.
- Punctuation and special characters are removed using regular expressions, leaving only alphabetic characters and spaces.
- The text is tokenized, splitting it into individual words.
- Stopwords, common words like 'the', 'a', 'an', etc., are removed from the tokenized list.
- Lemmatization is applied to each word to convert it into its base or dictionary form (e.g., 'running' to 'run', 'better' to 'good').
- The cleaned tokens are joined back together to form the final preprocessed text.
- The cleaned text is returned as the output of the function.
3. It uses the Bag of Words (BoW) and TF-IDF (Term Frequency-Inverse Document Frequency) vectorization techniques to convert the textual data into numerical features.
4. The data is split into training and test sets using a subset of 10,000 rows or more depending on RAM size.
5. The code defines five different models for classification: DecisionTrees, RandomForest, Naive-Bayes, K-NN Classifier and Support Vector Machine (SVM).
6. Each model is trained using both BoW and TF-IDF features separately.
7. Metrics such as precision, recall, F1-score and ROC-AUC are calculated for each model and vectorization technique combination.
8. Bar charts are generated to visualize the precision, recall and F1-score for each model and vectorization technique.
9. Confusion matrices and ROC curves are plotted to evaluate the performance of each model.
10. All visualizations and metrics are saved in a single PDF named `combined_report.pdf`.
  
# Conclusion
Based on the results, the script infers that Bag of Words (BoW) vectorization performs better in models like DecisionTrees and RandomForest, which are not sensitive to word context but rely more on individual word frequencies. On the other hand, TF-IDF vectorization is more suitable for models like Naive-Bayes, K-NN Classifier and SVM, where the importance of words in the context of the entire corpus plays a crucial role in classification.
