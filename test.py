import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the data
data = pd.read_csv('spam.csv')

# Clean the data and add 'Spam' column
data['Spam'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data.Message, data.Spam, test_size=0.25)

# Using CountVectorizer explicitly
cv = CountVectorizer(stop_words='english')

# Transform the data using CountVectorizer
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

# Print the dimensions of the transformed data
print(X_train_cv.toarray().shape)

# Initialize the Naive Bayes model
nb = MultinomialNB()

# Train the model with transformed data
nb.fit(X_train_cv, y_train)

# Make predictions on the test data
y_pred = nb.predict(X_test_cv)

# Evaluate the model
accuracy = nb.score(X_test_cv, y_test)
print(f'Model accuracy: {accuracy}')

# Save the model and vectorizer using joblib
joblib.dump(nb, 'nb_model.joblib')
joblib.dump(cv, 'vectorizer.joblib')

# Plotting confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, xticklabels=['predicted_ham', 'predicted_spam'],
            yticklabels=['actual_ham', 'actual_spam'], annot=True, fmt='d',
            annot_kws={'fontsize': 20}, cmap="YlGnBu")
plt.show()

# Extracting values from the confusion matrix
true_neg, false_pos, false_neg, true_pos = cm.ravel()

# Calculating metrics
accuracy = round((true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg), 3)
precision = round((true_pos) / (true_pos + false_pos), 3)
recall = round((true_pos) / (true_pos + false_neg), 3)
f1 = round(2 * (precision * recall) / (precision + recall), 3)

# Displaying the evaluation metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
