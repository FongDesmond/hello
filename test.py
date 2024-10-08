import streamlit as st
from joblib import load
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load the pre-trained classifier model (which includes the vectorizer)
clf_loaded = load('clf.joblib')

def main():
    st.title("Sentiment Analysis App")
    st.write("Enter a review or upload a text file/CSV file to predict sentiments.")

    with st.sidebar:
        user_input = st.text_area("Enter a review:")
        uploaded_file = st.file_uploader("Or upload a text, CSV, or Excel file:", type=['txt', 'csv', 'xlsx'])

    # Process the uploaded file or text input
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            content = uploaded_file.read().decode("utf-8")
            reviews = content.split('\n')
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file)
            reviews = df.iloc[:, 0].tolist() 
        elif uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            reviews = df.iloc[:, 0].tolist()
    else:
        reviews = [user_input] if user_input else []

    # Ensure all reviews are strings
    reviews = [str(review) for review in reviews]

    if st.sidebar.button('Predict'):
        if reviews:
            predict_and_display(reviews)
        else:
            st.error("Please enter a review or upload a file for prediction.")

def predict_and_display(reviews):
    # Use the vectorizer from the pre-trained classifier model to transform the input
    transformed_reviews = clf_loaded.named_steps['vectorizer'].transform(reviews)

    # Predict using the classifier
    predictions = clf_loaded.predict(transformed_reviews)
    
    # Map predictions to sentiment labels
    sentiments = ["Positive" if res else "Negative" for res in predictions]
    
    # Display results in a table
    result_df = pd.DataFrame({
        'Review': reviews,
        'Sentiment': sentiments
    })

    with st.expander("Show Prediction Results"):
        st.table(result_df)

    # Display a histogram of the results if there are multiple reviews
    if len(reviews) > 1:
        st.write("Histogram of Predictions:")
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        pd.Series(sentiments).value_counts().plot(kind='bar', ax=ax, color=['blue']).set_title("Sentiment Distribution")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.tight_layout()
        st.pyplot(fig)

if __name__ == '__main__':
    main()
