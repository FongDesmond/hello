import streamlit as st
from joblib import load
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


lr_loaded = load('lr.joblib')
nb_loaded = load('nb.joblib')
svm_loaded = load('svm.joblib')
cv_loaded = load('cv.joblib')

def main():
    st.title("Sentiment Analysis App")
    st.write("Enter a review or upload a text file/CSV file to predict sentiments.")

    with st.sidebar:
        user_input = st.text_area("Enter a review:")
        uploaded_file = st.file_uploader("Or upload a text, CSV, or Excel file:", type=['txt', 'csv', 'xlsx'])


    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            content = uploaded_file.read().decode("utf-8")
            reviews = content.split('\n') 
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file)
            reviews = df.iloc[:,0].tolist() 
        elif uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            reviews = df.iloc[:,0].tolist()
    else:
        reviews = [user_input] if user_input else []

  
    if st.sidebar.button('Predict'):
        if reviews:
            predict_and_display(reviews)
        else:
            st.error("Please enter a review or upload a file for prediction.")

def predict_and_display(reviews):
    transformed_reviews = cv_loaded.transform(reviews)
    results_lr = lr_loaded.predict(transformed_reviews)
    results_nb = nb_loaded.predict(transformed_reviews)
    results_svm = svm_loaded.predict(transformed_reviews)
    
    sentiment_lr = ["Positive" if res else "Negative" for res in results_lr]
    sentiment_nb = ["Positive" if res else "Negative" for res in results_nb]
    sentiment_svm = ["Positive" if res else "Negative" for res in results_svm]
    
    result_df = pd.DataFrame({
        'Review': reviews,
        'LR Sentiment': sentiment_lr,
        'NB Sentiment': sentiment_nb,
        'SVM Sentiment': sentiment_svm
    })

    with st.expander("Show Prediction Results"):
        st.table(result_df)

   
    if len(reviews) > 1:
        st.write("Histogram of Predictions:")
        fig, ax = plt.subplots(3, 1, figsize=(8, 12))
        pd.Series(sentiment_lr).value_counts().plot(kind='bar', ax=ax[0], color=['blue']).set_title("LR Sentiment Distribution")
        pd.Series(sentiment_nb).value_counts().plot(kind='bar', ax=ax[1], color=['green']).set_title("NB Sentiment Distribution")
        pd.Series(sentiment_svm).value_counts().plot(kind='bar', ax=ax[2], color=['red']).set_title("SVM Sentiment Distribution")
        for a in ax:
            a.set_xlabel("Sentiment")
            a.set_ylabel("Count")
            a.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.tight_layout()
        st.pyplot(fig)

if __name__ == '__main__':
    main()
