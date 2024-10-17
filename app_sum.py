import streamlit as st
import pickle
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BartForConditionalGeneration, BartTokenizer
from keybert import KeyBERT

# Load your pre-trained NMF model and vectorizer
with open('nmf_model.pkl', 'rb') as model_file:
    nmf_model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

model_name = "sshleifer/bart-tiny-random"

# Make sure the SummarizationModel class is defined or imported
class SummarizationModel:
    def __init__(self, model_name='sshleifer/bart-tiny-random'):
        self.model_name = model_name
        # Define any other attributes or methods

# Now load the pickled model
with open('summarization_model.pkl', 'rb') as sum_file:
    summarization_model = pickle.load(sum_file)

# Assuming you have a list or mapping of topics to categories
topics_to_categories = {0: "Sports", 1: "Politics", 2: "Business", 3: "Entertainment", 4: "Technology"}

# Streamlit app
st.title("News Text Summarization & Categorization")

# Text input
text = st.text_area("Enter news text to analyze", height=300)

if st.button("Analyze Text"):
    if text:
        # Step 1: Transform the text using the TF-IDF vectorizer
        text_tfidf = vectorizer.transform([text])

        # Step 2: Get topic distribution from the NMF model
        topic_distribution = nmf_model.transform(text_tfidf)

        # Step 3: Get the most relevant topic (category) based on highest score
        predicted_topic = topic_distribution.argmax()
        predicted_category = topics_to_categories.get(predicted_topic, "Unknown")

         # Display the predicted category
        st.subheader("Predicted Category:")
        st.write(predicted_category)

        # Step 4: Summarize the article
        # Load the summarization pipeline
        summarization_model = pipeline("summarization")
        summary = summarization_model(text, max_length=60, min_length=40, temperature=0.7, top_p=0.9)
        
        # Display the summary
        st.subheader("Summary:")
        st.write(summary[0]['summary_text'])


        # Extract top 5 keywords using KeyBERT
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), top_n=5)
    
        # Display top 5 keywords
        st.subheader("Top 5 Keywords:")
        for keyword, score in keywords:
            st.write(f"{keyword} : {score:.2f * 100}")

        
        # keywords = summarization_model.extract_top_keywords(text)
        # for keyword, score in keywords:
        #     st.write(f"{keyword} => Score: {score:.4f}")
        # else:
        #     st.write("Please enter some text to analyze.")
