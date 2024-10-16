import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your pre-trained NMF model and vectorizer
with open('nmf_model.pkl', 'rb') as model_file:
    nmf_model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@st.cache(allow_output_mutation=True)
def load_model():
    return pipeline("summarization", model="sshleifer/bart-tiny-random")

summarizer = load_model()
# Load the summarization model
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

        # Step 4: Summarize the article
        summary = summarization_model.summarizer(text, max_length=50, min_length=25, do_sample=False)
        
        # Display the predicted category
        st.subheader("Predicted Category:")
        st.write(predicted_category)

        # Display the summary
        st.subheader("Summary:")
        st.write(summary[0]['summary_text'])

        # Display top 5 keywords
        st.subheader("Top 5 Keywords:")
        keywords = summarization_model.extract_top_keywords(text)
        for keyword, score in keywords:
            st.write(f"{keyword} => Score: {score:.4f}")
    else:
        st.write("Please enter some text to analyze.")