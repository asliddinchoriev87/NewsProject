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

# Define 50 keywords for each category
oversea_keywords = [
    'international', 'foreign', 'global', 'overseas', 'worldwide', 'USA', 'Europe', 'China', 
    'Japan', 'Germany', 'Australia', 'Russia', 'France', 'UK', 'Canada', 'Brazil', 'India', 
    'Africa', 'Middle East', 'Latin America', 'North America', 'United States', 'Western', 
    'Eastern', 'Asia', 'Pacific', 'UN', 'United Nations', 'World Trade', 'World Bank', 
    'OECD', 'NATO', 'G7', 'G20', 'IMF', 'World Health Organization', 'climate change', 
    'global market', 'international relations', 'diplomacy', 'foreign policy', 'immigration', 
    'export', 'import', 'foreign trade', 'currency exchange', 'global economy', 
    'international travel', 'tourism', 'visa', 'embassy', 'global politics'
]

south_korea_keywords = [
    'Seoul', 'South Korea', 'K-pop', 'Hangeul', 'Busan', 'Incheon', 'Gyeonggi', 'Daegu', 
    'Gwangju', 'Daejeon', 'Jeju', 'Hallyu', 'Samsung', 'LG', 'Hyundai', 'SK Telecom', 'Korean War', 
    'North Korea', 'DMZ', 'Gangnam', 'K-drama', 'Korean cuisine', 'kimchi', 'bibimbap', 
    'bulgogi', 'hanbok', 'Korea University', 'Yonsei', 'KAIST', 'POSTECH', 'Korean language', 
    'K-beauty', 'K-culture', 'K-fashion', 'Sejong', 'Jinro', 'Soju', 'Taekwondo', 
    'Korean peninsula', 'Han River', 'Gyeongbokgung', 'Dongdaemun', 'Namdaemun', 'Korean politics', 
    'Moon Jae-in', 'Yoon Suk-yeol', 'Korean history', 'Korean government', 'Korean economy', 
    'Pyeongchang', 'Korean traditional music'
]
   
# Make sure the SummarizationModel class is defined or imported
class SummarizationModel:
    def __init__(self, model_name):
        self.model_name = model_name

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

        # Initialize scores for each category
        oversea_score = 0
        south_korea_score = 0

        # Convert the text to lowercase for case-insensitive matching
        text = text.lower()

        # Check for keyword matches
        for keyword in oversea_keywords:
            if keyword.lower() in text:
                oversea_score += 1

        for keyword in south_korea_keywords:
            if keyword.lower() in text:
                south_korea_score += 1

        # Determine the category with the higher score
        if south_korea_score > oversea_score:
        return "South Korea"
        elif oversea_score > south_korea_score:
        return "Oversea"
        else:
        return "Unclassified"

         # Display the predicted category
        st.subheader("Predicted Category:")
        st.write(predicted_category)
        st.write(f"The text is classified as country : **{category}**")

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

        # Join the keywords and their scores into one line
        keywords_line = ', '.join([f"{keyword}: {score:.2f}" for keyword, score in keywords])
        st.write(f"Keywords: {keywords_line}")

        
        # keywords = summarization_model.extract_top_keywords(text)
        # for keyword, score in keywords:
        #     st.write(f"{keyword} => Score: {score:.4f}")
        # else:
        #     st.write("Please enter some text to analyze.")
