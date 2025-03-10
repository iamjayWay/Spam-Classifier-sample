# Link to dataset https://www.kaggle.com/datasets/wcukierski/enron-email-dataset/data

# Import Streamlit for building the web app interface
import streamlit as st
# Import joblib for loading the pre-trained pipeline model
import joblib
# Import pandas for data manipulation and displaying tables
import pandas as pd
# Import matplotlib.pyplot for creating visualizations (here, a pie chart)
import matplotlib.pyplot as plt
# Import the re module for regular expression operations
import re
# Import string to access string constants like punctuation
import string

# Define a simple set of stopwords (customizable)
stop_words = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
    'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
    "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn',
    "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
    'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

# Define a text preprocessing function that cleans input text.
def preprocess_text_re(text):
    """
    Preprocesses the input text by:
    - Removing non-alphanumeric characters (while retaining spaces)
    - Normalizing multiple spaces into one
    - Converting text to lowercase and trimming whitespace
    - Tokenizing and removing defined stopwords
    """
    # Remove non-alphanumeric characters (retain spaces)
    text = re.sub(r"[^\w\s]", " ", text)
    # Normalize multiple spaces to a single space
    text = re.sub(r"\s+", " ", text)
    # Trim whitespace and convert text to lowercase
    text = text.strip().lower()
    # Tokenize the text by splitting on whitespace
    words = text.split()
    # Remove stopwords from the tokenized words
    filtered_words = [word for word in words if word not in stop_words]
    # Join the filtered words back into a single string
    return " ".join(filtered_words)

# Define a function to check for unsafe links (URLs using HTTP instead of HTTPS)
def check_http_links(text):
    """
    Detects URLs in the input text and flags those that use HTTP instead of HTTPS.
    """
    # Define a regex pattern to match URLs
    url_pattern = r"http[s]?://(?:[a-zA-Z0-9$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    # Find all URLs in the text
    urls = re.findall(url_pattern, text)
    # Filter URLs that start with "http://"
    unsafe_links = [url for url in urls if url.startswith("http://")]
    return unsafe_links

# Define a function to detect the type of spam based on specific keywords.
def detect_spam_type(text):
    """
    Detects the type of spam based on specific keywords.
    Returns a descriptive string indicating the spam category.
    """
    text_lower = text.lower()
    promotional_keywords = {"free", "win", "prize", "congratulations", "gift", "selected", "bonus", "offer"}
    phishing_keywords = {"account", "bank", "verify", "password", "login", "security", "update"}
    financial_keywords = {"loan", "credit", "finance", "mortgage", "investment", "cash", "money"}
    scam_keywords = {"unsubscribe", "urgent", "confirm", "alert", "limited", "deal", "risk"}
    
    if any(keyword in text_lower for keyword in promotional_keywords):
        return "Promotional/Prize Spam"
    elif any(keyword in text_lower for keyword in phishing_keywords):
        return "Phishing Spam"
    elif any(keyword in text_lower for keyword in financial_keywords):
        return "Financial Spam"
    elif any(keyword in text_lower for keyword in scam_keywords):
        return "Scam Spam"
    else:
        return "General Spam"

# Define a function to detect the type of legitimate message based on keywords.
def detect_legit_type(text):
    """
    Detects the type of legitimate message based on keywords.
    Returns a descriptive string indicating the message category.
    This function considers personal, professional, official, social, and transactional messages.
    """
    text_lower = text.lower()
    
    # Expanded keywords for personal messages (casual conversations)
    personal_keywords = {"hi", "hello", "dear", "hey", "good morning", "good afternoon", "good evening", 
                         "how are you", "love", "miss", "bye", "talk soon", "see you", "take care"}
    # Expanded keywords for professional messages (work-related communications)
    professional_keywords = {"meeting", "schedule", "project", "report", "invoice", "order", "confirmation", 
                             "appointment", "deadline", "attached", "minutes", "update", "review", "client", "contract", "team"}
    # Keywords for official messages (formal notifications)
    official_keywords = {"government", "official", "announcement", "notice", "policy", "circular", "press", 
                         "release", "regulation", "public", "office", "department"}
    # Keywords for social messages (invitations, gatherings)
    social_keywords = {"party", "dinner", "lunch", "birthday", "wedding", "gathering", "hang out", "celebration", 
                       "coffee", "social", "get-together", "catch up"}
    # Keywords for transactional messages (orders, payments)
    transactional_keywords = {"payment", "receipt", "transaction", "bank", "statement", "delivery", "order", 
                              "shipment", "bill", "invoice", "confirm", "purchase"}
    
    categories = []
    if any(keyword in text_lower for keyword in personal_keywords):
        categories.append("Personal Message")
    if any(keyword in text_lower for keyword in professional_keywords):
        categories.append("Professional Message")
    if any(keyword in text_lower for keyword in official_keywords):
        categories.append("Official Message")
    if any(keyword in text_lower for keyword in social_keywords):
        categories.append("Social Message")
    if any(keyword in text_lower for keyword in transactional_keywords):
        categories.append("Transactional Message")
    
    # If multiple categories match, join them into a comma-separated string; otherwise, return a general label.
    if categories:
        return ", ".join(categories)
    else:
        return "General Legit Message"

# Load the pre-trained pipeline using joblib
try:
    pipeline = joblib.load('text_classifier_pipeline.pkl')
except Exception as e:
    st.error(f"Error loading pipeline: {e}")
    st.stop()

# Streamlit App UI Setup
st.title("Email/SMS Spam Classifier")
st.markdown("""
Welcome! Input text to predict classification and visualize the results.
This app will also flag unsafe links and identify the type of message.
""")

# Input text area for user to enter text
user_input = st.text_area("Enter the text for classification.", "")

# Process the input when the "Classify Text" button is clicked
if st.button("Classify Text"):
    if user_input.strip():
        # First, check for unsafe links in the input text
        unsafe_links = check_http_links(user_input)
        if unsafe_links:
            st.warning(f"The message contains unsafe links (HTTP): {', '.join(unsafe_links)}")
            st.write("### Prediction: spam")
            st.write("### This message is 100% spam due to unsafe links.")
            st.write("### Spam Type: General Spam (Flagged by unsafe link detection)")
        else:
            # Preprocess the input text using the custom preprocessing function
            preprocessed_input = preprocess_text_re(user_input)
            
            # Use the pre-trained pipeline to predict the classification
            prediction = pipeline.predict([preprocessed_input])[0]
            probabilities = pipeline.predict_proba([preprocessed_input])[0]
            
            # Determine the predicted label (spam or legit)
            prediction_label = "spam" if prediction == 1 else "legit"
            # Calculate the spam percentage (if spam, else 0%)
            spam_percentage = round(probabilities[1] * 100, 2) if prediction == 1 else 0
            
            # Display the prediction result and probability table
            st.markdown(f"### Prediction: {prediction_label}")
            st.write("### Prediction Probabilities:")
            prob_df = pd.DataFrame({'Class': ['legit', 'spam'], 'Probability': probabilities})
            st.table(prob_df)
            
            # Determine and display the type of message based on classification
            if prediction == 1:
                spam_type = detect_spam_type(user_input)
                st.write(f"### Spam Type: {spam_type}")
                st.write(f"### This message is {spam_percentage}% spam.")
            else:
                legit_type = detect_legit_type(user_input)
                st.write(f"### Legit Message Type: {legit_type}")
            
            # Display a pie chart for probability distribution
            st.write("### Probability Distribution:")
            fig, ax = plt.subplots()
            # Create a pie chart with labels and percentage annotations
            ax.pie(probabilities, labels=['legit', 'spam'], autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
            ax.axis('equal')  # Ensure the pie chart is circular
            st.pyplot(fig)
    else:
        st.error("Please enter some text!")
