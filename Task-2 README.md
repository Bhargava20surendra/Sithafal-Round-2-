from bs4 import BeautifulSoup
import requests
import os
import pickle
import time
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Initialize LLM
try:
    llm_model = ChatGroq(temperature=0, groq_api_key="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", model_name="llama-3.1-70b-versatile")
except Exception as e:
    print(f"Error initializing LLM model: {e}")
    llm_model = None

store_file = "faiss_store_openai.pkl"

# Function to scrape content from a website
def scrape_website_content(url):
    try:
        # Send a request to the website
        response = requests.get(url)
        response.raise_for_status()

        # Parse the HTML content of the page
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract text from all paragraph tags (you can adjust this as needed)
        paragraphs = soup.find_all('p')
        text = '\n'.join([para.get_text() for para in paragraphs])
        return text
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve the webpage: {url}. Error: {e}")
        return ""

# Process websites after input
def handle_websites(urls):
    aggregated_text = ""

    # Scrape text from all websites
    for url in urls:
        print(f"Processing website: {url}")
        extracted_text = scrape_website_content(url)
        aggregated_text += extracted_text + "\n"

    if not aggregated_text.strip():
        print("No content was scraped from the provided URLs.")
        return

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(aggregated_text)

    # Create FAISS index
    try:
        embeddings_model = HuggingFaceEmbeddings()
        vector_store = FAISS.from_texts(chunks, embeddings_model)

        # Save the FAISS index to a pickle file
        with open(store_file, "wb") as f:
            pickle.dump(vector_store, f)

        print("Text extracted and FAISS index saved.")
    except Exception as e:
        print(f"Error while creating or saving FAISS index: {e}")

# User input for URLs
input_urls = input("Enter the website URLs (comma-separated): ").split(",")

# Run processing after URL input
if input_urls:
    handle_websites(input_urls)

# Query input
user_query = input("Ask a Question: ")

if user_query:
    if os.path.exists(store_file):
        try:
            with open(store_file, "rb") as f:
                vector_store = pickle.load(f)

            if llm_model is None:
                print("LLM model is not initialized. Please check your API key.")
            else:
                qa_chain = RetrievalQA.from_llm(llm=llm_model, retriever=vector_store.as_retriever())

                # Get response
                answer = qa_chain.run(user_query)

                # Display answer
                print("Answer:")
                print(answer)
        except Exception as e:
            print(f"Error while retrieving or querying FAISS index: {e}")
    else:
        print("FAISS index file not found. Please run the scraper first.")
