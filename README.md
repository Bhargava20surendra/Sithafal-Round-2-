Task-1
import pdfplumber
from google.colab import files
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Upload and retrieve the PDF file
uploaded = files.upload()
pdf_path = list(uploaded.keys())[0]

# Extract text and tables from the PDF
def extract_all_pdf_content(pdf_path):
    all_text = []
    all_tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num in range(len(pdf.pages)):
            page_text = pdf.pages[page_num].extract_text()
            if page_text:
                all_text.append(page_text)

            page_table = pdf.pages[page_num].extract_table()
            if page_table:
                all_tables.append(page_table)

    return all_text, all_tables

all_text, all_tables = extract_all_pdf_content(pdf_path)

if all_text or all_tables:
    chunks = all_text + [str(row) for table in all_tables for row in table]
else:
    print("No text or tables available to process.")
    chunks = []

# Encode chunks and store in FAISS index
if chunks:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    print("Embeddings successfully stored in FAISS index.")
else:
    print("No chunks to process.")

# Function to search relevant chunks
def search_relevant_chunks(query):
    if chunks:
        query_embedding = model.encode([query])
        distances, indices = index.search(np.array(query_embedding), k=3)
        relevant_chunks = [chunks[i] for i in indices[0]]
        return relevant_chunks
    else:
        return []

# Load local LLM model
def load_local_model():
    model_name = "google/flan-t5-base"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("Local LLM model loaded successfully!")
        return tokenizer, llm_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

tokenizer, llm_model = load_local_model()

# Generate answer from retrieved chunks
def generate_answer_from_retrieved_chunks(retrieved_chunks, query):
    context = "\n".join(retrieved_chunks)
    prompt = f"Based on the following retrieved information:\n{context}\n\nAnswer the following question: {query}"

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    output = llm_model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Query input and process
def main():
    user_query = input("Please enter your query: ")
    retrieved_chunks = search_relevant_chunks(user_query)

    if retrieved_chunks:
        print("Relevant Chunks Found:\n", "\n".join(retrieved_chunks))
        if llm_model:
            final_response = generate_answer_from_retrieved_chunks(retrieved_chunks, user_query)
            print("\nGenerated Answer:\n", final_response)
        else:
            print("\nCannot generate answer due to missing LLM model.")
    else:
        print("No relevant chunks found for your query.")

if __name__ == "__main__":
    main()
