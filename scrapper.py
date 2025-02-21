import boto3
import json
import faiss
import numpy as np
import fitz  # PyMuPDF for PDF processing
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import os
import sys
import asyncio

# =========================
# 📌 Step 1: Initialize Bedrock Client
# =========================
def get_bedrock_client():
    os.environ["AWS_ACCESS_KEY_ID"] = "KEY"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "KEY"
    os.environ["AWS_REGION"] = "ap-south-1"
    return boto3.client("bedrock-runtime", region_name="ap-south-1")

# =========================
# 📌 Step 2: Extract Text from PDF
# =========================
def extract_text_from_pdf(pdf_path):
    """Reads text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip() if text else "No readable text found in PDF."

# =========================
# 📌 Step 3: Generate Embeddings using Amazon Bedrock
# =========================
def generate_embedding(text, client, input_type="search_document"):
    max_length = 2048  
    truncated_text = text[:max_length]

    payload = {
        "texts": [truncated_text],
        "input_type": input_type
    }
    
    response = client.invoke_model(
        modelId="cohere.embed-multilingual-v3",
        body=json.dumps(payload),
        contentType="application/json"
    )
    
    response_body = json.loads(response['body'].read())
    return np.array(response_body["embeddings"][0]).astype("float32") if "embeddings" in response_body else np.zeros(1024, dtype="float32")

# =========================
# 📌 Step 4: Store Data in FAISS (Shopify + PDF)
# =========================
def store_in_faiss(texts, client):
    text_vectors = [generate_embedding(text, client) for text in texts]
    text_vectors = np.array(text_vectors).astype("float32")

    index = faiss.IndexFlatL2(len(text_vectors[0]))
    index.add(text_vectors)

    return index, texts

# =========================
# 📌 Step 5: Scrape Shopify Page
# =========================
def scrape_shopify_page(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")  
    driver_path = "/usr/local/bin/chromedriver"
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        driver.get(url)
        time.sleep(3)  
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        return text.strip() if text else "No content found on the page."
    finally:
        driver.quit()

# =========================
# 📌 Step 6: Retrieve Data from FAISS
# =========================
def search_faiss(query, index, text_corpus, client, top_k=3):
    query_vector = generate_embedding(query, client, input_type="search_query").reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)

    if len(indices[0]) == 0:
        return ["No relevant data found."]
    
    return [text_corpus[i] for i in indices[0]]

# =========================
# 📌 Step 7: Generate AI Response from Amazon Bedrock
# =========================
def generate_answer(query, context, client):
    prompt = f"Answer the question based on the following information:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    payload = {"prompt": prompt, "max_tokens": 300}

    response = client.invoke_model(
        modelId="mistral.mistral-7b-instruct-v0:2",
        body=json.dumps(payload),
        contentType="application/json"
    )

    response_body = json.loads(response['body'].read())
    return response_body["outputs"][0]["text"] if "outputs" in response_body else "⚠️ AI did not return a valid response."

# =========================
# 📌 Step 8: Print AI Response with Typing Effect
# =========================
def print_ai_response(text, typing_speed=0.02):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(typing_speed)
    print()

# =========================
# 📌 Step 9: Main Execution (Interactive CLI)
# =========================
if __name__ == "__main__":
    shopify_url = "https://www.siz.ae"
    pdf_path = "/Users/apple/Work/CustBot/docs/LENDERS_TERMS_AND_CONDITIONS.pdf"  # 🔹 Change this to your actual PDF path

    print("🔍 Scraping Shopify Page...")
    scraped_text = scrape_shopify_page(shopify_url)

    print("📄 Reading PDF content...")
    pdf_text = extract_text_from_pdf(pdf_path)

    print("🔄 Storing Data in FAISS...")
    bedrock_client = get_bedrock_client()
    faiss_index, text_corpus = store_in_faiss([scraped_text, pdf_text], bedrock_client)

    print("✅ Data stored successfully!")

    # **Interactive Chat**
    print("\n💬 Ask me anything (type 'exit' to quit):")

    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            print("👋 Exiting... Goodbye!")
            break
        
        retrieved_texts = search_faiss(query, faiss_index, text_corpus, bedrock_client)
        combined_context = " ".join(retrieved_texts)

        ai_response = generate_answer(query, combined_context, bedrock_client)
        print("\n🤖 AI Answer: ", end="")
        print_ai_response(ai_response)
