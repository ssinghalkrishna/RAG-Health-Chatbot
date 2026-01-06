
# Import standard libraries for file handling and text processing
import os, pathlib, textwrap, glob

# Load documents from various sources (URLs, text files, PDFs)
from langchain_community.document_loaders import UnstructuredURLLoader, TextLoader, PyPDFLoader

# Split long texts into smaller, manageable chunks for embedding
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector store to store and retrieve embeddings efficiently using FAISS
from langchain.vectorstores import FAISS

# Generate text embeddings using OpenAI or Hugging Face models
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, SentenceTransformerEmbeddings

# Use local LLMs (e.g., via Ollama) for response generation
from langchain.llms import Ollama

# Build a retrieval chain that combines a retriever, a prompt, and an LLM
from langchain.chains import ConversationalRetrievalChain

# Create prompts for the RAG system
from langchain.prompts import PromptTemplate

print("✅ Libraries imported! You're good to go!")

pdf_paths = glob.glob("data/Everstorm_*.pdf")
raw_docs = []

for path in pdf_paths:
    loader = PyPDFLoader(path)
    docs = loader.load()
    raw_docs.extend(docs)

print(f"Loaded {len(raw_docs)} PDF pages from {len(pdf_paths)} files.")

URLS = [
    # --- BigCommerce – shipping & refunds ---
    "https://developer.bigcommerce.com/docs/store-operations/shipping",
    "https://developer.bigcommerce.com/docs/store-operations/orders/refunds",
    # --- Stripe – disputes & chargebacks ---
     "https://docs.stripe.com/disputes",  
    # --- WooCommerce – REST API reference ---
     "https://woocommerce.github.io/woocommerce-rest-api-docs/v3.html",
]

try:
    ########################
    #### Your code here (~2-3 lines of code) ####
    ########################
    loader = UnstructuredURLLoader(URLS)
    raw_docs1 = loader.load()
    print(f"Fetched {len(raw_docs1)} documents from the web.")
except Exception as e:
    print("⚠️  Web fetch failed, using offline copies:", e)
    raw_docs1 = []
    ########################
    #### Your code here ####
    ########################  
      # Example: load PDFs as fallback
    pdf_paths = glob.glob("data/Everstorm_*.pdf")
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        raw_docs1.extend(docs)
    print(f"Loaded {len(raw_docs1)} offline documents.")

chunks = []
splitter = RecursiveCharacterTextSplitter( chunk_size=300, 
                                           # approx 300 tokens 
                                           chunk_overlap=30,
                                           # with 30-token overlap 
                                           length_function=len, # defaults to character count; 
                                           #for tokens, use token-based tokenizer if available 
                                           separators=["\n\n", "\n", " ", ""] )
chunks = splitter.split_documents(raw_docs + raw_docs1)
print(f"✅ {len(chunks)} chunks ready for embedding")

# Expected steps:
    # 1. Build the FAISS index from the list of document chunks and their embeddings.
    # 2. Create a retriever object with a suitable k value (e.g., 8).
    # 3. Save the vector store locally (e.g., under "faiss_index").
    # 4. Print a short confirmation showing how many embeddings were stored.

# Load embedding model (assuming already done earlier)
embedding_model = SentenceTransformerEmbeddings(model_name="thenlper/gte-small")

# 1. Build FAISS index from chunks
vectordb = FAISS.from_documents(chunks, embedding_model)

# 2. Create a retriever with k=8
retriever = vectordb.as_retriever(search_kwargs={"k": 8})

# 3. Save the FAISS index locally
vectordb.save_local("faiss_index")

# 4. Print how many embeddings were stored
print("✅ Vector store with", vectordb.index.ntotal, "embeddings")

SYSTEM_TEMPLATE = """ 
You are a **Customer Support Chatbot**. Use only the information in CONTEXT to answer.
If the answer is not in CONTEXT, respond with “I'm not sure from the docs.”

Rules:
1) Use ONLY the provided <context> to answer.
2) If the answer is not in the context, say: "I don't know based on the retrieved documents."
3) Be concise and accurate. Prefer quoting key phrases from the context.

CONTEXT:
{context}

USER:
{question}
"""

# Expected steps:
    # 1. Create a PromptTemplate that uses the SYSTEM_TEMPLATE you defined earlier, with input variables for "context" and "question".
    # 2. Initialize your LLM using Ollama with the gemma3:1b model and a low temperature (e.g., 0.1) for reliable, grounded responses.
    # 3. Build a ConversationalRetrievalChain by combining the LLM, the retriever, and your custom prompt and name it "chain".

# 1. Initialize the model with low temperature for more factual outputs
llm = Ollama(model="gemma3:1b", temperature=0.1)

prompt = PromptTemplate(
    template=SYSTEM_TEMPLATE,
    input_variables=["context", "question"]
)

# 3. Build the ConversationalRetrievalChain
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    combine_docs_chain_kwargs={"prompt": prompt}
)

# Response function to be used in Streamlit
def get_rag_response(query: str, chat_history) -> str:
    
    result = chain.invoke({
        "question": query,
        "chat_history": chat_history
    })
    
    return result['answer']
