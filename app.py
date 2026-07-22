import streamlit as st
import torch
import os
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq

# Page Config
st.set_page_config(page_title="Clinical Protocol RAG Assistant", layout="wide")
st.title("🩺 Clinical Protocol & Guidelines RAG Assistant")
st.markdown("Powered by Auto-Ingestion, Hybrid MMR/BM25 Search, and Flashrank Cross-Encoder Reranking.")

@st.cache_resource
def load_rag_backend():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    emb_fn = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

    db_path = "./chroma_db"
    
    vector_store = Chroma(
        collection_name="Medical_RAG_Collection",
        embedding_function=emb_fn,
        persist_directory=db_path
    )

    chroma_data = vector_store.get()

    # Auto-ingestion fallback if database is empty
    if not chroma_data or not chroma_data.get("ids"):
        st.info("🔄 Checking for source documents...")
        available_pdfs = [f for f in os.listdir(".") if f.endswith(".pdf")]
        
        if available_pdfs:
            source_pdf = available_pdfs[0]
            #st.warning(f"⚠️ Found '{source_pdf}'. Ingesting into Chroma DB...")
            loader = PyPDFLoader(source_pdf)
            raw_docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(raw_docs)
            
            vector_store = Chroma.from_documents(
                documents=docs,
                embedding=emb_fn,
                collection_name="Medical_RAG_Collection",
                persist_directory=db_path
            )
            chroma_data = vector_store.get()
        else:
            from langchain_core.documents import Document
            fallback_doc = Document(
                page_content="Clinical Guidelines Fallback: Please upload a medical guideline PDF to your Colab environment files directory.",
                metadata={"Section": "General Notice"}
            )
            vector_store = Chroma.from_documents(
                documents=[fallback_doc],
                embedding=emb_fn,
                collection_name="Medical_RAG_Collection",
                persist_directory=db_path
            )
            chroma_data = vector_store.get()

    vector_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 60, "lambda_mult": 0.5}
    )

    all_child_chunks = vector_store.similarity_search("", k=min(len(chroma_data["ids"]), 10000))
    if not all_child_chunks:
        raise ValueError("Could not fetch child chunks from Chroma vector store for BM25 initialization.")

    keyword_retriever = BM25Retriever.from_documents(all_child_chunks)
    keyword_retriever.k = 8

    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, keyword_retriever],
        weights=[0.5, 0.5]
    )

    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=6)
    final_pipeline_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=hybrid_retriever
    )

    return final_pipeline_retriever

try:
    final_pipeline_retriever = load_rag_backend()
except Exception as e:
    st.error(f"Error loading pipeline back-end: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a clinical query (e.g., Metformin side effects & practical tips):"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing hybrid index matrices, reranking documents, and querying Llama-3..."):
            try:
                retrieved_context = final_pipeline_retriever.invoke(query)
                context_str = "\n\n".join([f"Source Metadata: {doc.metadata}\nContent: {doc.page_content}" for doc in retrieved_context])

                system_prompt = f"""
                    You are an super specialist clinical support AI Assistant.You do not have knowledge in fields OTHER THAN 
                    CLINICAL field, YOU have no knoweldge in general medicine or general health conditions/deseases. 
                    if the query is not clinical or is abot general health conditions say SORRY.

                    GUIDELINES FOR ANSWERING:
                    1. **You are a Medical Professional, you can reply to uestions related to mdicial field and is related to the conext {context_str}.
                    2. **Clinical & Medical Queries:** Answer comprehensively using *only* the provided context chunks. Combine information from multiple sections across the chunks if needed. Do not extrapolate outside medical knowledge. If clinical info is genuinely absent, state: 'Insufficient clinical guidelines data provided.'
                    3. **General or Personal Queries (e.g., names like Manju, casual chat, greetings):** If the user asks about a person, general knowledge, or casual conversation that is not a medical guideline query, respond naturally, politely, and helpfully using common sense, while letting them know whether it appears in the documents or not. 
                    4. **Safety First:** If a user expresses any form of distress, self-harm, or emotional crisis, respond with SORRY.
                    5. **DO NOT ANSWER IF IT IS NOT CLINICAL.
                    6. ** DO NOT ANSWER only the query has matching conext chunks.
                    7. ** if the chunk is "Clinical Guidelines Fallback: Please upload a medical guideline PDF to your Colab environment files directory.",say sorryyyy...


                CONTEXT:
                {context_str}

                Clinical Query: {query}
                Answer:
                """                

                groq_api_key = os.environ.get("GROQ_API_KEY")
                groq_client = Groq(api_key=groq_api_key)

                completion = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
                    ],
                    temperature=0.0
                )

                answer = completion.choices[0].message.content
                st.markdown(answer)

                with st.expander("🔍 View Retrieved Multi-Part Evidence & Metadata Chunks"):
                    for idx, doc in enumerate(retrieved_context, 1):
                        st.markdown(f"**Chunk {idx}** | Section: `{doc.metadata.get('Section', 'Unknown')}`")
                        st.text(doc.page_content.strip())
                        st.markdown("---")

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Error handling clinical inference: {e}")
import os
import subprocess
from pyngrok import ngrok
from google.colab import userdata

# Load Groq API key securely from Colab secrets (Name it 'groq_api')

os.environ["GROQ_API_KEY"] = userdata.get('groq_api')

# Configure Ngrok
ngrok.kill()
ngrok.set_auth_token("2gbEhmyrGpSn4VIcX0ebELtA0Vs_6rXpaAcuxH3AtqKxHiKBS")

# Launch Streamlit via Subprocess and connect Ngrok
subprocess.Popen(["streamlit", "run", "app.py", "--server.port=8501"])
public_url = ngrok.connect(8501)
print(f"🚀 Streamlit App is live at: {public_url}")
