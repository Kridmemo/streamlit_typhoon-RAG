!pip install langchain langchain-community chromadb pymupdf sentence-transformers

import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import tempfile
import os
import requests

st.set_page_config(page_title="📚 RAG Chatbot - Typhoon", layout="wide")
st.title("🧪 ระบบ RAG Chatbot จากเอกสาร PDF")

# ส่วนรับ input จากผู้ใช้
uploaded_file = st.file_uploader("📤 อัปโหลดไฟล์ PDF คู่มือแนวทางการจัดการขยะติดเชื้อ", type=["pdf"])
system_prompt = st.text_area(
    "🧠 ป้อน System Prompt",
    "คุณคือผู้เชี่ยวชาญด้านการจัดการขยะติดเชื้อในสถานพยาบาล หากคำถามใดไม่เกี่ยวข้องกับการจัดการขยะติดเชื้อ ให้ตอบว่า 'คำถามนี้อยู่นอกขอบเขตของฉัน'"
)

# ฝังไฟล์เมื่อผู้ใช้ upload
if uploaded_file:
    with st.spinner("🔄 กำลังประมวลผลไฟล์..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()

        splitter = CharacterTextSplitter(separator="", chunk_size=1000, chunk_overlap=100)
        texts = splitter.split_documents(documents)

        for i, doc in enumerate(texts):
            doc.metadata["chunk_id"] = i

        filtered_docs = [doc.page_content for doc in texts if doc.metadata["chunk_id"] <= 274]

        embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device": "cpu"})
        vectordb = Chroma.from_texts(texts=filtered_docs, embedding=embedding)
        retriever = vectordb.as_retriever()

        # เก็บประวัติการสนทนา
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [{"role": "system", "content": system_prompt}]

        user_input = st.chat_input("💬 พิมพ์คำถามของคุณ")
        if user_input:
            with st.spinner("🤖 กำลังประมวลผล..."):
                docs = retriever.get_relevant_documents(user_input)
                selected_docs = docs[:3]
                context = "\n\n".join([doc.page_content for doc in selected_docs])

                st.session_state.chat_history.append({
                    "role": "user",
                    "content": f"ข้อมูลอ้างอิง:\n{context}\n\nคำถาม: {user_input}"
                })

                def ask_typhoon(chat_history):
                    headers = {
                        "Authorization": "Bearer sk-HQqPVR5RVGvTKVFcDHRJdVRtH3sQnH3VqKPHyYr5hoFsBFDj",
                        "Content-Type": "application/json"
                    }
                    data = {
                        "model": "typhoon-v2-70b-instruct",
                        "messages": chat_history,
                        "temperature": 0.5,
                        "max_tokens": 512
                    }
                    res = requests.post("https://api.opentyphoon.ai/v1/chat/completions", headers=headers, json=data)
                    return res.json()["choices"][0]["message"]["content"]

                answer = ask_typhoon(st.session_state.chat_history)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # แสดงประวัติการสนทนา
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.chat_message("user").markdown(msg["content"])
            elif msg["role"] == "assistant":
                st.chat_message("assistant").markdown(msg["content"])
