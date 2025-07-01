import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
import os
import threading
from langchain.vectorstores import FAISS
import base64

# กำหนดพื้นหลังสีชมพูและรูปภาพโลโก้
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{base64.b64encode(open('โลโก้โรงพยาบาลพรหมคีรี.png', 'rb').read()).decode()}");
        background-size: 100px;
        background-repeat: no-repeat;
        background-position: top left;
        
    }}
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_vectorstore():
    with st.status("📦 กำลังโหลดเวกเตอร์ที่บันทึกไว้...", expanded=True):
        embedding = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"}
        )
        vectordb = FAISS.load_local("medwaste_vectorstore", embedding, allow_dangerous_deserialization=True)
        return vectordb.as_retriever()

# Typhoon API
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

# ส่วนติดต่อผู้ใช้
st.set_page_config(page_title="🗑 RAG Chatbot - Medwaste", layout="wide")
st.title("🧪 ระบบ RAG Chatbot แนวทางกำจัดการขยะในโรงพยาบาลพรหมคีรี")

# ใส่ system prompt
system_prompt = "คุณคือผู้เชี่ยวชาญด้านการจัดการขยะติดเชื้อในสถานพยาบาล หากคำถามใดไม่เกี่ยวข้องกับการจัดการขยะ ให้ตอบว่า 'คำถามนี้อยู่นอกขอบเขตของฉัน'"
retriever = load_vectorstore()
# เก็บประวัติการสนทนา
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": system_prompt}]

user_input = st.text_input("🔍 พิมพ์คำถามของคุณ:")

if user_input:
    # RAG ขั้น context
    docs = retriever.get_relevant_documents(user_input)
    selected_docs = docs[:3]
    context = "\n\n".join([doc.page_content for doc in selected_docs])

    st.markdown("### 📄 Context ที่ใช้ในการตอบจาก คู่มือแนวทางการกำจัดขยะ:")
    for i, doc in enumerate(selected_docs):
        st.markdown(f"**Context {i+1}**: {doc.page_content.strip()}")

    prompt = f"ข้อมูลอ้างอิง:\n{context}\n\nคำถาม: {user_input}"
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    response = ask_typhoon(st.session_state.chat_history)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    st.markdown("### 🤖 คำตอบจากผู้ช่วย:")
    st.write(response)


# Add footer
st.markdown("""
    ---
    <div style='text-align: center;'>
        <p>ด้วยความปรารถนาดี</p>
        <p>งานจัดการสิ่งแวดล้อม โรงพยาบาลพรหมคีรี</p>
    </div>
    """, unsafe_allow_html=True)
