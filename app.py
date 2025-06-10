import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import tempfile
import os
import requests

# ---- PAGE CONFIG ----
st.set_page_config(page_title="📚 RAG Chatbot - Typhoon", layout="wide")
st.title("🧪 ระบบ RAG Chatbot สำหรับไฟล์ PDF")

# ---- UPLOAD PDF ----
uploaded_file = st.file_uploader("📤 อัปโหลดไฟล์ PDF คู่มือแนวทางการจัดการขยะติดเชื้อ", type=["pdf"])

# ---- SYSTEM PROMPT ----
default_prompt = "คุณคือผู้เชี่ยวชาญด้านการจัดการขยะติดเชื้อในสถานพยาบาล หากคำถามใดไม่เกี่ยวข้องกับการจัดการขยะติดเชื้อ ให้ตอบว่า 'คำถามนี้อยู่นอกขอบเขตของฉัน'"
system_prompt = st.text_area("🧠 System Prompt", value=default_prompt, height=100)

if uploaded_file:
    with st.spinner("🔍 กำลังประมวลผล PDF และสร้างเวกเตอร์..."):
        # บันทึกไฟล์ชั่วคราว
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        # Load & Split
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()

        splitter = CharacterTextSplitter(separator="", chunk_size=1000, chunk_overlap=100)
        texts = splitter.split_documents(documents)

        for i, doc in enumerate(texts):
            doc.metadata["chunk_id"] = i

        filtered_docs = [doc.page_content for doc in texts if doc.metadata["chunk_id"] <= 274]

        # ฝังเวกเตอร์
        embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device": "cpu"})
        vectordb = Chroma.from_texts(texts=filtered_docs, embedding=embedding)
        retriever = vectordb.as_retriever()

        st.success("✅ โหลดและฝัง PDF สำเร็จแล้ว! เริ่มสนทนาได้เลยด้านล่าง")

        # --- Session for chat ---
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [{"role": "system", "content": system_prompt}]

        # --- Input Q&A ---
        user_input = st.chat_input("💬 พิมพ์คำถามของคุณที่นี่...")
        if user_input:
            with st.spinner("🤖 กำลังค้นข้อมูลและตอบกลับ..."):
                docs = retriever.get_relevant_documents(user_input)
                context = "\n\n".join([doc.page_content for doc in docs[:3]])

                st.session_state.chat_history.append({
                    "role": "user",
                    "content": f"ข้อมูลอ้างอิง:\n{context}\n\nคำถาม: {user_input}"
                })

                def ask_typhoon(chat_history):
                    headers = {
                        "Authorization": "Bearer sk-HQqPVR5RVGvTKVFcDHRJdVRtH3sQnH3VqKPHyYr5hoFsBFDj",  # <-- เปลี่ยน token ตามจริง
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

                reply = ask_typhoon(st.session_state.chat_history)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})

        # --- Display Chat History ---
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.chat_message("user").markdown(message["content"])
            elif message["role"] == "assistant":
                st.chat_message("assistant").markdown(message["content"])
