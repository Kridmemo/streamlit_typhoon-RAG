!pip install streamlit langchain langchain-community chromadb pymupdf sentence-transformers pyngrok --quiet

%%writefile app.py
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import requests

# โหลดและฝังเอกสาร (run แค่รอบแรก)
@st.cache_resource
def load_vectorstore():
    with st.status("กำลังโหลดเวกเตอร์จาก PDF...", expanded=True) as status:
        progress = st.progress(0, text="📄 Loading PDF...")
        
        pdf_path = "/content/คู่มือแนวทางการปฏิบัติงาน medwaste (1).pdf"
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        progress.progress(20, text="✂️ ตัดข้อความ...")

        splitter = CharacterTextSplitter(separator="", chunk_size=1000, chunk_overlap=100)
        texts = splitter.split_documents(documents)
        for i, doc in enumerate(texts):
            doc.metadata["chunk_id"] = i
        documents = [doc.page_content for doc in texts if doc.metadata["chunk_id"] <= 274]
        progress.progress(50, text="🔍 กำลังฝังข้อมูล (embedding)...")

        embedding = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"}
        )
        progress.progress(70, text="💾 สร้างเวกเตอร์สโตร์...")

        vectordb = Chroma.from_texts(
            texts=documents,
            embedding=embedding,
            persist_directory="./chroma_db"
        )
        vectordb.persist()

        progress.progress(100, text="✅ เสร็จสิ้น")
        status.update(label="โหลดเวกเตอร์เสร็จแล้ว ✅", state="complete")
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
st.set_page_config(page_title="📚 RAG Chatbot - Medwaste", layout="wide")
st.title("🧪 ระบบ RAG Chatbot สำหรับขยะติดเชื้อ")

# ใส่ system prompt
system_prompt = "คุณคือผู้เชี่ยวชาญด้านการจัดการขยะติดเชื้อในสถานพยาบาล หากคำถามใดไม่เกี่ยวข้องกับการจัดการขยะติดเชื้อ ให้ตอบว่า 'คำถามนี้อยู่นอกขอบเขตของฉัน'"
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

    st.markdown("### 📄 Context ที่ใช้ในการตอบ:")
    for i, doc in enumerate(selected_docs):
        st.markdown(f"**Context {i+1}**: {doc.page_content.strip()}")

    prompt = f"ข้อมูลอ้างอิง:\n{context}\n\nคำถาม: {user_input}"
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    response = ask_typhoon(st.session_state.chat_history)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    st.markdown("### 🤖 คำตอบจากผู้ช่วย:")
    st.write(response)

# ปิด tunnel ทั้งหมด
ngrok.kill()
!pkill -f streamlit
from pyngrok import ngrok
import threading
import os

# ตั้งค่า Auth Token ของ ngrok
ngrok.set_auth_token("2ky4O7VCdeEAwsVQgws6wB6YjD5_2jjvwyMhCvfMFZK9og9wV")

# เปิดพอร์ต 8501 ผ่าน ngrok
public_url = ngrok.connect(8501)
print(f"🌐 เปิดให้เข้าจากภายนอกที่: {public_url}")

# รัน Streamlit ใน background
def run_app():
    os.system("streamlit run app.py")

thread = threading.Thread(target=run_app)
thread.start()

