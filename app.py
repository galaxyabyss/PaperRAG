import streamlit as st
import fitz
import re
import numpy as np
import pickle
import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# -------------------------- 配置 --------------------------
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

# -------------------------- 缓存目录 --------------------------
CACHE_FILE = "cache/embeddings_cache.pkl"

# -------------------------- 初始化会话状态（关键！） --------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # 存放对话：[{"role":"user","content":"..."}, ...]

# -------------------------- 工具函数（加缓存） --------------------------
@st.cache_data
def read_pdf(file_path):
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

@st.cache_data
def split_overlap_chunks(text, max_len=700, overlap_len=150):
    sentences = re.split(r'([。！？；\n])', text)
    chunks = []
    current = ""
    for i in range(0, len(sentences) - 1, 2):
        sent = sentences[i] + sentences[i + 1]
        if len(current) + len(sent) > max_len:
            chunks.append(current.strip())
            current = sent[-overlap_len:] + sent
        else:
            current += sent
    if current.strip():
        chunks.append(current.strip())
    return chunks

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text, model):
    return model.encode(text, convert_to_numpy=True)

def load_or_create_embeddings(chunks, model):
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    embeddings = [get_embedding(chunk, model) for chunk in chunks]
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(embeddings, f)
    return embeddings

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_top_k(question_emb, chunks, embeddings, top_k=3):
    scores = [cosine_sim(question_emb, emb) for emb in embeddings]
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_k]]

def build_prompt(context, question):
    return f"""
你是数学建模竞赛专家，请根据文件内容依次回答给你的问题。
如果题目内容里给你的问题没有答案，必须直接说：“根据题目内容，无法回答这个问题”，绝对不要编造。

----------------
题目内容：
{context}
----------------

问题：{question}
回答：
"""

# -------------------------- 页面UI --------------------------
st.title("📄 论文PDF智能问答系统")

# 1. 上传PDF
uploaded_file = st.file_uploader("上传PDF文件", type="pdf")

# 2. 显示历史对话
st.subheader("💬 对话历史")
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 3. 输入问题 + 发送按钮
question = st.text_input("请输入你的问题:")

if st.button("发送问题"):
    if uploaded_file is not None and question.strip() != "":
        # 先把用户问题加入历史（立刻显示）
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.spinner("正在处理..."):
            # 保存PDF
            with open("uploads/temp_upload.pdf", "wb") as f:
                f.write(uploaded_file.read())

            # 处理PDF、分块、向量
            pdf_content = read_pdf("temp_upload.pdf")
            chunks = split_overlap_chunks(pdf_content)
            model = get_embedding_model()
            chunk_embeddings = load_or_create_embeddings(chunks, model)

            # 检索+提问
            q_emb = get_embedding(question, model)
            context = "\n".join(search_top_k(q_emb, chunks, chunk_embeddings))
            prompt = build_prompt(context, question)

            res = client.chat.completions.create(
                model="mimo-v2.5-pro",
                messages=[{"role": "user", "content": prompt}]
            )
            answer = res.choices[0].message.content

            # 把AI回答也加入历史
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # 刷新页面，新对话会追加显示
        st.rerun()

    else:
        st.warning("请先上传PDF并输入问题！")

# 清空历史按钮
if st.button("清空对话历史"):
    st.session_state.chat_history = []
    st.rerun()
