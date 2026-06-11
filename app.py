import streamlit as st
import fitz
import re
import numpy as np
import pickle
import os
import hashlib
from openai import OpenAI
from sentence_transformers import SentenceTransformer,CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

os.makedirs("cache", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

safe_model_name=EMBEDDING_MODEL_NAME.replace("/","_")

SEARCH_TOP_K = 20
RERANK_TOP_K = 5

# -------------------------- 配置 --------------------------
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)


# -------------------------- 初始化会话状态 --------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_rewrite" not in st.session_state:
    st.session_state.last_rewrite = ""

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
            current = current[-overlap_len:] + sent
        else:
            current += sent
    if current.strip():
        chunks.append(current.strip())
    return chunks





@st.cache_data
def split_chunks(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            " "
        ]
    )

    return splitter.split_text(text)





@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )
    reranker_model = CrossEncoder(
        RERANKER_MODEL_NAME
    )
    return embedding_model, reranker_model

def get_embedding(text, model):
    return model.encode(text, convert_to_numpy=True)

def load_or_create_embeddings(chunks, model,cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    embeddings = [get_embedding(chunk, model) for chunk in chunks]
    with open(cache_file, "wb") as f:
        pickle.dump(embeddings, f)
    return embeddings

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))



def search_top_k(question_emb, chunks, embeddings, top_k):
    scores = [cosine_sim(question_emb, emb) for emb in embeddings]
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_k]]



def rerank(question, chunks, reranker,top_k):
    pairs = [[question, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(
        zip(scores, chunks),
        key=lambda x:x[0],
        reverse=True
    )
    return [c for _, c in ranked[:top_k]]

#----------------------------------------------------


def rewrite_question(history, question):
    history_text = ""

    role_name = {
        "user": "用户",
        "assistant": "助手"
    }

    for msg in history[-6:]:
        history_text += (
            f"{role_name[msg['role']]}:"
            f"{msg['content']}\n"
        )

    rewrite_prompt = f"""
请根据历史对话补全当前问题。

历史对话：
{history_text}

当前问题：
{question}

要求：
如果当前问题已经完整，直接返回原问题。
如果当前问题依赖历史上下文，请补全为独立完整的问题。

只输出问题本身。
"""

    res = client.chat.completions.create(
        model="mimo-v2.5-pro",
        messages=[
            {
                "role":"user",
                "content":rewrite_prompt
            }
        ]
    )

    return res.choices[0].message.content.strip()



#----------------------------------------------------




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



# -------------------------- DE bug --------------------------

DEBUG = st.sidebar.checkbox(
    "Debug Mode",
    value=True
)

# -------------------------- 页面UI --------------------------  caution:刷新后记录，不是发过去就记录
st.title("📄 论文PDF智能问答系统")

if DEBUG and st.session_state.last_rewrite:
    st.info(
        f"Rewrite后问题：{st.session_state.last_rewrite}"
    )

#  上传PDF
uploaded_file = st.file_uploader("上传PDF文件", type="pdf")

#  显示历史对话
st.subheader("💬 对话历史")
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

#  输入问题 + 发送按钮
question = st.text_input("请输入你的问题:")


if st.button("发送问题"):
    if uploaded_file is not None and question.strip() != "":

        #Query rewrite and check
        rewritten_question = rewrite_question(
            st.session_state.chat_history,
            question
        )

        st.session_state.last_rewrite = rewritten_question



        # 先把用户问题加入历史（立刻显示）


        with st.spinner("正在处理..."):
            # 保存PDF
            pdf_bytes = uploaded_file.read()
            file_hash = hashlib.md5(pdf_bytes).hexdigest()
            pdf_path=os.path.join("uploads",f"{file_hash}.pdf")
            cache_file = os.path.join("cache", f"{file_hash}_{safe_model_name}.pkl")
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)

            # 处理PDF、分块、向量
            pdf_content = read_pdf(pdf_path)




            # chunks = split_overlap_chunks(pdf_content)
            chunks = split_chunks(pdf_content)




            embedding_model, reranker_model = load_models()
            chunk_embeddings = load_or_create_embeddings(chunks, embedding_model,cache_file)


            # embedding
            q_emb = get_embedding( rewritten_question,  embedding_model)

            # context = "\n".join(search_top_k(q_emb, chunks, chunk_embeddings))
            candidate_chunks = search_top_k(
                q_emb,
                chunks,
                chunk_embeddings,
                SEARCH_TOP_K
            )

            if DEBUG:
                st.write("### Recall Top3")

                for i, chunk in enumerate(candidate_chunks[:3]):
                    st.write(f"Chunk {i + 1}")
                    st.write(chunk[:200])

# ----------------------------------------------------

            reranked_chunks = rerank(
                rewritten_question,
                candidate_chunks,
                reranker_model,
                RERANK_TOP_K
            )

            if DEBUG:
                st.write("### Rerank Top5")

                for i, chunk in enumerate(reranked_chunks):
                    st.write(f"Rank {i + 1}")
                    st.write(chunk[:200])

# ----------------------------------------------------
            context = "\n".join(reranked_chunks)

            if DEBUG:
                st.write("### Final Context")
                st.write(context[:3000])


            prompt = build_prompt(context, rewritten_question)
            if DEBUG:
                st.write("### Prompt")
                st.code(prompt)


            res = client.chat.completions.create(
                model="mimo-v2.5-pro",
                messages=[{"role": "user", "content": prompt}]
            )
            answer = res.choices[0].message.content

            # AI与用户历史

            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # 刷新页面，新对话会追加显示
        st.rerun()

    else:
        st.warning("请先上传PDF并输入问题！")

# 清空历史按钮
if st.button("清空对话历史"):
    st.session_state.chat_history = []
    st.session_state.last_rewrite = ""
    st.rerun()