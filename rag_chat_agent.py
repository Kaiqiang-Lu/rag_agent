import os
import tempfile
import streamlit as st
from typing import List, Optional
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.llms.base import LLM
from google import genai
from sentence_transformers import CrossEncoder


# 页面配置
st.set_page_config(page_title="RAG 文档问答智能体", layout="wide")
st.title("📄 文档问答小助手")

# 文件上传
uploaded_files = st.sidebar.file_uploader(
    label="📂 上传 txt 文件（支持多选）", type=["txt"], accept_multiple_files=True
)
if not uploaded_files:
    st.info("请先上传 txt 文档。")
    st.stop()


# 构建 FAISS 检索器
@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()

    for file in uploaded_files:
        temp_path = os.path.join(temp_dir.name, file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getvalue())
        loader = TextLoader(temp_path, encoding="utf-8")
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="moka-ai/m3e-large")
    vectordb = FAISS.from_documents(splits, embeddings)
    return vectordb.as_retriever(search_kwargs={"k": 8}) 


retriever = configure_retriever(uploaded_files)


# Gemini 模型封装
load_dotenv()
google_client = genai.Client()


class GeminiLLM(LLM):
    model: str = "gemini-2.5-flash"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = google_client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text.strip()

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model}

    @property
    def _llm_type(self) -> str:
        return "google_gemini"


llm = GeminiLLM()


# 重排模型 
@st.cache_resource()
def load_reranker():
    model_name = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    return CrossEncoder(model_name)

reranker = load_reranker()


def rerank_docs(query: str, retrieved_docs: List[str], top_k: int = 3) -> List[str]:
    pairs = [(query, doc) for doc in retrieved_docs]
    scores = reranker.predict(pairs)
    scored = list(zip(retrieved_docs, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:top_k]]


# 聊天记忆
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# Prompt 模板
qa_prompt = PromptTemplate.from_template("{query}")


# 构建 LLM 链
qa_chain = LLMChain(llm=llm, prompt=qa_prompt)  



# 页面交互逻辑
if "messages" not in st.session_state or st.sidebar.button("🧹 清除聊天记录"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "您好，我是文档问答智能小助手。"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
st.markdown("""
    <style>
        /* 仅当 Streamlit 产生浅灰复制时，隐藏其重复元素 */
        .stChatFloatingInput + div .stChatMessage:last-child {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)
user_query = st.chat_input(placeholder="请输入您的问题...")
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("正在检索文档并生成回答..."):
            try:
                # 检索
                retrieved_docs = retriever.get_relevant_documents(user_query)
                retrieved_texts = [doc.page_content for doc in retrieved_docs]

                # 重排
                reranked_texts = rerank_docs(user_query, retrieved_texts, top_k=3)

                # 生成
                context = "\n\n".join(reranked_texts)
                # 从 memory 取出历史聊天记录
                past_msgs = memory.load_memory_variables({}).get("chat_history", [])
                history_text = ""
                if past_msgs:
                    for msg in past_msgs:
                        role = "用户" if msg.type == "human" else "助手"
                        history_text += f"{role}: {msg.content}\n"

                # 构造单一的 query
                composed_query = f"""
                你是一位中文知识助手，请根据以下文档信息回答最后的问题，要求自然、准确、逻辑清晰。

                【文档内容】
                {context}

                【历史对话】
                {history_text}

                【当前问题】
                {user_query}

                若无明确答案，请回答：“抱歉，我没有在上传文档中找到相关信息。”。
                """

                response = qa_chain.invoke({"query": composed_query})
                answer = response["text"]

                # 手动写入memory
                memory.save_context({"input": user_query}, {"output": answer})

            except Exception as e:
                answer = f"❌ 调用出错：{e}"

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.write(answer)
