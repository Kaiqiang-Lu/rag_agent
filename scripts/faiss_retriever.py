import os
import argparse
import wikipediaapi
from opencc import OpenCC
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 路径定义
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_BASE_DIR = os.path.join(BASE_DIR, "indexes")

# 获取 Wikipedia 词条
def fetch_wikipedia_article(title, save_path=None):
    wiki = wikipediaapi.Wikipedia(
    language="zh",
    user_agent="rag_agent/1.0 (https://github.com/yourname; luca@example.com)"
)

    page = wiki.page(title)

    if not page.exists():
        raise ValueError(f"词条 '{title}' 不存在或获取失败。")

    text = page.text

    # 繁体转简体
    cc = OpenCC("t2s")
    text = cc.convert(text)
 

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)

    return text

# 文本转 Document
def create_document_from_text(text):
    from langchain_core.documents import Document
    return [Document(page_content=text)]

# 拆分文本
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "。", "！", "？", "；", "，"]
    )
    splits = splitter.split_documents(docs)
    print(f"文本被拆分为 {len(splits)} 个片段。")
    return splits

# FAISS 创建向量索引
def build_or_load_faiss(splits, embeddings, index_dir, topic_name=None):
    if topic_name is None:
        topic_name = os.path.basename(index_dir).replace("wiki_", "").replace("_index", "")
    index_name = f"wiki_{topic_name}_index"

    parent_dir = os.path.dirname(index_dir)
    os.makedirs(parent_dir, exist_ok=True)

    faiss_file = os.path.join(parent_dir, f"{index_name}.faiss")

    if os.path.exists(faiss_file):
        db = FAISS.load_local(
            folder_path=parent_dir,
            embeddings=embeddings,
            index_name=index_name,
            allow_dangerous_deserialization=True
        )
    else:
        db = FAISS.from_documents(splits, embeddings)
        db.save_local(folder_path=parent_dir, index_name=index_name)

    return db



# 查询
def query_faiss(db, query, k=5):
    results = db.similarity_search(query, k=k)
    for i, res in enumerate(results):
        print(f"—— 结果 {i+1} ——")
        print(res.page_content.strip().replace("\n", " "), "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wikipedia + FAISS 中文语义检索系统")
    parser.add_argument("--title", type=str, default="人工智能", help="Wikipedia 词条标题（默认为人工智能）")
    args = parser.parse_args()
    title = args.title.strip()

    print(f"当前词条：{title}")


    wiki_file_path = os.path.join(DATA_DIR, f"wiki_{title}.txt")
    index_dir = os.path.join(INDEX_BASE_DIR, f"wiki_{title}_index")

    if not os.path.exists(wiki_file_path):
        text = fetch_wikipedia_article(title=title, save_path=wiki_file_path)
    else:
        with open(wiki_file_path, "r", encoding="utf-8") as f:
            text = f.read()

    docs = create_document_from_text(text)
    splits = split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="moka-ai/m3e-large")

    db = build_or_load_faiss(splits, embeddings, index_dir)

    # 交互式查询
    while True:
        q = input("\n请输入查询内容（输入 'exit' 退出）：")
        if q.lower().strip() == "exit":
            break
        query_faiss(db, q, k=5)
