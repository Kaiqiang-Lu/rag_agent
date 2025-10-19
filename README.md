# RAG Agent 项目

本项目使用 Python 实现一个基于 RAG（Retrieval-Augmented Generation）框架的智能体项目。  主要包括数据抓取、文本清洗、分块与向量索引构建过程。

---

## 项目结构
rag_agent/
├── data/ # 存放抓取的维基文本数据
├── indexes/ # 存放生成的向量索引
├── scripts/ # 主要脚本文件
│      .env      #环境配置
└── README.md # 项目说明文件

---

## 环境配置

本项目基于 **Python 3.11**。  
使用 **Anaconda** 创建独立环境：

```bash
conda create -n rag_agent python=3.11
conda activate rag_agent
安装依赖：

pip install "transformers>=4.42" "huggingface_hub>=0.24"
pip install "sentence-transformers>=2.7"
pip install faiss-cpu
pip install "langchain>=0.2.16" "langchain-community>=0.2.16" "langgraph>=0.2.22" "tiktoken>=0.7"
pip install python-dotenv wikipedia duckduckgo-search

知识库构建流程

python scripts/fetch_wiki.py

输出：data/wiki_XXXX.txt

清洗与向量化

python scripts/build_vectorstore.py

输出：indexes/wiki_index/（FAISS 索引）
显示分块统计与章节分布。
