# RAG Agent 项目

本项目使用 Python 实现一个基于 RAG（Retrieval-Augmented Generation）框架的智能体项目。  

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
pip install python-dotenv wikipediaapi duckduckgo-search

使用方法与结果展示

运行指令
```bash
python scripts/faiss_retriever.py
```
或指定其他词条：
```bash
python scripts/faiss_retriever.py --title 艾伦·图灵
```
交互式查询
程序运行后，输入查询语句：
```bash
请输入查询内容（输入 'exit' 退出）：
>>> 人工智能的发展阶段有哪些？
```

运行结果：

文本拆分数量

向量索引创建过程

查询输出的匹配结果
