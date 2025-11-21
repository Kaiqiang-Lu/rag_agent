RAG 文档问答智能体

本项目基于 LangChain + FAISS + Gemini + CrossEncoder 实现了一个多轮问答智能体。
系统可从用户上传的文本文件中构建知识库，支持上下文记忆、多轮对话与结果重排。

● 项目结构
```bash
rag_agent/
├── data/                 # 存放原始数据文件
├── .env                  # 环境变量文件（需包含 GOOGLE_API_KEY）
├── rag_chat_agent.py     # 主程序（Streamlit 多轮问答界面）
└── README.md             # 项目说明文件
```
● 环境配置
Python 版本

建议使用 Python 3.11

● 创建虚拟环境
```bash
conda create -n rag_agent python=3.11
conda activate rag_agent
```
● 安装依赖

本项目核心依赖包括 LangChain、FAISS、Gemini SDK、SentenceTransformers 等：
```bash
pip install "transformers>=4.42" "huggingface_hub>=0.24"
pip install "sentence-transformers>=2.7"
pip install faiss-cpu
pip install "langchain>=0.2.16" "langchain-community>=0.2.16" "langgraph>=0.2.22" "tiktoken>=0.7"
pip install streamlit python-dotenv
pip install google-generativeai
```
● 系统功能说明
1. 文档上传与知识库构建

支持多文件 txt 上传，系统自动进行分块。

使用 HuggingFaceEmbeddings（moka-ai/m3e-large）提取向量。

基于 FAISS 构建向量索引，实现语义检索。

2. 检索与重排

检索到的候选文本通过 CrossEncoder('mmarco-mMiniLMv2-L12-H384-v1') 进行语义相关性重排。

选取排名靠前的内容作为上下文输入 LLM。

3. 生成回答

使用 Google Gemini 2.5 Flash 模型生成自然语言回答。

若文档中无答案，将自动输出“抱歉，我没有在上传文档中找到相关信息。”

4. 多轮对话与记忆机制

使用 ConversationBufferMemory 存储上下文，实现多轮问答。

模型结合历史聊天记录与当前检索结果进行响应。

5. 前端交互界面

基于 Streamlit 构建，支持：

文件上传区（支持多文件）

聊天消息区

“清除聊天记录”按钮

输入框固定底部

● 运行方式

在项目目录下运行：
```bash
streamlit run rag_chat_agent.py
```
启动后打开浏览器访问（默认）：
```bash
http://localhost:8501
```

上传 txt 文档后，即可进行智能问答。

● 系统工作流程

用户上传文本文件（txt 格式）。

系统将文本分块并生成向量表示。

使用 FAISS 建立向量索引库。

根据用户问题进行语义检索。

使用 CrossEncoder 对检索结果进行重排。

将最相关内容与上下文历史结合后输入 Gemini 模型生成回答。

回答结果显示在前端对话界面中。

● 注意事项

.env 文件需包含：

GOOGLE_API_KEY=你的Gemini密钥


若需切换模型，可修改：

model: str = "gemini-2.5-flash"
