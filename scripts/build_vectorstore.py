# scripts/build_vectorstore.py
from pathlib import Path
import re
from typing import List, Tuple
from collections import Counter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = Path("data")
INDEX_DIR = Path("indexes/wiki_index")  

BAD_TAIL_SECTIONS = (
    "參考文獻|參考資料|參考|引用|來源|来源|注解|擴展閱讀|扩展阅读|外部連結|外部链接|參看|参看"
)
LIST_PREFIX = r"(?m)^\s*[-*•\d]+\.\s*"

#清理文本
def clean_text(s: str) -> str:
    s = s.replace("\r", "\n")

    s = re.sub(rf"==+\s*({BAD_TAIL_SECTIONS})\s*==+.*", "", s, flags=re.S)
    s = re.sub(r"\[\d+\]", "", s)
    s = re.sub(LIST_PREFIX, "", s)
    s = s.replace("\u3000", " ")  
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

#根据wiki词条进行章节划分
SECTION_RE = re.compile(r"(?m)^==+\s*(.+?)\s*==+\s*$")

def split_sections(full_text: str) -> List[Tuple[str, str]]:
    parts: List[Tuple[str, str]] = []
    matches = list(SECTION_RE.finditer(full_text))
    if not matches:
        return [("导言", full_text.strip())]

    head = full_text[: matches[0].start()].strip()
    if head:
        parts.append(("导言", head))

    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        title = m.group(1).strip()
        body = full_text[start:end].strip()
        if body:
            parts.append((title, body))
    return parts

def main():
    raw_all = []
    for f in DATA_DIR.glob("*.txt"):
        raw = f.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_text(raw)
        if cleaned:
            raw_all.append((f.name, cleaned))

    if not raw_all:
        raise RuntimeError("data/ 下没有可用文本，请先执行 fetch_wiki.py。")

    texts, metas = [], []
    for fname, text in raw_all:
        for section, body in split_sections(text):
            texts.append(body)
            metas.append({"source": fname, "section": section})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=["\n\n", "\n", "。", "！", "？", "；", "：", "，", " "],
    )
    docs = splitter.create_documents(texts, metadatas=metas)

#输出划分结果
    total_chunks = len(docs)
    unique_sections = set(d.metadata["section"] for d in docs)
    print(f"\n分割为 {total_chunks} 个文本块；共 {len(unique_sections)} 个独立章节。")

    counts = Counter(d.metadata["section"] for d in docs)
    print("\n章节分布统计：")
    for sec, c in counts.items():
        print(f" - {sec}: {c} 个文本块")


    print("\n开始嵌入与索引构建...")
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        # model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    vs = FAISS.from_documents(docs, embedding)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(INDEX_DIR))
    print(f"\n索引已保存至：{INDEX_DIR.resolve()}")


if __name__ == "__main__":
    main()
