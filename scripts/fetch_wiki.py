# scripts/fetch_wiki.py
from pathlib import Path
import wikipedia

def main():
    wikipedia.set_lang("zh")
    topic = "人工智能"

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    try:
        page = wikipedia.page(topic, auto_suggest=False)
    except Exception:
        page = wikipedia.page(topic)

    text = page.content.strip()
    path = out_dir / f"wiki_{topic}.txt"
    path.write_text(text, encoding="utf-8")
    print(f"已保存: {path}")

if __name__ == "__main__":
    main()
