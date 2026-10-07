"""Build a Qwen chat dataset and a metadata-rich retrieval JSONL from current articles.

This pilot deliberately keeps source text small and auditable. Add official articles to
data/legal/current_articles.jsonl; do not paste undocumented web content into training data.
"""
import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                row["_line"] = line_no
                yield row


def build_sft(articles):
    rows = []
    for article in articles:
        rows.append({
            "id": f"article-{article['clause_id']}",
            "conversations": [
                {"from": "user", "value": f"请给出{article['law_name']}{article['article_label']}的现行内容，并说明版本。"},
                {"from": "assistant", "value": (
                    f"{article['law_name']}{article['article_label']}（{article['version_label']}）规定："
                    f"{article['text']}"
                )},
            ],
            "metadata": {"clause_id": article["clause_id"], "task": "current_article_qa"},
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/legal/current_articles.jsonl")
    parser.add_argument("--sft-output", default="data/sft/current_law_sft.generated.json")
    parser.add_argument("--retrieval-output", default="data/legal/current_retrieval.jsonl")
    args = parser.parse_args()

    articles = list(read_jsonl(Path(args.input)))
    if not articles:
        raise SystemExit("No articles found")
    for article in articles:
        required = {"clause_id", "law_id", "law_name", "version_id", "article_no", "text", "source_url"}
        missing = required - article.keys()
        if missing:
            raise ValueError(f"line {article['_line']} missing fields: {sorted(missing)}")

    sft_path = Path(args.sft_output)
    sft_path.parent.mkdir(parents=True, exist_ok=True)
    sft_path.write_text(json.dumps(build_sft(articles), ensure_ascii=False, indent=2), encoding="utf-8")

    retrieval_path = Path(args.retrieval_output)
    retrieval_path.parent.mkdir(parents=True, exist_ok=True)
    with retrieval_path.open("w", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps({
                "page_content": article["text"],
                "metadata": {k: v for k, v in article.items() if not k.startswith("_") and k != "text"},
            }, ensure_ascii=False) + "\n")
    print(f"articles={len(articles)} sft={sft_path} retrieval={retrieval_path}")


if __name__ == "__main__":
    main()
