import json
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


def load_factchecks(path: Path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def build_vectorstore():

    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "raw"

    input_file = data_path / "factchecks.jsonl"

    if not input_file.exists():
        raise FileNotFoundError(
            "Missing factchecks.jsonl in data/raw/"
        )

    records = load_factchecks(input_file)

    docs = []

    for r in records:
        docs.append(
            Document(
                page_content=f"""
Claim: {r['claim']}
Verdict: {r['verdict']}
Explanation: {r['explanation']}
Source: {r.get('source', 'unknown')}
"""
            )
        )

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(),
        persist_directory=str(project_root / "data/vectorstore/factchecks")
    )

    vectorstore.persist()

    print(f"Built vectorstore with {len(docs)} documents.")


if __name__ == "__main__":
    build_vectorstore()