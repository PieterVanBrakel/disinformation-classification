import yaml
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# ✅ Use this if you installed langchain-chroma
from langchain_chroma import Chroma

# ❗ If you did NOT install langchain-chroma, use this instead:
# from langchain_community.vectorstores import Chroma


def load_config():
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "src" / "config.yaml"

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class RAGExplainer:
    def __init__(self):
        cfg = load_config()["rag"]
        project_root = Path(__file__).resolve().parents[2]

        # 🔎 Embeddings
        self.embeddings = OpenAIEmbeddings()

        # 📚 Vector store
        self.vectorstore = Chroma(
            persist_directory=str(project_root / cfg["vectorstore_path"]),
            embedding_function=self.embeddings
        )

        # 🔍 Retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": cfg["k"]}
        )

        # 🤖 LLM
        self.llm = ChatOpenAI(
            model=cfg["model"],
            temperature=cfg["temperature"]
        )

        # 🧠 Prompt
        self.prompt = ChatPromptTemplate.from_template("""
You are a strict fact-checking assistant.

You are given:
- a model prediction
- a text
- retrieved fact-check evidence

RULES:
- Use ONLY provided sources
- If insufficient evidence, say "Unverified"
- Do NOT hallucinate

---

Model prediction: {label}

Text:
{question}

Retrieved sources:
{context}

---

Return:
Verdict: True / False / Unverified
Explanation: grounded reasoning using sources
Evidence: bullet points from sources
""")

    def _format_docs(self, docs):
        return "\n\n".join(d.page_content for d in docs)

    def explain(self, text: str, label: str):
        """
        Run RAG pipeline:
        1. Retrieve documents
        2. Format context
        3. Call LLM with prompt
        """

        # 🔍 Retrieve relevant docs
        docs = self.retriever.invoke(text)
        context = self._format_docs(docs)

        # 🤖 Run LLM
        response = self.llm.invoke(
            self.prompt.format_messages(
                label=label,
                question=text,
                context=context
            )
        )

        return response.content