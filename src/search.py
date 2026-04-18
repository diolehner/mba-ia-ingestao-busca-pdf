import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector


EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-2.5-flash-lite"
TOP_K = 10

PROMPT_TEMPLATE = """CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Variável de ambiente {name} não definida. "
            f"Copie .env.example para .env e preencha os valores."
        )
    return value


def _build_vector_store() -> PGVector:
    google_api_key = _require_env("GOOGLE_API_KEY")
    database_url = _require_env("DATABASE_URL")
    collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME", "pdf_documents")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=google_api_key,
    )
    return PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=database_url,
        use_jsonb=True,
    )


def search_context(question: str) -> str:
    store = _build_vector_store()
    results = store.similarity_search_with_score(question, k=TOP_K)
    return "\n\n".join(document.page_content for document, _score in results)


def build_prompt(question: str) -> str:
    contexto = search_context(question)
    template = PromptTemplate.from_template(PROMPT_TEMPLATE)
    return template.format(contexto=contexto, pergunta=question)


def ask(question: str) -> str:
    load_dotenv()

    prompt_text = build_prompt(question)

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=_require_env("GOOGLE_API_KEY"),
        temperature=0,
    )
    response = llm.invoke(prompt_text)
    return response.content.strip()
