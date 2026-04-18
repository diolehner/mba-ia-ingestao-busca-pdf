import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "models/gemini-embedding-001"
BATCH_SIZE = 90
BATCH_COOLDOWN_SECONDS = 65


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Variável de ambiente {name} não definida. "
            f"Copie .env.example para .env e preencha os valores."
        )
    return value


def _sanitize_metadata(metadata: dict) -> dict:
    return {
        key: value
        for key, value in metadata.items()
        if isinstance(value, (str, int, float, bool))
    }


def ingest_pdf() -> None:
    load_dotenv()

    google_api_key = _require_env("GOOGLE_API_KEY")
    database_url = _require_env("DATABASE_URL")
    collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME", "pdf_documents")
    pdf_path = os.getenv("PDF_PATH", "document.pdf")

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(
            f"PDF não encontrado em '{pdf_path}'. "
            f"Coloque o arquivo na raiz do projeto e tente novamente."
        )

    print(f"Carregando PDF: {pdf_file.resolve()}")
    loader = PyPDFLoader(str(pdf_file))
    pages = loader.load()
    print(f"Páginas carregadas: {len(pages)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=False,
    )
    chunks = splitter.split_documents(pages)
    for chunk in chunks:
        chunk.metadata = _sanitize_metadata(chunk.metadata)
    print(f"Chunks gerados (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}): {len(chunks)}")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=google_api_key,
    )

    store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=database_url,
        use_jsonb=True,
    )

    print(f"Limpando collection '{collection_name}' antes da ingestão...")
    try:
        store.delete_collection()
    except Exception as exc:
        print(f"  (collection inexistente ou já vazia: {exc})")

    store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=database_url,
        use_jsonb=True,
    )

    total = len(chunks)
    batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    print(
        f"Gerando embeddings e inserindo no pgVector "
        f"({batches} lote(s) de até {BATCH_SIZE}, cooldown {BATCH_COOLDOWN_SECONDS}s)..."
    )
    for index in range(0, total, BATCH_SIZE):
        batch = chunks[index : index + BATCH_SIZE]
        batch_number = index // BATCH_SIZE + 1
        print(f"  Lote {batch_number}/{batches} ({len(batch)} chunks)...", flush=True)
        store.add_documents(batch)
        if index + BATCH_SIZE < total:
            print(
                f"  Aguardando {BATCH_COOLDOWN_SECONDS}s para respeitar o free tier...",
                flush=True,
            )
            time.sleep(BATCH_COOLDOWN_SECONDS)

    print(
        f"\nIngestão concluída: {total} chunks salvos na collection "
        f"'{collection_name}'."
    )


if __name__ == "__main__":
    ingest_pdf()
