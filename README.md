# Ingestão e Busca Semântica com PDF

Sistema de **RAG (Retrieval-Augmented Generation)** em Python que ingere um PDF, armazena seus embeddings em PostgreSQL com pgVector e permite fazer perguntas sobre o conteúdo via chat no terminal. As respostas são geradas **exclusivamente** a partir do que está no documento — se a informação não estiver lá, o sistema responde que não tem informações para responder.

Projeto desenvolvido como desafio do MBA em IA (Full Cycle).

## Stack

- **Python 3.10+**
- **LangChain** (core, community, text-splitters, google-genai, postgres)
- **Google Gemini**
  - Embeddings: `models/embedding-001`
  - LLM: `gemini-2.5-flash-lite`
- **PostgreSQL 16 + pgVector** (via Docker)

## Estrutura

```
mba-ia-ingestao-busca-pdf/
├── docker-compose.yml       # Postgres + pgVector
├── requirements.txt         # Dependências Python
├── .env.example             # Template de variáveis de ambiente
├── src/
│   ├── ingest.py            # Ingestão do PDF (chunks 1000/150 + embeddings)
│   ├── search.py            # Busca vetorial (k=10) + prompt
│   └── chat.py              # CLI interativa
├── document.pdf             # PDF a ser ingerido (você fornece)
└── README.md
```

## Pré-requisitos

- Python 3.10 ou superior
- Docker e Docker Compose
- Chave de API do Google Gemini — veja a seção [Criando a chave do Gemini](#criando-a-chave-do-gemini)

## Setup

### 1. Clonar o repositório

```bash
git clone https://github.com/diolehner/mba-ia-ingestao-busca-pdf.git
cd mba-ia-ingestao-busca-pdf
```

### 2. Criar e ativar o ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar variáveis de ambiente

```bash
cp .env.example .env
```

Edite o `.env` e preencha `GOOGLE_API_KEY` com a sua chave do Gemini.

### 5. Subir o banco de dados

```bash
docker compose up -d
```

O Postgres ficará disponível em `localhost:5432` com:
- usuário: `postgres`
- senha: `postgres`
- banco: `rag`

### 6. Colocar o PDF na raiz

Coloque seu arquivo PDF na raiz do projeto com o nome **`document.pdf`**. Se quiser trocar o documento depois, basta substituir o arquivo e rodar a ingestão de novo (o script limpa a collection anterior antes de inserir).

## Ordem de execução

### Ingestão

```bash
python src/ingest.py
```

Saída esperada:
```
Carregando PDF: /.../document.pdf
Páginas carregadas: N
Chunks gerados (size=1000, overlap=150): M
Limpando collection 'pdf_documents' antes da ingestão...
Gerando embeddings e inserindo no pgVector...

Ingestão concluída: M chunks salvos na collection 'pdf_documents'.
```

### Chat

```bash
python src/chat.py
```

Exemplo de interação:
```
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

Digite `sair` (ou `Ctrl+C`) para encerrar.

## Criando a chave do Gemini

1. Acesse [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Faça login com sua conta Google.
3. Clique em **Create API key** e gere uma nova chave.
4. Copie a chave e cole em `.env` na variável `GOOGLE_API_KEY`.

> **Atenção**: nunca faça commit do arquivo `.env`. Ele já está listado no `.gitignore`.

## Como funciona

### Ingestão (`src/ingest.py`)
1. Carrega o PDF com `PyPDFLoader`.
2. Divide em chunks de **1000 caracteres** com **overlap de 150** usando `RecursiveCharacterTextSplitter`.
3. Limpa a collection anterior no pgVector (re-ingestão limpa).
4. Gera embeddings com `GoogleGenerativeAIEmbeddings` (`models/embedding-001`).
5. Persiste os vetores no PostgreSQL via `langchain_postgres.PGVector`.

### Busca (`src/search.py`)
1. Vetoriza a pergunta do usuário.
2. Executa `similarity_search_with_score(query, k=10)`.
3. Concatena os 10 trechos mais relevantes no campo `CONTEXTO` do prompt.
4. Envia o prompt ao LLM (`gemini-2.5-flash-lite`) com regras rígidas de "responda apenas pelo contexto".

### Chat (`src/chat.py`)
Loop interativo no terminal que recebe perguntas, chama `ask()` e imprime a resposta.

## Troubleshooting

**Erro de conexão com o banco**: verifique se o container está rodando com `docker compose ps`.

**`GOOGLE_API_KEY não definida`**: confirme que o `.env` foi criado a partir do `.env.example` e que a chave foi preenchida.

**Respostas sempre "Não tenho informações..."**: verifique se a ingestão foi executada com sucesso (`python src/ingest.py`) **antes** de iniciar o chat. O banco precisa ter os embeddings do PDF.

**Trocar de PDF**: substitua o arquivo `document.pdf` na raiz e rode `python src/ingest.py` novamente — a collection é limpa automaticamente antes da nova ingestão.

## Licença

MIT.
