# book-reviews

Este repositório contém uma série de notebooks e scripts para preparação de dados, análise de dados, classificação de gêneros, análise de sentimentos, indexação e criação de um pipeline de RAG (Retrieval-Augmented Generation) para resenhas de livros usando Python 3.10.12.


# Preparação do ambiente

git clone https://github.com/karysoares/book-reviews.git

cd book-reviews

python -m venv env

source env/bin/activate 

pip install -r requirements.txt

Crie um ficheiro `.env` na raiz do repositório com `OPENAI_API_KEY=sua_chave` para o Streamlit e para o notebook 06.

# Dados completos (RAG com o catálogo inteiro)

Os ficheiros grandes **não estão no Git** (pastas `data/`, `books.db`, `books_vector.db` estão no `.gitignore`). Para o pipeline completo, usa os **mesmos CSVs de sempre** (por exemplo `books_data.csv` e `books_rating.csv` que já usaste no projeto):

1. Cria `data/` na raiz e copia para lá os CSVs do teu backup ou da fonte original.
2. Abre e corre **`notebooks/01_data_preparation.ipynb`** até ao fim (gera `books.db` na raiz e os CSVs de treino em `data/`).
3. Abre e corre **`notebooks/05_indexing_pipeline.ipynb`** até ao fim (gera **`books_vector.db`** na raiz com o índice Chroma).

Para ver o que falta localmente:

```bash
python scripts/check_data.py
```

Só para testar a app **sem** o dataset completo, podes gerar um índice mínimo de demonstração (alguns livros fictícios):

```bash
python scripts/build_demo_index.py
```

# Testes

pytest

# Execução dos Notebooks
Para executar os notebooks, inicie o Jupyter Notebook:

  jupyter notebook
  
# Execução do Streamlit
Na raiz do repositório:

  streamlit run rag_pipeline.py
