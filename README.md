# book-reviews

Este repositório contém uma série de notebooks e scripts para preparação de dados, análise de dados, classificação de gêneros, análise de sentimentos, indexação e criação de um pipeline de RAG (Retrieval-Augmented Generation) para resenhas de livros usando Python 3.10.12.


# Preparação do ambiente

git clone https://github.com/karysoares/book-reviews.git
cd book-reviews
python -m venv env
source env/bin/activate 
pip install -r requirements.txt

# Execução dos Notebooks
Para executar os notebooks, inicie o Jupyter Notebook:
  jupyter notebook
  
# Execução do Streamlit
Para executar o pipeline RAG utilizando Streamlit, execute o comando:

  streamlit run rag_pipeline.py
