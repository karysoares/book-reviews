import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import openai
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Exibir mensagem de carregamento
st.title("Book QA System")
st.write("Loading models. This could take a while. Please wait.")

# Configuração dos modelos
model = "intfloat/multilingual-e5-large"
embeddings = HuggingFaceEmbeddings(model_name=model)

vector_store = Chroma(
    collection_name="books",
    embedding_function=embeddings,
    persist_directory="./books_vector.db"
)

# Função para gerar respostas
def generate_answer(prompt):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Given a context, question, you generate an answer to that given question."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    return response.choices[0].message.content

# Pipeline de perguntas e respostas
def qa_pipeline(question):
    search_results = vector_store.similarity_search(question, k=5)
    context = "\n\n".join([doc.page_content for doc in search_results])
    prompt = f"Context: {context}\n\nQuestion: {question}"
    answer = generate_answer(prompt)
    return answer

# Interface do Streamlit
st.header("Ask me anything about books!")
question = st.text_input("Type your question here:")

if st.button("Submit"):
    if question:
        with st.spinner("Generating answer..."):
            answer = qa_pipeline(question)
            st.write("**Answer:**")
            st.write(answer)
    else:
        st.write("Please enter a question.")

st.write("Note: This application uses a pre-trained language model to generate answers based on context.")