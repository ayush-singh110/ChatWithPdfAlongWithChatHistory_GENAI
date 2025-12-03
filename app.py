import streamlit as st
from langchain_classic.chains import create_retrieval_chain,create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
embeddings=OllamaEmbeddings(model="nomic-embed-text")

## set up streamlit
st.title("Conversational RAG with PDF Uploads and Chat History")
st.write("Upload the PDF and chat with their content")

api=os.getenv("GROQ_API_KEY")
llm=ChatGroq(api_key=api,model="llama-3.1-8b-instant")

session_id=st.text_input("Session ID",value="default-session")

if 'store' not in st.session_state:
    st.session_state.store={}

##Upload Pdfs
uploaded_files=st.file_uploader("Upload PDF",type="pdf",accept_multiple_files=True)
if uploaded_files:
    documents=[]
    for uploaded_file in uploaded_files:
        temppdf=f"./temp.pdf"
        with open(temppdf,"wb") as file:
            file.write(uploaded_file.getvalue())
            file_name=uploaded_file.name
        loader=PyPDFLoader(temppdf)
        docs=loader.load()
        documents.extend(docs)
    ##Split and create embeddings
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
    splits=text_splitter.split_documents(documents)
    vectorstore=FAISS.from_documents(splits,embeddings)
    retriever=vectorstore.as_retriever()

    contextualize_q_system_prompt=(
        "Given a chat history and the latest user question"
        "which might reference context in the chat history,"
        "formulate a standalone question which can be understood"
        "without the chat history. Do Not answer the question,"
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt=ChatPromptTemplate.from_messages(
        [
            ("system",contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human","{input}")
        ]
    )

    history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

    ##Answer question
    system_prompt=(
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer"
        "the question. If you don't know the answer, say that you"
        "don't know. Use three sentences maximum and keep the answer concise"
        "\n\n"
        "{context}"
    )
    qa_prompt=ChatPromptTemplate.from_messages(
        [
            ("system",system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human","{input}")
        ]
    )

    question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
    rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

    def get_session_history(session_id:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]

    conversation_rag_chain=RunnableWithMessageHistory(
        rag_chain,get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    user_input=st.text_input("Your Question: ")
    if user_input:
        session_history=get_session_history(session_id)
        response=conversation_rag_chain.invoke({"input":user_input},
            config={"configurable":{"session_id":session_id}})
        st.write("Assistant:",response['answer'])
        with st.expander("Click to see details"):
            st.write(st.session_state.store)
            st.write("Chat History:",session_history.messages)