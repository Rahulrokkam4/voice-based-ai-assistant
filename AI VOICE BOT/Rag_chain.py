from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory


# os
import os
from dotenv import load_dotenv
load_dotenv()


def build_qa_chain():
    # Load CSV
    loader = CSVLoader(file_path="E:\\Datasets\\stuff_data.csv")
    documents = loader.load()

    # Split text (optional for large rows)
    text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Embedding
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Retriever
    retriever = vectorstore.as_retriever()

    # LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQCLOUD_API_KEY"),
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    # prompt for model
    prompt_template = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template=''' You are a helpful and professional AI assistant at Global EDC company.
        
                Always follow this interaction flow:
        
                Greet the user.
        
                RULES:
                - Use only the provided context to answer the question.
                - Do NOT make up answers if the information is not present in the context.
                - If unsure, say: "I'm not sure based on the current information."
                - Keep responses short, relevant, and in a friendly tone.
        
                Chat History:
                {chat_history}
        
                Context:
                {context}
        
                Question:
                {question}
        
                Helpful Answer:
                '''
    )

    # memory
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        input_key="question",
        return_messages=True,
        k=15
    )

    # Retrieval QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template, "memory": memory},
        memory=memory,
        input_key="question"
    )
    return qa_chain











# _qa_chain=None
# def load_qa_chain():
#     global _qa_chain
#     if _qa_chain is None:
#         print("⚙️ Building RAG chain only once...")
#         _qa_chain = build_qa_chain()
#     return _qa_chain



