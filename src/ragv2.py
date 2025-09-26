import os
from dotenv import load_dotenv
from langsmith import traceable

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

os.environ['LANGCHAIN_PROJECT']='RAG Chatbot'

load_dotenv()

pdf_path = "data/islr.pdf"

#traced setup steps
@traceable(name='load_pdf', tags=['pdf', 'loader'], metadata={'loader':'PyPDFLoader'})
def load_pdf(path:str):
    loader = PyPDFLoader(path)
    return loader.load()

@traceable(name='split_documents')
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

@traceable(name='build_vectorstore')
def build_vectorstore(splits):
    embedding = OpenAIEmbeddings(model='text-embedding-3-small')
    vector_store = FAISS.from_documents(splits, embedding)
    return vector_store

@traceable(name='setup_pipeline')
def setup_pipeline(pdf_path:str):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs)
    vector_store = build_vectorstore(splits)
    return vector_store

# ---------- pipeline ----------
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you Don't Know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

#Build the index under traced setup
vectorstore = setup_pipeline(pdf_path)
retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={"k":4})

parallel= RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
})

chain = parallel | prompt | llm | StrOutputParser()

# ---------- run a query (also traced) ----------------
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ").strip()

#Give the visible run name + tags/metadata so its easy to find:
config = {
    "run_name":"pdf_rag_query"
}

ans = chain.invoke(q, config=config)
print("\nA: ", ans)