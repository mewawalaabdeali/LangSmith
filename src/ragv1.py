import os
from dotenv import load_dotenv
from langchain_community.document_loaders import csv_loader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

pdf_path = "data/islr.pdf"

#load pdf
loader = PyMuPDFLoader(pdf_path)
docs = loader.load()

#Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = splitter.split_documents(docs)

#Embed + index
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
vector_store = FAISS.from_documents(splits, embeddings)
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={"k":4})

#Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

#Chain
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    "context":retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()

#Ask question
print("PDF RAG ready. Ask a question (or Ctrl+C to exit)")
q = input("\n Q:")
ans = chain.invoke(q.strip())
print("\nA: ", ans)
