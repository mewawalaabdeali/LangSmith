from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

#Simple One-line Prompt
prompt = PromptTemplate.from_template("{question}")

model = ChatOpenAI()
parser = StrOutputParser()

#Chain: prompt -> model -> parser
chain = prompt | model | parser

result = chain.invoke({"question": "What is the implication of AI on AI/ML engineer jobs"})
print(result)