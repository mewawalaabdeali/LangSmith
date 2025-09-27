from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv
import os

os.environ['LANGCHAIN_PROJECT']='Agent'

load_dotenv()

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str)-> str:
    """
    This function fetches the current weather data for a given city
    """
    url = f'https://api.weatherstack.com/current?access_key=f07d9636974c4120025fadf60678771b&query={city}'
    response = requests.get(url)
    return response.json()

llm = ChatOpenAI()

#Step2 - Pull the ReAct prompt from Langchain hub
prompt = hub.pull('hwchase17/react')

#Step3 - Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

#Step4 - Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True,
    max_iterations=5
)

#Step5 - Invoke
response = agent_executor.invoke({"input": "What is the birthplace of Virat Kohli and where does he currently reside and whats the temperature of his current city"})

print(response)

print(response['output'])