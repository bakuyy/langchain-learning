'''
Playing around with OpenAI and langchain!

> set up langchain / venv with poetry
> set up environment and tracing with langsmith
> played around with prompt templates (templates take in raw user input to return data)
'''
import getpass
import os
from dotenv import load_dotenv
from langsmith import Client
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"  

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

model = ChatOpenAI(model_name="gpt-4o-mini")
client = Client(api_key=os.environ.get("LANGSMITH_API_KEY"))

messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("My name is sophie!"),
]
print(client.list_runs())
response = model.invoke(messages)
print("response:", response.content)


from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"language": "Italian", "text": "My name is Sophie and I love watermelon."})
print(prompt)

prompt.to_messages()

response = model.invoke(prompt)
print(response.content)