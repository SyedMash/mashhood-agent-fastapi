import os
from fastapi import FastAPI
from agents import Agent, Runner, OpenAIChatCompletionsModel, function_tool, set_tracing_disabled
from openai import AsyncOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
import requests

load_dotenv()
set_tracing_disabled(disabled=True)

client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_AI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
app = FastAPI()


@function_tool
def get_mashhood_info():
    response = requests.get("https://syedmash.vercel.app/api/profile")
    return response.json()


mashhood_agent = Agent(
    name="Mashhood Agent",
    instructions="You are personal agent for mashhood. You can have all the information about Mashhood using get_mashhood_info tool. You have to give smart and precise answer",
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
    tools=[get_mashhood_info]
)


class ChatInput(BaseModel):
    message: str


@app.post("/")
async def root(user_input: ChatInput):
    result = await Runner.run(mashhood_agent, user_input.message)
    return {"answer": result.final_output}
