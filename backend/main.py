from dotenv import load_dotenv
from pydantic import BaseModel

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import uvicorn

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messagges import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

import ytfinance as yf

load_dotenv()

app = FastAPI() 

model = ChatOpenAI(
    model = "gpt-4o",
    base_url = "https://api.openai.com/v1"
)

checkpointer = InMemorySaver()

agent = create_agent(
    llm = model,
    checkpointer = checkpointer,
    tools = []
)

class PromptObject(BaseModel):
    content: str
    id: str
    role: str

class RequestObject(BaseModel):
    prompt: PromptObject
    threadId: str
    responseId: str

@app.post("/api/chat")
async def chat(request: RequestObject):
    config = {
        'configurable': {
            'thread_id': request.threadId
        }
    }

    def generate():
        for token, _ in agent.stream(
            {'messages': [
                SystemMessage(content="You are a helpful financial assistant."),
                HumanMessage(content=request.prompt.content)
            ]},
            stream_mode = 'messages',
            config = config
        ):
            yield token.content

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers = {'Cache-Control': 'no-cache, no-transform',
                                      'Connection': 'keep-alive'
                                    })

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8888)