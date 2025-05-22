import os
import json
import asyncio
import logging
import uuid
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from agents import Agent, Runner, TResponseInputItem, ItemHelpers, trace, GuardrailFunctionOutput, InputGuardrail, WebSearchTool, RunItemStreamEvent
from tools.cinema_schedule_tools import cinema_schedule_tool
from tools.send_note_tools import send_note

load_dotenv(".env.local")

logging.basicConfig(level=logging.INFO)

app = FastAPI()

class Request(BaseModel):
    messages: list[TResponseInputItem]

class MovieOutput(BaseModel):
    reasoning: str
    related_to_movies: bool

cinema_schedule_agent = Agent(
    name="Cinema Schedule Agent",
    handoff_description="A specialist that knows cinema schedules and can recommend movies.",
    instructions="You should be able to tell the user the schedule of a movie. You can recommend movies. Today is 22052025. The price from cinema_schedule_tool 15000 means 150 UAH 00 kop.",
    tools=[cinema_schedule_tool]
)

notes_agent = Agent(
    name="Notes Agent",
    instructions="You can take notes of the conversation and send them to the user in Telegram",
    handoff_description="A specialist that can take notes of the conversation and send them to the user in Telegram.",
    tools=[send_note]
)

guardrails_agent = Agent(
    name="Guardrails Agent",
    instructions="User's questions should relate to movies. Greetings are OK. NO homework questions.",
    output_type=MovieOutput,
)

async def movie_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrails_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(MovieOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.related_to_movies
    )

web_search_agent = Agent(
    name="Web Search Agent",
    handoff_description="A specialist that can search the web for reviews for movies. Trigger this agent when user asks for reviews for movies.",
    instructions="You can search the web for reviews for movies. Let the user know what is the rating for the given movie. Was it critically acclaimed etc.",
    tools=[WebSearchTool(search_context_size="high")]
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="You are a specialist that delegates tasks to other agents.",
    handoffs=[cinema_schedule_agent, notes_agent, web_search_agent],
    input_guardrails=[InputGuardrail(guardrail_function=movie_guardrail)]
)

async def stream_agent_events(input_text: str):
    thread_id = uuid.uuid4().hex
    try:
        result_stream = Runner.run_streamed(triage_agent, input=input_text)
        message_output_created = False
        async for event in result_stream.stream_events():
            if isinstance(event, RunItemStreamEvent):
                logging.info(f"name of event: {event.name}")
                if event.name == 'tool_output':
                    message_output_created = True
            if not message_output_created:
                continue
            if event.type == "raw_response_event":
                try:
                    delta_text = event.data.delta
                    logging.info(f"raw delta token: {delta_text}")
                except AttributeError:
                    delta_text = None
                if delta_text:
                    logging.info(f"Delta token: {delta_text}")
                    yield f'0:{json.dumps(delta_text)}\n'
    except Exception as e:
         logging.error(f"Exception in stream_agent_events: {e}")
         yield '0:"... I\'m sorry, I cannot answer this question. Would you like me to help you with picking a movie to watch? 🍿"'

@app.post("/api/chat")
async def handle_chat_data(request: Request):
    try: 
        response = StreamingResponse(stream_agent_events(request.messages))
        response.headers['x-vercel-ai-data-stream'] = 'v1'
        return response
    except:
        return StreamingResponse(iter([f'0:"I\'m sorry, I cannot answer this question. Would you like me to help you with picking a movie to watch? 🍿"']), media_type="text/event-stream")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
