from dotenv import load_dotenv
import chainlit as cl

load_dotenv()

# Note: If switching to LangSmith, uncomment the following, and replace @observe with @traceable
# from langsmith.wrappers import wrap_openai
# from langsmith import traceable
# client = wrap_openai(openai.AsyncClient())

from langfuse.decorators import observe
from langfuse.openai import AsyncOpenAI
 
client = AsyncOpenAI()

gen_kwargs = {
    "model": "gpt-4o",
    "temperature": 0.2,
    "max_tokens": 500
}

SYSTEM_PROMPT = """\
You are a chatbot dedicated to helping users discover movies currently playing in theaters. Your primary role is to provide accurate and up-to-date information about films, showtimes, and theater locations. You can perform the following functions based on user requests:

List Movies: Call the `get_now_playing_movies()` function to provide users with a list of movies currently playing in theaters.
Showtimes: Call the `get_showtimes(movie_id)` function to provide detailed showtimes for a selected movie.
Movie Ratings: Call the `get_reviews(movie_id)` function to share ratings and reviews for a specific movie.
Buy Tickets: Call the buy_ticket(theater, movie, showtime) function to purchase a ticket for a movie.
User Persona: Your users may include moviegoers of all ages looking for entertainment options. Tailor your responses to their interests and preferences.
Tone: Maintain a friendly, enthusiastic, and informative tone. Aim to enhance the user's movie-going experience by providing clear and helpful information.
"""

@observe
@cl.on_chat_start
def on_chat_start():    
    message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    cl.user_session.set("message_history", message_history)

@observe
async def generate_response(client, message_history, gen_kwargs):
    response_message = cl.Message(content="")
    await response_message.send()

    stream = await client.chat.completions.create(messages=message_history, stream=True, **gen_kwargs)
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)
    
    await response_message.update()

    return response_message

@cl.on_message
@observe
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})
    
    response_message = await generate_response(client, message_history, gen_kwargs)

    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)

if __name__ == "__main__":
    cl.main()
