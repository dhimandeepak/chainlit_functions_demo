from dotenv import load_dotenv
import chainlit as cl
from movie_functions import get_now_playing_movies, get_showtimes, buy_ticket
import json
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
You are a virtual assistant who has the capability to provide a user information about movies, showtimes and possible help them book tickets.

For list current movies requests, use the following function call format:
{"function": "get_now_playing_movies", "parameters": {}}

To get showtimes for a movie, use the following function call format:
{"function": "get_showtimes", "parameters": {"title": "title", "location": "location"}}

If the user indicates they want to buy a ticket, use the following function call format to confirm the details first:
{"function": "buy_ticket", "parameters": {"theater": "theater", "movie": "movie", "showtime": "showtime"}}

After receiving the results of a function call, incorporate that information into your response to the user.
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
    print("response " + response_message.content)
    function_name = ""
    try:
        if response_message.content.__contains__("\n\n"):
            print("Response contains double newline.")
            # Split the string to isolate the JSON part
            split_string = response_message.content.split("\n\n")  # Split by the double newline
            
            # The JSON part is usually the second part after the split
            json_part = split_string[1]
            parsed_json = json.loads(json_part)

            # Print the extracted and parsed JSON
            print(f"parsed json ")
            print (parsed_json)
            function_name = parsed_json["function"]
        else:
            print("Response does not contain double newline.")
              # Split the string to isolate the JSON part
            split_string = response_message.content.split("\n\n")  # Split by the double newline

            # The JSON part is usually the first part after the split            
            json_part = split_string[0]
            parsed_json = json.loads(json_part)

            # Print the extracted and parsed JSON
            print(f"parsed json ")
            print (parsed_json)
            function_name = parsed_json["function"]
    except Exception as e:
            print(f"Error calling get_showtimes: {e}")
    
    if function_name == "get_now_playing_movies":
        print("Function is get_now_playing_movies.")

        # Check if the response contains get_now_playing_movies
        # call the function from movie_functions.py
        current_movies = get_now_playing_movies()

        #Append the function result to the message history
        message_history.append({"role" : "function", "name" : "get_now_playing_movies",  "content": current_movies})
        response_message = await generate_response(client, message_history, gen_kwargs)
    elif function_name == "get_showtimes":
        print("Function is get_showtimes.")
        try:
            parameters = parsed_json["parameters"]
            print(f"parameters: {parameters}")
            title = parameters["title"]
            location = parameters["location"]
           
            show_times = get_showtimes(title, location)
            #Append the function result to the message history
            message_history.append({"role" : "function", "name" : "get_showtimes",  "content": show_times})
            response_message = await generate_response(client, message_history, gen_kwargs)

        except Exception as e:
            print(f"Error calling get_showtimes: {e}")
    elif function_name == "buy_ticket":
        print("Function is buy_ticket.")
        try:
            parameters = parsed_json["parameters"]
            print(f"parameters: {parameters}")
            theater = parameters["theater"]
            movie = parameters["movie"]
            showtime = parameters["showtime"]
          
            #if theater, movie, showtime is not empty  
            if theater and movie and showtime:
                buy_ticket_result = buy_ticket(theater, movie, showtime)
                #Append the function result to the message history
                message_history.append({"role": "function", "name" : "buy_ticket", "content": buy_ticket_result})
                response_message = await generate_response(client, message_history, gen_kwargs)
            else:
                    print("Missing theater, movie or showtime for buy_ticket")

        except Exception as e:
            print(f"Error calling get_showtimes: {e}")
    
    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)

if __name__ == "__main__":
    cl.main()