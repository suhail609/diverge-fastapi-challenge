# import nltk
# nltk.download('punkt')
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from nltk.tokenize import sent_tokenize
from googleapiclient.discovery import build
import openai
import re
import json
import requests
from mangum import Mangum

from fastapi import FastAPI
from typing import Optional

load_dotenv()


app = FastAPI()
handler = Mangum(app)

api_key=os.getenv("DEVELOPER_KEY")
cse_id=os.getenv("CSE_ID")
service = build(
    "customsearch",
    "v1",
    developerKey=api_key,
    static_discovery=False
)

openai.api_key=os.getenv("OPENAPI_API_KEY")

def chat_model_gpt(messages, temperature=0.1, functions_json=None, n=1):
    # Session may not be necessary, and AI completion could occur one without it
    openai.requestssession = requests.Session()
    
    # Actual request for a response from OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=temperature,
        n=n)
    # Extracting the useful data
    answer = response["choices"][0]["message"]
    try:
        openai.requestssession.close()
    except Exception as e:
        pass
    return answer

def improve_query(user_query, chat_model):
    # Prompt to give to ChatGPT to produce desired query
    prompt = [{
        "role": "system", 
        "content": '''
        You are a search engine expert, specifically on Google. 
        Given multiple user questions convert them to optimised google search queries.
        In output generate the queries in same order as input.
        
        Example-1
        #####
        initial query=Why has web3 gone out of limelight? Who was the founder of FTX?
        output=1. reasons behind failure of web3
        2. founder of FTC
        #####

        Example-2
        #####
        initial query=What is the global perspective on United Arab Emirate hosting COP28? Why is Israel fighting Hamas? When is the solar solstice?
        output=1. united arab emirates hosting cop28 perspectives
        2. israel-hamas war reasons
        3. solar solstice date
        '''
    }, {
        "role": "user", 
        "content": f'''
        Given input user-questions convert to appropriate google search query.
        Ensure that the generated query is optimal for google search, i.e, we get best results.

        input-query=[user given question]

        output=[optimal google search query]

        input-query={user_query}

        output='''
    }]

    return (
        # sent_tokenize may not be needed, 
        # chat_gpt could be sufficient inshaAllah
        sent_tokenize(user_query) + 
        [
            re.sub("[0-9]+\.", '', q).strip(' ') for q in
            chat_model(prompt, temperature=0.5, n=2)["content"].split('\n')
        ]
    )

def google_search(search_query, time_range):
    # Google search for the first 10 results
    res = service.cse().list(q=search_query, start=1, num=10, cx=cse_id, dateRestrict=time_range).execute()
    return res

def print_search_results(res):
    # Printing search results with indent
    results = []
    for result in res['items']:
        result['update_time'] = result['snippet'].split('...')[0] if len(result['snippet'].split('...')[0])<=15 else ''
        results.append(result)
    print(json.dumps(results, indent=2))



def perform_optimized_search(search_query):
    # Use OpenAI's ChatGPT to improve the query, 
    # to be more relevant to get more relevant results from google
    improved_search_query = improve_query(search_query, chat_model_gpt)
    # Make the google search
    res = google_search(improved_search_query, 'w1')
    # Prints info about search result links to console
    return res

if __name__ == '__main__':
    # Feel free to select a different search query
    search_query = "Elon Musk news"
    result = perform_optimized_search(search_query)
    print_search_results(result)


@app.get("/")
def root():
    return 'Welcome'


@app.get("/default")
def default():
    search_query = "Elon Musk news"
    result = perform_optimized_search(search_query)
    print_search_results(result)

    return {
            "status": "executed",
            "result": result
        }

@app.get("/search")
def optimized_search(q: Optional[str] = None):

    if q != None:
        result = perform_optimized_search(q)
        return result
    
    return 'Please provide search query'
