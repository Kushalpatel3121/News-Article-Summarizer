# Import necessary libraries for news extraction.
import requests
import json
import os
from dotenv import load_dotenv
from newspaper import Article

# Defining headers
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
}

# URL of the original news article
article_url = 'https://www.aljazeera.com/news/2024/7/16/trump-receives-heros-welcome-in-first-appearance-since-assassination-bid'

# Creating the Session
session = requests.session()

# Fetching the news from the web
try:
    response = session.get(article_url, headers=headers, timeout=10)
    if response.status_code == 200:
        article = Article(article_url)
        article.download()
        article.parse()
        
        print(f'Title : {article.title}')
        print(f'Text : {article.text}')
        
    else:
        print(f'Failed to fetch article at {article_url}')
        
except Exception as e:
    print(f'Error occured while fetching article at {article_url}: {e}')
    
# Loading the environment variables  
load_dotenv()

# Loading the API KEY for Huggingface from the environment variable
HF_API_KEY = os.environ['HF_API_KEY']

# Importing necessary libraries for News Summarizer
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.prompts.prompt import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import field_validator, BaseModel, Field
from typing import List

# Defining the Pydantic class to be used as ouput parser
class ArticleSummary(BaseModel) : 
    title: str = Field(description="Title of the article")
    summary: List[str] = Field(description="Bulleted list summary of the article")
    
    # Field validator to check if the summary has more then three lines or not. The summary is expected to be of more than three lines.
    @field_validator('summary')
    def check_lines(cls, list_of_lines):
        if len(list_of_lines) < 3:
            raise ValueError("Generated summary has less than three bullet points.")
        return list_of_lines

# Creating the output parser object.
parser = PydanticOutputParser(pydantic_object=ArticleSummary)

# The prompt Template for LLM
propmt_template = PromptTemplate(
    input_variables=['article_title','article_text'],
    template='''
        You are an advanced AI assistant that summarizes online articles.

        Here's the article you want to summarize.

        ==================
        Title: {article_title}

        {article_text}
        ==================

        {format_instructions}
    ''',
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

# The actual prompt to be passed to the LLM
prompt = propmt_template.format(article_title=article.title, article_text=article.text)

# Initializing the LLM via HuggingFace.
# The model used is Mixtral - 7B - Instruct. (A chat Model).
llm = HuggingFaceEndpoint(
    huggingfacehub_api_token=HF_API_KEY,
    repo_id='mistralai/Mixtral-8x7B-Instruct-v0.1',
    temperature=0.2
)

# Extracting the response from the LLM
response = llm.invoke(prompt)

# Parsing the response using the ouput parser.
parsed_response = parser.parse(response)
print(parsed_response)

# Parsed Reponse (Title and the Bullet Points.)
print(parsed_response.title)
for item in parsed_response.summary:
    print('-',item)