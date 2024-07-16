import requests
import json
import os
from dotenv import load_dotenv
from newspaper import Article

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
}

article_url = 'https://www.nytimes.com/2024/07/06/world/europe/france-parliamentary-election-2024.html'

session = requests.session()

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
    
load_dotenv()
HF_API_KEY = os.environ['HF_API_KEY']

from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.prompts.prompt import PromptTemplate

propmt_template = PromptTemplate(
    input_variables=['article_title','article_text'],
    template='''
        You are an advanced AI assistant that summarizes online articles into bulleted lists

        Here's the article you want to summarize.

        ==================
        Title: {article_title}

        {article_text}
        ==================

        Now, provide a summarized version of the article in a bulleted list format.
    '''
)

prompt = propmt_template.format(article_title=article.title, article_text=article.text)

llm = HuggingFaceEndpoint(
    huggingfacehub_api_token=HF_API_KEY,
    repo_id='mistralai/Mixtral-8x7B-Instruct-v0.1',
    temperature=0.2
)

response = llm.invoke(prompt)
print(response)