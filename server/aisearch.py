#!/usr/bin/env python
# coding: utf-8

import os
import jwt
import json
import time
import logging
import requests
import datetime
from opensearchpy import OpenSearch
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.document_loaders import TextLoader, S3DirectoryLoader
from langchain.chains import LLMChain
from yagpt import YandexGPTEmbeddings, YandexLLM
import uvicorn
from fastapi import FastAPI, Body
from pydantic import BaseModel


def read_json(file_path):
    with open(file_path) as file:
        data = json.load(file)
    return data


ROOT_PATH = '.'
PORT = 40000
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
BULK_SIZE = 1000
CA_PATH = f'{ROOT_PATH}/.opensearch/root.crt'
creds = read_json(f'{ROOT_PATH}/configs/credentials.json')
DB_USER = creds['db_user']
DB_PASS = creds['db_password']
DB_HOSTS = creds['db_hosts']
LLM_SERVICE_ACCOUNT_ID = creds['service_account_id']
LLM_KEY_ID = creds['key_id']
LLM_PRIVATE_KEY = creds['private_key']
S3_BUCKET = creds['bucket']
S3_BUCKET_PREFIX = creds['bucket_prefix']
S3_KEY_ID = creds['aws_access_key_id']
S3_SECRET_KEY = creds['aws_secret_access_key']
S3_ENDPOINT_URL= creds['endpoint_url']
FOLDER_ID = creds['folder_id']


class BotChain():
    def __init__(self, llm_service_account_id, llm_private_key, llm_key_id,
                 bucket, bucket_prefix, s3_key_id, s3_secret_key, s3_endpoint_url,
                 chunk_size, chunk_overlap, 
                 db_hosts, db_user, db_pass, ca_path, bulk_size,
                 folder_id):
        self.llm_service_account_id = llm_service_account_id
        self.llm_private_key = llm_private_key
        self.llm_key_id = llm_key_id
        self.bucket = bucket
        self.bucket_prefix = bucket_prefix
        self.s3_key_id = s3_key_id
        self.s3_secret_key = s3_secret_key
        self.s3_endpoint_url = s3_endpoint_url
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.db_hosts = db_hosts
        self.db_user = db_user
        self.db_pass = db_pass
        self.ca_path = ca_path
        self.bulk_size = bulk_size
        self.folder_id = folder_id
        self.token = self.ya_token(lag=360)
        self.embeddings = self.ya_embed()
        self.docsearch = self.db_docsearch()

    def db_connect(self, db_hosts, db_user, db_pass, ca_path):
        conn = OpenSearch(
            db_hosts,
            http_auth=(db_user, db_pass),
            use_ssl=True,
            verify_certs=True,
            ca_certs=ca_path
        )
        print('connection:', conn.info())
        return conn

    def ya_embed(self):
        embeddings = YandexGPTEmbeddings(
            iam_token=self.token['iamToken'], 
            folder_id=self.folder_id
        )
        return embeddings

    def db_docsearch(self):
        loader = S3DirectoryLoader(
            self.bucket,
            prefix=self.bucket_prefix,
            aws_access_key_id=self.s3_key_id, 
            aws_secret_access_key=self.s3_secret_key,
            endpoint_url=self.s3_endpoint_url
        )
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        docs = text_splitter.split_documents(documents)
        docsearch = OpenSearchVectorSearch.from_documents(
            docs,
            self.embeddings,
            opensearch_url=self.db_hosts[0],
            http_auth=(self.db_user, self.db_pass),
            use_ssl=True,
            verify_certs=True,
            ca_certs=self.ca_path,
            engine='lucene',
            bulk_size=self.bulk_size
        )
        return docsearch

    def db_simularity_search(self, query, k=2):
        self.refresh_token()
        query_docs = self.docsearch.similarity_search(query, k=k)
        return query_docs

    def ya_chain(self, temperature, instructions):
        llm = YandexLLM(
            api_key=self.llm_private_key,
            folder_id=self.folder_id,
            temperature=temperature,
            instruction_text=instructions
        )
        document_prompt = langchain.prompts.PromptTemplate(
            input_variables=['page_content'], 
            template='{page_content}'
        )
        document_variable_name = 'context'
        prompt_override = """
            Ответь на вопрос, используя информацию из текста ниже.
            Текст:
            -----
            {context}
            -----
            Вопрос:
            {query}
            """
        prompt = langchain.prompts.PromptTemplate(
            template=prompt_override,
            input_variables=['context', 'query']
        )
        llm_chain = langchain.chains.LLMChain(
            llm=llm, 
            prompt=prompt
        )
        chain = langchain.chains.combine_documents.stuff.StuffDocumentsChain(
            llm_chain=llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name
        )
        return chain


LOG_PATH = 'logs'
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
fh = logging.FileHandler(f'{LOG_PATH}/server.log')
LOGGER.addHandler(fh)
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
fh.setFormatter(formatter)
msg = 'Bot server started, logger initialized'
LOGGER.info(msg)

BOTCHAIN = BotChain(
    llm_service_account_id=LLM_SERVICE_ACCOUNT_ID, 
    llm_private_key=LLM_PRIVATE_KEY,
    llm_key_id=LLM_KEY_ID,
    db_hosts=DB_HOSTS, 
    db_user=DB_USER, 
    db_pass=DB_PASS, 
    ca_path=CA_PATH,
    chunk_size=CHUNK_SIZE, 
    chunk_overlap=CHUNK_OVERLAP,
    bulk_size=BULK_SIZE,
    bucket=S3_BUCKET,
    bucket_prefix=S3_BUCKET_PREFIX,
    s3_key_id=S3_KEY_ID, 
    s3_secret_key=S3_SECRET_KEY, 
    s3_endpoint_url=S3_ENDPOINT_URL,
    folder_id=FOLDER_ID
)
CHAIN, DOCS = None, None
msg = 'BotChain started, documents uploaded to database'
LOGGER.info(msg)


app = FastAPI()


"""
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
return JSONResponse(
status_code=exc.status_code,
content={“message”: “Server Error”}
)
"""


class InitParams(BaseModel):
    temperature: float
    instructions: str


class SearchParams(BaseModel):
    query: str
    k: int


class AskParams(BaseModel):
    query: str
    

@app.get('/datainfo')
async def data_config():
    data = {
        'bucket': creds['bucket'], 
        'bucket_info': creds['bucket_info']
    }
    return data


@app.get('/creds')
async def server_config():
    return {'data' : list(creds.keys())}


@app.get('/logs')
async def server_logs():
    with open(f'{LOG_PATH}/server.log') as file:
        logs = file.readlines()
    return {'data' : logs}


@app.post('/init')
async def init_chain(initparams: InitParams = Body(...)):
    instructions = initparams['instructions']
    temperature = initparams['temperature']
    global CHAIN
    CHAIN = BOTCHAIN.ya_chain(temperature, instructions)
    msg = 'Chain for instructions -{}- done, temperature = {}'.format(
        instructions.replace('\n', '').replace('\t', ''),
        str(temperature)
    )
    LOGGER.info(msg)
    return {'result' : msg}


@app.post('/search')
async def search_db(searchparams: SearchParams = Body(...)):
    query = searchparams['query']
    k = searchparams['k']
    global DOCS
    DOCS = BOTCHAIN.db_simularity_search(query=query, k=k)
    msg = 'Search for query -{}- done, k = {}'.format(
        query,
        k
    )
    LOGGER.info(msg)
    return {'result' : msg}

    
@app.post('/ask')
async def ask_chain(askparams: AskParams = Body(...)):
    query = askparams['query']
    response = CHAIN.run(input_documents=DOCS, query=query)
    msg = 'Q - {} | A - {}'.format(
        query,
        response
    )
    LOGGER.info(msg)
    return {'answer' : response}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=PORT, debug=True)
