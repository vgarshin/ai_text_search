#!/usr/bin/env python
# coding: utf-8

import os
import jwt
import json
import time
import boto3
import requests
import datetime
import uvicorn
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
import langchain
from opensearchpy import OpenSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import OpenSearchVectorSearch, Chroma
from langchain_community.vectorstores import utils as chromautils
from langchain.chains import (
    ConversationalRetrievalChain, 
    LLMChain, 
    create_history_aware_retriever,
    create_retrieval_chain
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.load import dumpd, dumps, load, loads
from langchain_community.document_loaders import TextLoader, S3DirectoryLoader, DirectoryLoader, S3FileLoader
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_community.llms import YandexGPT
from langchain_community.embeddings.yandex import YandexGPTEmbeddings

from bookbuilder import booktools
from auth import authorization_middleware
from config import SETTINGS
from logger import book_logger_qh


class BookSearcher():
    def __init__(self, settings, logger):
        self.settings = settings
        self.logger = logger
        self.embeddings = YandexGPTEmbeddings(
            folder_id=self.settings.folder_id,
            api_key=self.settings.secret_key,
            sleep_interval=self.settings.sleep_emb
        )
        self.vectorstore = self.db_vectorstore()

    def rag_files(self):
        session = boto3.session.Session()
        s3 = session.client(
            service_name='s3',
            aws_access_key_id=self.settings.aws_access_key_id,
            aws_secret_access_key=self.settings.aws_secret_access_key,
            endpoint_url=self.settings.endpoint_url
        )
        all_files = [
            key['Key'] for key
            in s3.list_objects(
                Bucket=self.settings.bucket, 
                Prefix=self.settings.bucket_prefix
            )['Contents']
        ]
        rag_files = [x for x in all_files if '.ipynb_checkpoints' not in x]
        rag_files = [x for x in rag_files if x[-1] != '/']
        return rag_files

    def docs_from_files(self, rag_files):
        docs = []
        for file_path in rag_files:
            try:
                # load file from storage
                doc = S3FileLoader(
                    self.settings.bucket,
                    file_path,
                    aws_access_key_id=self.settings.aws_access_key_id,
                    aws_secret_access_key=self.settings.aws_secret_access_key,
                    endpoint_url=self.settings.endpoint_url
                ).load()
        
                # metadata extract
                for d in doc:
                    prompt = d.model_dump()['page_content'][:2000]
                    instruction_text = """Определи название документа,
                    авторов и организацию, период, к которому относится документ. 
                    Представь результат в виде JSON с полями title, authors, org, period. 
                    Если названий, авторов и периодов несколько, то укажи их через 
                    запятую как одну строку. Пустые значения заполни пустой строкой."""
                    res = booktools.ask_llm(
                        model_name=self.settings.model_name,
                        prompt=prompt,
                        instruction_text=instruction_text,
                        folder_id=self.settings.folder_id,
                        api_key=self.settings.secret_key,
                        temperature=self.settings.temperature,
                        max_tokens=self.settings.max_tokens
                    )
                    res = eval(res.replace('`', ''))
                    msg = f'Metadata extract - {res}'
                    self.logger.info(msg)
                    if not isinstance(res, list):
                        d.metadata.update(res)
                
                # collection of docs
                docs.extend(doc)
            except Exception as e:
                msg = f'Error file -{file_path}- processing: {e}'
                self.logger.error(msg)
        return docs

    def docs_splitted(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=[
                '\n\n',
                '\n',
                ' ',
                '.',
                ','
            ],
            chunk_size=self.settings.chunk_size, 
            chunk_overlap=self.settings.chunk_overlap
        )
        docs_splitted = text_splitter.split_documents(docs)
        return docs_splitted

    def db_connect(self):
        client = OpenSearch(
          self.settings.db_hosts,
          http_auth=(self.settings.db_user, self.settings.db_password),
          use_ssl=True,
          verify_certs=True,
          ca_certs=self.settings.ca
        )
        indices_exist = client.indices.stats()['indices'].keys()
        msg = f'Database connection - {client.info()}'
        self.logger.info(msg)
        msg = f'Indices exist: {indices_exist}'
        self.logger.info(msg)
        if self.settings.new_index:
            if self.settings.os_index in client.indices.stats()['indices'].keys():
                client.indices.delete(index=self.settings.os_index)
            else:
                msg = f'Index {self.settings.os_index} does not exist'
                self.logger.info(msg)
        else:
            indices_stats = {}
            for k, v in client.indices.stats(index=self.settings.os_index)['indices'].items():
                indices_stats[k] = v['primaries']['docs']
                msg = f'Index info {k}, {v["primaries"]["docs"]}'
                self.logger.info(msg)
        return client, indices_exist, indices_stats

    def db_vectorstore(self):
        if self.settings.new_index:
            rag_files = self.rag_files()
            msg = f'Creating new index info {self.settings.os_index}, from {len(rag_files)} files'
            self.logger.info(msg)
            docs = self.docs_from_files(rag_files)
            msg = f'Documents processed to load: {len(docs)}'
            self.logger.info(msg)
            docs_splitted = self.docs_splitted(docs)
            msg = f'Total chunks to load: {len(docs_splitted)}'
            self.logger.info(msg)
            vectorstore = OpenSearchVectorSearch.from_documents(
                docs_splitted,
                self.embeddings,
                index_name=self.settings.os_index,
                opensearch_url=self.settings.db_hosts,
                http_auth=(self.settings.db_user, self.settings.db_password),
                use_ssl=True,
                verify_certs=True,
                ca_certs=self.settings.ca,
                engine='lucene',
                bulk_size=self.settings.bulk_size
            )
            msg = f'Vectorstore loaded with documents'
            self.logger.info(msg)
        else:
            vectorstore = OpenSearchVectorSearch(
                embedding_function=self.embeddings,
                index_name=self.settings.os_index, 
                opensearch_url=self.settings.db_hosts,
                http_auth=(self.settings.db_user, self.settings.db_password),
                use_ssl=True,
                verify_certs=True,
                ca_certs=self.settings.ca,
                engine='lucene'
            )
            msg = f'Vectorstore initialized, pre-loaded index {self.settings.os_index}'
            self.logger.info(msg)
        return vectorstore

    def db_simularity_search(self, query, k_max):
        query_docs = self.vectorstore.similarity_search(
            query, 
            k=k_max
        )
        return query_docs

    def rag_chain(self, k_max, temperature):
        self.llm = YandexGPT(
            model_name=self.settings.model_name,
            api_key=self.settings.secret_key,
            folder_id=self.settings.folder_id,
            temperature=temperature
        )
        retriever = self.vectorstore.as_retriever(
            search_type='similarity', 
            search_kwargs={'k': k_max}
        )
        contextualize_q_system_prompt = ("""
        Учитывая историю чата и последний вопрос пользователя,
        который может ссылаться на контекст в истории чата,
        сформулируй отдельный вопрос, который можно понять
        без истории чата. Не отвечай на вопрос, просто
        при необходимости переформулируй его, а в противном 
        случае верни как есть.
        """)
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ('system', contextualize_q_system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}'),
        ])
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )
        
        qa_system_prompt = ("""
        Ты являешься поисковым ассистентом и должен искать информацию 
        в приложенных документах. Если информация не найдена, то отвечай,
        что информация в документах не содержится. Отвечай на вопрос, 
        используя информацию из текста ниже.
        
        Текст:
        -----
        {context}
        -----
        """)
        qa_prompt = ChatPromptTemplate.from_messages([
            ('system', qa_system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}')
        ])
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )
        return rag_chain


LOGGER, QL = book_logger_qh(
    settings=SETTINGS, 
    file_name='server.log', 
    name=__name__
)
QL.start()
msg = 'Search server started, logger initialized'
LOGGER.info(msg)
BOOK_SEARCHER = BookSearcher(SETTINGS, LOGGER)
msg = 'BookSearcher instance created, documents uploaded to database'
LOGGER.info(msg)

app = FastAPI()
if not SETTINGS.no_auth:
    app.add_middleware(
        BaseHTTPMiddleware, 
        dispatch=authorization_middleware
    )


class InitParams(BaseModel):
    k_max: int
    temperature: float


class SearchParams(BaseModel):
    k_max: int
    query: str


class AskParams(BaseModel):
    query: str


@app.get('/datainfo')
async def data_config():
    data = {
        'bucket': SETTINGS.bucket, 
        'bucket_info': SETTINGS.bucket_info,
        'bucket_prefix': SETTINGS.bucket_prefix,
        'endpoint_url': SETTINGS.endpoint_url
    }
    return data


@app.get('/creds')
async def server_config():
    return {'data': list(SETTINGS.model_fields_set)}


@app.get('/logs')
async def server_logs():
    with open(f'{SETTINGS.logs_path}/server.log') as file:
        logs = file.readlines()
    return {'data': logs}


@app.post('/init')
async def init_chain(initparams: InitParams = Body(...)):
    k_max = initparams.k_max
    temperature = initparams.temperature
    global RAG_CHAIN
    RAG_CHAIN = BOOK_SEARCHER.rag_chain(k_max, temperature)
    msg = 'RAG chain for k-max={} done, temperature={}'.format(
        str(k_max),
        str(temperature)
    )
    LOGGER.info(msg)
    return {'result': msg}


@app.post('/search')
async def search_db(searchparams: SearchParams = Body(...)):
    k_max = searchparams.k_max
    query = searchparams.query
    docs = BOOK_SEARCHER.db_simularity_search(query=query, k_max=k_max)
    msg = 'Search for query -{}- done, k_max={}'.format(
        query,
        k_max
    )
    LOGGER.info(msg)
    return {'result': msg, 'data': docs}

    
@app.post('/ask')
async def ask_chain(askparams: AskParams = Body(...)):
    query = askparams.query
    chat_history = []  # if needed
    response = RAG_CHAIN.invoke({
        'input': query, 
        'chat_history': chat_history
    })
    msg = 'Q - {} | A - {} symbols, {} context documents'.format(
        query,
        len(response['answer']),
        len(response['context'])
    )
    LOGGER.info(msg)
    return {'result': msg, 'answer': response}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=SETTINGS.server_port)
