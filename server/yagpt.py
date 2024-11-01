#!/usr/bin/env python
# coding: utf-8

import os
import time
import logging
import requests
from typing import Any, List, Mapping, Optional
import langchain
from langchain.embeddings.base import Embeddings
from langchain.callbacks.manager import CallbackManagerForLLMRun

LOG_PATH = 'logs'
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
fh = logging.FileHandler(f'{LOG_PATH}/yagpt.log')
LOGGER.addHandler(fh)
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
fh.setFormatter(formatter)


class YandexGPTEmbeddings(Embeddings):
    def __init__(self, iam_token=None, api_key=None, folder_id=None, sleep_interval=1):
        self.iam_token = iam_token
        self.sleep_interval = sleep_interval
        self.api_key = api_key
        self.folder_id = folder_id
        if self.iam_token:
            self.headers = {'Authorization': 'Bearer ' + self.iam_token}
        if self.api_key:
            self.headers = {
                'Authorization': 'Api-key ' + self.api_key,
                "x-folder-id" : self.folder_id 
            }
                
    def embed_document(self, text):
        j = {
            'modelUri': f'emb://{self.folder_id}/text-search-doc/latest',
            'text': text
        }
        res = requests.post(
            'https://llm.api.cloud.yandex.net:443/foundationModels/v1/textEmbedding',
            json=j, 
            headers=self.headers
        )
        vec = res.json()['embedding']
        return vec

    def embed_documents(self, texts, chunk_size=0):
        res = []
        for x in texts:
            res.append(self.embed_document(x))
            time.sleep(self.sleep_interval)
        return res
        
    def embed_query(self, text):
        j = {
            'modelUri': f'emb://{self.folder_id}/text-search-query/latest',
            'text': text
        }
        res = requests.post(
            'https://llm.api.cloud.yandex.net:443/foundationModels/v1/textEmbedding',
            json=j, 
            headers=self.headers
        )
        vec = res.json()['embedding']
        time.sleep(self.sleep_interval)
        return vec
    

class YandexLLM(langchain.llms.base.LLM):
    api_key: str = None
    iam_token: str = None
    folder_id: str = None
    max_tokens: int = 1500
    temperature: float = 1
    instruction_text: str = None

    @property
    def _llm_type(self) -> str:
        return 'yagpt'

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError('Stop kwargs are not permitted.')
        headers = {
            'x-folder-id': self.folder_id,
            'Content-type': 'application/json'
        }
        if self.iam_token:
            headers['Authorization'] = f'Bearer {self.iam_token}'
        if self.api_key:
            headers['Authorization'] = f'Api-key {self.api_key}'
        req = {
            'modelUri': f'gpt://{self.folder_id}/yandexgpt/latest',
            'completionOptions': {
                'stream': False,
                'temperature': self.temperature,
                'maxTokens':  self.max_tokens
            },
            'messages': [
                {
                    'role': 'system',
                    'text': self.instruction_text
                },
                {
                    'role': 'user',
                    'text': prompt
                }
            ]
        }
        res = requests.post(
            'https://llm.api.cloud.yandex.net/foundationModels/v1/completion',
            headers=headers, 
            json=req
        ).json()
        return res['result']['alternatives'][0]['message']['text']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        Get the identifying parameters.
        
        """
        return {'max_tokens': self.max_tokens, 'temperature': self.temperature }
