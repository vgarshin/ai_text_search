#!/usr/bin/env python
# coding: utf-8

import os
import time
import glob
import json
import base64
import requests
import numpy as np
from tqdm.auto import tqdm
from pdf2image import convert_from_path
from PIL import Image

SLEEP = 2


def json_data(file_path):
    with open(file_path) as file:
        data = json.load(file)
    return data


def ask_llm(model_name, prompt, instruction_text, 
            folder_id, api_key, temperature, max_tokens):
    headers = {
        'x-folder-id': folder_id,
        'Content-type': 'application/json'
    }
    headers['Authorization'] = f'Api-key {api_key}'
    req = {
        'modelUri': f'gpt://{folder_id}/{model_name}/latest',
        'completionOptions': {
            'stream': False,
            'temperature': temperature,
            'maxTokens':  max_tokens
        },
        'messages': [
            {
                'role': 'system',
                'text': instruction_text
            },
            {
                'role': 'user',
                'text': prompt
            }
        ]
    }
    flag = True
    while flag:
        try:
            res = requests.post(
                'https://llm.api.cloud.yandex.net/foundationModels/v1/completion',
                headers=headers, 
                json=req
            )
            text = res.json()['result']['alternatives'][0]['message']['text']
            flag = False
        except Exception as e:
            print('Error:', e, '!', res)
            time.sleep(SLEEP)
    return text


def text_from_jsons(pdf_file_path, pdfs_path, raw_ocr_path,
                    tables_flag=False, pages_flag=False,
                    instruction_text=None, folder_id=None, 
                    api_key=None, temperature=None, max_tokens=None):
    text = ''
    if '.pdf' in pdf_file_path:
        rawocr_dir = pdf_file_path.replace(
            pdfs_path,
            raw_ocr_path
        ).replace(
            '.pdf', ''
        )
        
        # read json files and make a text
        file_name = pdf_file_path.split('/')[-1]
        json_files = glob.glob(f'{rawocr_dir}/*.json', recursive=True)
        for n_page, json_file in enumerate(tqdm(json_files, desc=file_name)):
            d = json_data(json_file)
            
            # block to print out tables to text
            tables_text = ''
            if tables_flag:
                tables = d['result']['textAnnotation']['tables']
                if tables:
                    for tbl in tables:
                        rows = int(tbl['rowCount'])
                        cols = int(tbl['columnCount'])
                        arr = np.empty(shape=[rows, cols], dtype=object)
                        for cell in tbl['cells']:
                            arr[int(cell['rowIndex']), int(cell['columnIndex'])] = cell['text']
                        table_text = ''
                        for row in arr:
                            row_txt = '|'
                            for col in row:
                                row_txt += (col.replace('\n', ' ') + '|' if col else '')
                            table_text += (row_txt + '\n')
                        if instruction_text and table_text:
                            table_text = ask_llm(table_text, instruction_text, folder_id, 
                                                 api_key, temperature, max_tokens)
                        tables_text += ('\n\n' + table_text)

            # block to process plain text
            text_page = d['result']['textAnnotation']['fullText']
            if instruction_text and text_page:
                text_page = ask_llm(text_page, instruction_text, folder_id,
                                    api_key, temperature, max_tokens)
            text = '{}\n{}{}\n{}'.format(
                text,
                f'\npage {str(n_page + 1)}\n\n' if pages_flag else '',
                text_page,
                tables_text if tables_text else ''
            )
    else:
        print('file skiped:', pdf_file)
    return text


def save_text(text, save_path):
    with open(save_path, 'w') as file:
        file.write(text)
