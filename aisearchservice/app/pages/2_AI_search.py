#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import requests
import pandas as pd
import streamlit as st

sys.path.append('../../')
from server.config import SETTINGS

URL_SERVER = 'http://{}:{}'.format(SETTINGS.ip, SETTINGS.server_port)
HEADER = {'Content-type': 'application/json'}



r = requests.get(
    URL_SERVER + '/datainfo',
    headers=HEADER,
    verify=True
)

st.set_page_config(
    page_title='Поиск по летописям природ заповедника Кедровая Падь',
    page_icon=':speech_balloon:'
)
st.sidebar.header('Поиск по векторной базе')
st.header('AI-поиск с использованием языковой модели по данным из летописей природы заповедника', divider='rainbow')
st.markdown(
    """
    Введите свой запрос и количество документов, которые
    будут выведены в качестве результатов. 
    Поиск ведется по отдельным блокам, на которые были разделены
    первоначальные документы, чтобы облегчить поиск и находить
    наиболее релевантные данные. Вы можете задать количество 
    блоков, которые попадут в итоговый результат.
    """
)
st.markdown(f'Источник (бакет): {SETTINGS.bucket_info}')
st.markdown(f'Источник (префикс): {SETTINGS.bucket_prefix}')
st.markdown(f'ID бакета: {SETTINGS.bucket}')

st.write('#### Введите количество релевантных документов')
st.write(
    """
    Необходимо указать максимальное количество документов внутри 
    одного поиска, чтобы ограничить область поиска для чат-бота.
    Значение по умолчанию: 2 документа
    """
)
k_max = st.slider('Задайте количество документов', 1, 10, 3)

st.write(
    """
    Введите ваш запрос для поиска релевантной информации
    в материалах
    """
)
query = st.text_area(
    'Ваш запрос',
    'Количество зафиксированных особей леопардов в заповеднике в 2013 году'
)
data = {'query': query, 'k_max': k_max}
with st.spinner('Поиск документов...'):
    r = requests.post(
        URL_SERVER + '/search',
        data=json.dumps(data),
        headers=HEADER,
        verify=True
    )
if r.status_code == 200:
    for d in r.json()['data']:
        st.write('## Источник')
        source = ('\n - ' + ', '.join([
            'Наименование: ' + d['metadata']['title'], 
            'Период: ' + d['metadata']['period'], 
            'Файл: ' + d['metadata']['source'].split('/')[-1]
        ]))
        st.write(source)
        st.write(d['page_content'])
else:
    st.error(
        'Ошибка запуска ассистента, попробуйте поменять параметры', 
        icon=':warning:'
    )
