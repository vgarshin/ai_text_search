#!/usr/bin/env python
# coding: utf-8

import os
import sys
import jwt
import json
import requests
import datetime
import pandas as pd
import streamlit as st

sys.path.append('/home/jovyan/aisearchservice')
from server.config import SETTINGS

URL_SERVER = 'http://{}:{}'.format(SETTINGS.ip, SETTINGS.server_port)
JWT_ALGORITHM = 'HS256'
payload = {
    'jwt_secret': SETTINGS.jwt_secret
}
token = jwt.encode(
    payload, 
    SETTINGS.jwt_secret, 
    algorithm=JWT_ALGORITHM
)
HEADERS = {
    'Content-type': 'application/json',
    'Authorization': token
}

r = requests.get(
    URL_SERVER + '/datainfo',
    headers=HEADERS,
    verify=True
)

st.set_page_config(
    page_title='Кедровая Падь: летописи природы',
    page_icon=':speech_balloon:'
)
st.sidebar.header('Чат-бот для интеллектуального поиска')
st.header('AI-бот для поиска и систематизации сведений из летописей природы заповедника', divider='rainbow')
st.markdown(
    """
    Для начала работы вам необходимо ввести ряд параметров  
    для чат-бота, а потом задать предметную область в виде 
    запроса по интересующей теме. 
    После этого вы сможете задавать чат-боту интересующие вас 
    запросы в формате диалога. Чат-бот при этом будет учитывать 
    в разговоре наиболее релевантные материалы и сведения из
    оцифрованных летописей природы.
    """
)
st.markdown(f'Источник (бакет): {SETTINGS.bucket_info}')
st.markdown(f'Источник (префикс): {SETTINGS.bucket_prefix}')
st.markdown(f'ID бакета: {SETTINGS.bucket}')

st.write('#### Задайте инструкцию')
default_instruction = """Вы являетесь поисковым ассистентом.
Ваша задача состоит в поиске и систематизации сведений из 
летописей природы, составленных в ходе наблюдения за природным
заповедником 'Кедровая Падь'."""
instruction = st.text_area(
    'Введите инструкцию для чат-бота',
    default_instruction
)

st.write('#### Температура чат-бота')
st.write(
    """
    Чем выше значение этого параметра, тем более креативными
    и случайными будут ответы модели. Принимает значения
    от 0 (включительно) до 1 (включительно).
    Значение по умолчанию: 0 (без креатива)
    """
)
temperature = st.slider("Введите температуру для чат-бота", .0, 1., .0, .1)

st.write('#### Введите количество релевантных документов')
st.write(
    """
    Необходимо указать максимальное количество документов внутри 
    одного поиска, чтобы ограничить область поиска для чат-бота.
    Значение по умолчанию: 2 документа
    """
)
k_max = st.slider('Задайте количество документов', 1, 5, 3)

if 'chat_id' not in st.session_state:
    st.session_state.chat_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S%s')

data = {'instruction': instruction, 'k_max': k_max, 'temperature': temperature}
with st.spinner('Запуск чат-бота...'):
    r = requests.post(
        URL_SERVER + '/init',
        data=json.dumps(data),
        headers=HEADERS,
        verify=True
    )
if r.status_code == 200:
    st.write('#### Задавайте вопросы чат-боту')
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    if query := st.chat_input('Введите ваше сообщение'):
        st.chat_message('user').markdown(query)
        st.session_state.messages.append(
            {
                'role': 'user',
                'content': query
            }
        )
        data = {'query': query, 'chat_id': st.session_state.chat_id}
        r = requests.post(
            URL_SERVER + '/ask',
            data=json.dumps(data),
            headers=HEADERS,
            verify=True
        )
        sources = ''
        for c in r.json()['answer']['context']:
            sources += ('\n - ' + ', '.join([
                'Наименование: ' + c['metadata']['title'], 
                'Период: ' + c['metadata']['period'], 
                'Файл: ' + c['metadata']['source'].split('/')[-1]
            ]))
        sources_info = ('\n\nИсточники:' + sources) if sources else ''
        answer = r.json()['answer']['answer'] + sources_info
        with st.chat_message('assistant'):
            st.markdown(answer)
        st.session_state.messages.append(
            {
                'role': 'assistant',
                'content': answer
            }
        )
else:
    st.error(
        'Ошибка запуска ассистента, попробуйте поменять параметры', 
        icon='⚠️'
    )
