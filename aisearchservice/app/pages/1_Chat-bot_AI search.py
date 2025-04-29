#!/usr/bin/env python
# coding: utf-8

import os
import json
import requests
import pandas as pd
import streamlit as st


def read_json(file_path):
    with open(file_path) as file:
        access_data = json.load(file)
    return access_data


APP_CONFIG = read_json(file_path='configs/config.json')
PORT = APP_CONFIG['port']
IP = APP_CONFIG['ip']
URL_SERVER = 'http://{}:{}'.format(IP, PORT)
HEADER = {'Content-type': 'application/json'}

r = requests.get(
    URL_SERVER + '/datainfo',
    headers=HEADER,
    verify=True
)
course_info = r.json()['data']

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
st.markdown(f'Курс: {course_info["bucket_info"]}')

st.write('#### Задайте инструкцию')
default_instructions = """Вы являетесь поисковым ассистентом.
Ваша задача состоит в поиске и систематизации сведений из 
летописей природы, составленных в ходе наблюдения за природным
заповедником 'Кедровая Падь'."""
instructions = st.text_area(
    'Введите инструкцию для чат-бота',
    default_instructions
)

st.write('#### Температура чат-бота')
st.write(
    """
    Чем выше значение этого параметра, тем более креативными
    и случайными будут ответы модели. Принимает значения
    от 0 (включительно) до 1 (включительно).
    Значение по умолчанию: 0.3
    """
)
temperature = st.slider("Введите температуру для чат-бота", .0, 1., .3, .1)

if instructions:
    data = {'instructions': instructions, 'temperature': temperature}
    with st.spinner('Запуск чат-бота...'):
        r = requests.post(
            URL_SERVER + '/init',
            data=json.dumps(data),
            headers=HEADER,
            verify=True
        )
    if r.status_code == 200:
        st.write('#### Введите запрос')
        st.write(
            """
            Сначала необходимо указать максимальное количество документов 
            внутри одного поиска, чтобы ограничить область поиска для чат-бота.
            Значение по умолчанию: 2 документа
            """
        )
        k = st.slider('Задайте количество документов', 1, 5, 2)
        st.write(
            """
            Укажите ваш запрос для поиска релевантной информации
            в материалах курса
            """
        )
        query = st.text_area(
            'Ваш запрос',
            'Среднегодовая температура за последние несколько лет'
        )
        data = {'query': query, 'k': k}
        with st.spinner('Поиск документов...'):
            r = requests.post(
                URL_SERVER + '/search',
                data=json.dumps(data),
                headers=HEADER,
                verify=True
            )
        if r.status_code == 200:
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
                data = {'query': query}
                r = requests.post(
                    URL_SERVER + '/ask',
                    data=json.dumps(data),
                    headers=HEADER,
                    verify=True
                )
                response = r.json()['answer']
                with st.chat_message('assistant'):
                    st.markdown(response)
                st.session_state.messages.append(
                    {
                        'role': 'assistant',
                        'content': response
                    }
                )
        else:
            sst.error(
                'Ошибка запуска ассистента, попробуйте поменять параметры', 
                icon=':warning:'
            )
