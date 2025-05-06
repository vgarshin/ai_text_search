#!/usr/bin/env python
# coding: utf-8

import io
import sys
import jwt
import json
import requests
import datetime
import streamlit as st
from io import StringIO

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
    'Authorization': token
}

st.set_page_config(
    page_title='Загрузка файлов',
    page_icon=':inbox_tray:'
)
st.sidebar.header('Загрузка файлов')
st.header(
    'Добавление новых файлов в базу данных',
    divider='rainbow'
)

st.write('#### Загрузите файл')
upload_folder = st.text_input(
    'Введите имя директории, куда будет сохранен файл (по умолчанию используется директория `external`)',
    'external'
)
uploaded_file = st.file_uploader('Выберите файл для загрузки (поддерживаются форматы doc, docx, txt, rtf, pdf, xls, xlsx, ppt)')
if uploaded_file is not None:
    file_name = uploaded_file.name
    with st.spinner('Загрузка...'):
        bytes_data = uploaded_file.getvalue()
        files = {'file': (file_name, io.BytesIO(bytes_data))}
        r = requests.post(
            URL_SERVER + f'/upload/{upload_folder}',
            headers=HEADERS,
            files=files,
            verify=True
        )
    if r.status_code == 200:
        st.write('#### Обновление индекса')
        st.write(
            """
            Файл загружен, но для использования его в качестве источника 
            данных и поиска в нем сведений при помощи AI-ассистента 
            необходимо перестроить посковый индекс базы данных, в которой
            хранятся текстовые данные для поиска.
            Для пересчета индекса введите имя нового индекса и нажмите 
            кнопку `Rebuild index` внизу, но учтите, что при большом объеме 
            текстовых данных пересчет индекса может занять длительное время 
            (до 30 минут).
            """
        )
        postfix = ''.join(x for x in str(datetime.datetime.now()) if x.isdigit())
        rebuild_index = st.text_input(
            'Введите имя индекса, который будет использоваться в базе данных',
            '-'.join([SETTINGS.os_index, postfix])
        )
        if st.button('Пересчитать индекс', type='primary'):
            with st.spinner('Идет пересчет...'):
                data = {'rebuild_index': rebuild_index}
                r = requests.post(
                    URL_SERVER + '/rebuild',
                    data=json.dumps(data),
                    headers=HEADERS,
                    verify=True
                )
            if r.status_code == 200:
                st.success('Пересчет индекса завершен', icon='✅')
    else:
        st.error(
            'Ошибка загрузки файла, попробуйте еще раз', 
            icon='⚠️'
        )
