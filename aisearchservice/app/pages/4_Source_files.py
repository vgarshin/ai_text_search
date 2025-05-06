#!/usr/bin/env python
# coding: utf-8

import sys
import jwt
import json
import requests
import streamlit as st
from io import StringIO
import pandas as pd

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

st.set_page_config(
    page_title='Источники данных',
    page_icon=':books:'
)
st.sidebar.header('Источники данных')
st.header(
    'Файлы в базе данных',
    divider='rainbow'
)

r = requests.get(
    URL_SERVER + '/sources',
    headers=HEADERS,
    verify=True
)
if r.status_code == 200:
    rag_files = r.json()['data']
    files = []
    for rf in rag_files:
        prefix = rf.split('/')[0]
        path = ''.join(rf.split('/')[1:-1])
        name = rf.split('/')[-1]
        files.append([
            prefix,
            path,
            name
        ])
    df = pd.DataFrame(files, columns=("Bucket", "Path", "File name"))
    st.dataframe(df)
    st.divider()
else:
    st.error(
        'Ошибка загрузки логов системы', 
        icon='⚠️'
    )
