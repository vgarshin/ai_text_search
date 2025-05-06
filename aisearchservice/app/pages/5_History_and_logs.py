#!/usr/bin/env python
# coding: utf-8

import sys
import jwt
import json
import requests
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
    'Content-type': 'application/json',
    'Authorization': token
}

st.set_page_config(
    page_title='Логи чатов',
    page_icon=':gear:'
)
st.sidebar.header('Логи чатов')
st.header(
    'История общения с чат-ботами',
    divider='rainbow'
)

r = requests.get(
    URL_SERVER + '/logs',
    headers=HEADERS,
    verify=True
)
if r.status_code == 200:
    logs = r.json()['data']
    st.write('\n'.join(logs))
    st.divider()
else:
    st.error(
        'Ошибка загрузки логов системы', 
        icon='⚠️'
    )
