#!/usr/bin/env python
# coding: utf-8

import json
import requests
import streamlit as st
from io import StringIO


def read_json(file_path):
    with open(file_path) as file:
        access_data = json.load(file)
    return access_data


APP_CONFIG = read_json(file_path='configs/config.json')
PORT = APP_CONFIG['port']
IP = APP_CONFIG['ip']
URL_SERVER = 'http://{}:{}'.format(IP, PORT)
HEADER = {'Content-type': 'application/json'}

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
    headers=HEADER,
    verify=True
)
logs = r.json()['data']

st.write('\n'.join(logs))
st.divider()
