import json


def settings(file_path):
    with open(file_path) as file:
        settings = json.load(file)
    return settings


SETTINGS = settings(file_path='../config.json')
