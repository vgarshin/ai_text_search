import json
from pydantic import BaseModel


class Settings(BaseModel):
    bucket: str
    bucket_prefix: str
    bucket_info: str
    aws_access_key_id: str
    aws_secret_access_key: str
    endpoint_url: str
    folder_id: str
    db_user: str 
    db_password: str 
    db_hosts: list
    chunk_size: int
    chunk_overlap: int
    new_index: bool
    os_index: str
    ca: str
    bulk_size: int
    k_max: int
    score_threshold: float
    model_name: str
    temperature: float
    max_tokens: int
    sleep_emb: float
    service_account_id: str 
    key_id: str
    secret_key: str
    jwt_secret: str
    logs_path: str
    server_port: int
    app_port: int
    ip: str
    no_auth: bool
    tg_token: str


def load_config(path: str) -> Settings:
    with open(path, 'r') as fp:
        data = json.load(fp)
    return Settings(**data)


SETTINGS = load_config('aisearchservice/config.json')
