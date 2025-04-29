FROM cr.yandex/crptfs8246iv37ojkorv/simbadsai:20250404v0

USER root

RUN apt update && apt install curl -y

USER $NB_UID

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt