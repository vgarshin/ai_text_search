# GSOM YaGPT bot for an AI search
Experimental AI bot based on [Yandex repository for Retrieval-Augmented Generation](https://github.com/yandex-cloud-examples/yc-yandexgpt-qa-bot-for-docs) with help of Yandex.Cloud services.

## Goal

Run a simple AI bot for a search within text documents based on YandexGPT LLM and OpenSearch database to store context in a vector form (embeddings).

## Tools

[LangChain](https://python.langchain.com) framework to operate with the data and run chat-bot, [Streamlit](https://streamlit.io/) for the interface of the application, [Flask](https://flask-docs.readthedocs.io/en/latest/) for the backend, [Docker](https://www.docker.com/) to run a services.

## Prerequisites

Data on a course course unpacked to object storage bucket. Data can contain text (txt, doc, docx, pdf) and presentations (ppt).

[Object Storage](https://yandex.cloud/en/docs/storage/quickstart) bucket for the course data, [Yandex Managed Service for OpenSearch](https://cloud.yandex.com/en/docs/managed-opensearch/) instance as a database. [Virtual machine](https://cloud.yandex.com/en/docs/compute/quickstart/) with Docker installed.

Credentials to access object storage bucket, database and YandexGPT API in a form of JSON files `configs/config_SAMPLE.json` and `configs/credentials_SAMPLE.json`.

## Manual

Start service for AI search assistant with `docker compose` to run all applications at once:

```bash
docker compose up
```
