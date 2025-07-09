#!/usr/bin/env python
# coding: utf-8

import os
import jwt
import json
import time
import uuid
import boto3
import requests
import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    CallbackQueryHandler
)
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
)

from config import SETTINGS
from logger import bot_logger_qh

LOGGER, QL = bot_logger_qh(
    settings=SETTINGS, 
    file_name='bot.log', 
    name=__name__
)
QL.start()
msg = 'Bot logger initialized'
LOGGER.info(msg)

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

RAGAS_DATA_POOL = []

S3_SESSION = boto3.session.Session()
S3_CLIENT = S3_SESSION.client(
    service_name='s3',
    aws_access_key_id=SETTINGS.aws_access_key_id,
    aws_secret_access_key=SETTINGS.aws_secret_access_key,
    endpoint_url=SETTINGS.endpoint_url
)


def create_presigned_url(bucket_name, object_name, expiration=3600):
    """
    Generates a presigned URL to share an S3 object.

    :bucket_name: string
    :object_name: string
    :expiration: time in seconds for the presigned URL to remain valid
    
    Returns presigned URL as string or `None` if error.
    
    """
    try:
        response = S3_CLIENT.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': object_name},
            ExpiresIn=expiration,
        )
    except ClientError as e:
        msg = f'create presigned url for source `{object_name}` error `{e}`'
        LOGGER.error(msg)
        return None
    return response


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    chat_id = update.effective_chat.id
    if 'interactions' in context.chat_data:
        context.chat_data['interactions'].clear()
    intro_text = (
        'Меня зовут SmartLeo, я могу отвечать на разные вопросы о заповеднике `Кедровая Падь`.\n'
        'Задайте ваш вопрос - я отвечу на него с учетом информации, которая есть в летописях природы заповедника.'
    )
    await update.message.reply_html(intro_text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        'Вас приветствует AI-бот для поиска и систематизации сведений из летописей природы заповедника `Кедровая Падь`.\n\n'
        'Как работает этот чат-бот?\n'
        '- в него уже загружены оцифрованные материалы (летописи природы с конца 1970-х годов)\n'
        '- чат-бот может искать ответы на вопросы в имеющихся материалах\n'
        '- чат-бот помнит историю общения и может ссылаться на источники\n'
        'Задавайте интересующие вас вопросы в формате диалога.\n\n'
        'Команды:\n'
        '/start - запустить или перезапустить диалог.\n'
        '/help - показать это сообщение.'
    )
    await update.message.reply_text(help_text)


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    username = update.effective_chat.username
    query = update.message.text
    processing_msg = await update.message.reply_text('Думаю над вопросом, пожалуйста подождите...')
    try:
        # Get data from RAG
        data = {'query': query, 'chat_id': str(chat_id)}
        r = requests.post(
            URL_SERVER + '/ask',
            data=json.dumps(data),
            headers=HEADERS,
            verify=True
        )
        contexts_docs = r.json()['answer']['context']
        await context.bot.delete_message(chat_id=chat_id, message_id=processing_msg.message_id)
        answer = r.json()['answer']['answer']
        answer_message = await update.message.reply_text(answer)
        msg = '{}: Q - {} | bot: A - {} symbols, {} context documents'.format(
            username,
            query,
            len(answer),
            len(contexts_docs)
        )
        LOGGER.info(msg)
        
        # Chat context
        interaction_id = str(uuid.uuid4())
        if 'interactions' not in context.chat_data: context.chat_data['interactions'] = {}
        context.chat_data['interactions'][interaction_id] = {
            'question': query, 
            'answer': answer, 
            'contexts_docs': contexts_docs,
            'answer_message_id': answer_message.message_id, 
            'feedback': None,
            'timestamp': datetime.datetime.now().isoformat()
        }

        # Feedback and sources buttons set
        buttons = []
        if contexts_docs:
            buttons.append(InlineKeyboardButton(
                'Источники 📄', 
                callback_data=f'sources_{interaction_id}'
            ))
        buttons.extend([
            InlineKeyboardButton("👍", callback_data=f"feedback_positive_{interaction_id}"),
            InlineKeyboardButton("👎", callback_data=f"feedback_negative_{interaction_id}")
        ])        
        if buttons:
            reply_markup = InlineKeyboardMarkup([buttons])
            await update.message.reply_text(
                'Этот ответ был полезен? Можно также посмотреть источники.', 
                reply_markup=reply_markup
            )

        # RAGAS data collection
        RAGAS_DATA_POOL.append({
            'interaction_id': interaction_id, 
            'question': query,
            'answer': answer,
            'contexts': [c['page_content'] for c in contexts_docs],
            'retrieved_document_sources_keys': [c['metadata']['source'] for c in contexts_docs],
            'chat_id': chat_id, 
            'timestamp': context.chat_data['interactions'][interaction_id]['timestamp'],
            'feedback': None
        })
    except Exception as e:
        msg = f'handling message `{query}` error {e}'
        LOGGER.error(msg, exc_info=True)
        try:
            await context.bot.edit_message_text(
                'Внутренняя ошибка, попробуйте еще раз.', 
                chat_id=chat_id,
                message_id=processing_msg.message_id
            )
        except:  # if editing fails (e.g. message too old or deleted)
            await update.message.reply_text('Внутренняя ошибка, попробуйте еще раз.')
        
        # log error for RAGAS
        RAGAS_DATA_POOL.append({
            'question': query,
            'answer': f'Error: {e}',
            'contexts': [],
            'chat_id': chat_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'error': str(e)
        })


async def sources_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    try:
        _, interaction_id = query.data.split('_', 1)
    except ValueError:
        msg = f'sources callback data `{query.data}` error'
        LOGGER.warning(msg)
        await query.edit_message_text(text='Ошибка, невозможно обработать запрос.')
        return

    interaction_data = context.chat_data.get('interactions', {}).get(interaction_id)
    if not interaction_data or not interaction_data.get('contexts_docs'):
        await query.edit_message_text(
            text='Документы недоступны.'
        )
        return

    sources = []
    for c in interaction_data['contexts_docs']:
        if 'metadata' in c.keys():
            bucket_name = c['metadata']['source'].replace('s3://', '').split('/')[0]
            object_name = c['metadata']['source'].replace('s3://', '').replace(bucket_name, '')[1:]
            url = create_presigned_url(
                bucket_name=bucket_name,
                object_name=object_name,
                expiration=600
            )
            sources.append(', '.join([
                '<u>Название:</u> ' + c['metadata'].get('title', 'Нет названия'),
                '<u>Период:</u> ' + c['metadata'].get('period', 'Неизвестный период'),
                f"<a href='{url}'>ссылка на источник</a>"
            ]))
    if sources:
        sources_display = '<b>Источники:</b>\n'
        sources_display += '\n'.join(
            [f'{i + 1}. {s}' for i, s in enumerate(sources)]
        )
        if len(sources_display) > 2048: 
            sources_display = sources_display[:2048] + '\n...(список слишком длинный)'
    else:
        sources_display = 'Нет нужных источников.'
    
    try:
        await query.edit_message_text(
            text=sources_display,
            parse_mode='HTML', 
            disable_web_page_preview=True, 
            reply_markup=None
        )
    except Exception as e:
        msg = f'failed to display sources, error `{e}`'
        LOGGER.error(msg)
        answer_msg_id = interaction_data.get('answer_message_id')
        await context.bot.send_message(
            chat_id=query.message.chat_id, 
            text=final_text,
            reply_to_message_id=answer_msg_id if answer_msg_id else None,
            parse_mode='HTML',
            disable_web_page_preview=True
        )


async def feedback_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    try:
        _, sentiment, interaction_id = query.data.split('_')
    except ValueError:
        msg = f'feedback callback data `{query.data}` error'
        LOGGER.warning(msg)
        await query.edit_message_text(text='Ошибка обработки обратной связи.')
        return

    feedback_recorded = False
    if 'interactions' in context.chat_data and interaction_id in context.chat_data['interactions']:
        context.chat_data['interactions'][interaction_id]['feedback'] = sentiment
        feedback_recorded = True
        for item in RAGAS_DATA_POOL:
            if item.get('interaction_id') == interaction_id:
                item['feedback'] = sentiment
                break
    
    if sentiment == 'positive':
        confirm_text = 'Спасибо за отзыв! 👍'
    else:
        confirm_text = 'Спасибо за отзыв, мы обязательно учтём его. 👎'
    if not feedback_recorded: confirm_text = 'Ошибка, отзыв не может быть учтен.'

    new_buttons_row = []
    if query.message.reply_markup:
        for row in query.message.reply_markup.inline_keyboard:
            new_buttons_row.extend(
                b for b in row 
                if b.callback_data and b.callback_data.startswith('sources_')
            )

    new_reply_markup = InlineKeyboardMarkup([new_buttons_row]) if new_buttons_row else None
    try:
        await query.edit_message_text(
            text=f'{query.message.text}\n\n_{confirm_text}_',
            reply_markup=new_reply_markup,
            parse_mode='Markdown'
        )
    except Exception as e:
        msg = f'editing message for feedback error `{e}`, sending new message'
        LOGGER.warning(msg)
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=confirm_text
        )


def telegram_bot_runner(tg_token, logger):
    if not tg_token:
        msg = 'Telegram token to access API not found'
        LOGGER.critical(msg)
        return
    application = Application.builder().token(tg_token).build()
    application.add_handler(CommandHandler('start', start_command))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(CallbackQueryHandler(sources_callback, pattern='^sources_'))
    application.add_handler(CallbackQueryHandler(feedback_callback, pattern='^feedback_'))
    msg = 'Telegram bot is starting...'
    LOGGER.info(msg)
    application.run_polling(close_loop=False, drop_pending_updates=True)


if __name__ == '__main__':
    telegram_bot_runner(SETTINGS.tg_token, LOGGER)
