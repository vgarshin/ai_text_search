import os
import queue
import logging
from pathlib import Path

FORMATTER = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')


def bot_logger_qh(settings, file_name, name):
    os.makedirs(settings.logs_path, exist_ok=True)
    log_path = Path(settings.logs_path)
    log_file = log_path.joinpath(file_name)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_queue = queue.Queue()
    qh = logging.handlers.QueueHandler(log_queue)  
    qh.setFormatter(FORMATTER)
    logger.addHandler(qh)           
    rh = logging.handlers.RotatingFileHandler(log_file)
    queue_listener = logging.handlers.QueueListener(log_queue, rh)
    return logger, queue_listener
