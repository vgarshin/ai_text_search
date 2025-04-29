import os
import queue
import logging
from pathlib import Path

FORMATTER = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')


def book_logger_fh(settings, name):
    log_path = Path(settings.assignments_path)
    log_file = log_path.joinpath('otter.log')
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    logger.addHandler(fh)
    fh.setFormatter(formatter)
    return logger


def book_logger_qh(settings, file_name, name):
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


"""
import asyncio
import logging.handlers
from multiprocessing import Queue

logfile = "log.txt"

logger = logging.getLogger()

loghandler = logging.handlers.RotatingFileHandler(
    logfile, maxBytes=102400, backupCount=5
)
formatter = logging.Formatter(
    "%(filename)s [LINE:%(lineno)d] #%(levelname)-8s [%(asctime)s]  %(message)s"
)
loghandler.setFormatter(formatter)

logqueue = Queue()
qhandler = logging.handlers.QueueHandler(logqueue)
logger.addHandler(qhandler)

qlistener = logging.handlers.QueueListener(logqueue, loghandler)

logger.setLevel(logging.INFO)
qlistener.start()


async def go_ahead():
    a = 10 / 0  # raise an exception


async def main():
    try:
        await go_ahead()
    except Exception as err:
        logger.info(err, exc_info=True)


asyncio.run(main())

"""