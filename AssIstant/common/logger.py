import os, time, logging, datetime, traceback
from logging.handlers import RotatingFileHandler
from functools import wraps
from collections.abc import Iterable

from AssIstant.common.config import LOG_DIR, FILE_LOG_LEVEL, STREAM_LOG_LEVEL

log_dir = LOG_DIR
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, r"test.{}.log".format(time.strftime("%Y%m%d")))

class Logger:

    def __init__(self):
        self.__clear_expire_logs()

    @staticmethod
    def __clear_expire_logs():
        now = datetime.datetime.now()
        offset = datetime.timedelta(days=-30)
        before = (now + offset).timestamp()
        files = os.listdir(log_dir)
        for file in files:
            if os.path.splitext(file)[1]:
                file_path = log_dir + '\\' + file
                if os.path.getctime(file_path) < before:
                        os.remove(file_path)

    @staticmethod
    def get_logger():
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logger.setLevel(FILE_LOG_LEVEL)
            log_format = logging.Formatter(
'%(levelname)s - %(asctime)s - %(filename)s:%(lineno)d -[%(module)s:%(funcName)s] - %(message)s')

            file_handler = RotatingFileHandler(
                filename=log_file,
                mode='a',
                maxBytes=5242880,
                backupCount=7,
                encoding='utf-8'
            )
            file_handler.setLevel(FILE_LOG_LEVEL)
            file_handler.setFormatter(log_format)
            logger.addHandler(file_handler)

            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(STREAM_LOG_LEVEL)
            stream_handler.setFormatter(log_format)
            logger.addHandler(stream_handler)
        return logger


logger = Logger.get_logger()


class FormatLog:
    def __init__(self, name: str = ""):
        self._name = str(name)

    def info(self, message, *args, **kwargs):
        logger.info(f"{type(self).__name__}({self._name}) - {message}", *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        logger.warning(f"{type(self).__name__}({self._name}) - {message}", *args, **kwargs)

    def error(self, message, *args, **kwargs):
        logger.error(f"{type(self).__name__}({self._name}) - {message}", *args, **kwargs)

    def debug(self, message, *args, **kwargs):
        logger.debug(f"{type(self).__name__}({self._name}) - {message}", *args, **kwargs)

    def exception(self, message, *args, **kwargs):
        logger.exception(f"{type(self).__name__}({self._name}) - {message}", *args, **kwargs)


def exception_logger(func=None, *, limit=5, catch=(Exception,), exclude=(), reraise=1):
    """
    打印异常日志
    :param func:
    :param limit: 堆栈打印深度，-1表示全部，0表示打印异常信息
    :param catch: 捕获这些异常
    :param exclude: 排除这些异常
    :param reraise: 是否再次引发被捕获的异常，>0引发，0不引发，<0执行exit(-1)
    :return:
    """
    if catch is None:
        catch = (Exception,)
    if not isinstance(catch, Iterable):
        catch = (catch,)
    if exclude is None:
        exclude = ()
    if not isinstance(exclude, Iterable):
        exclude = (exclude,)
    reraise = int(reraise)

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except catch as e:
                if not any(isinstance(e, exc) for exc in exclude):
                    if limit > 0:
                        logger.error(traceback.format_exc(limit=limit))
                    elif 0 == limit:
                        logger.error(f"{type(e).__name__}: {e}")
                    else:
                        logger.error(traceback.format_exc())
                if reraise > 0: raise
                elif reraise < 0: exit(-1)
        return wrapper

    if callable(func):
        return decorator(func)
    else:
        return decorator

if __name__ == '__main__':
    # @exception_logger(limit=0, catch=ValueError, reraise=-1)
    # def func():
    #     # assert False
    #     raise ValueError('fda')
    #
    # func()

    class A(FormatLog):
        pass

    a = A("name")
    a.info("abc")

    ...