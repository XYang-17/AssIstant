import os, time, logging, datetime, traceback
from logging.handlers import RotatingFileHandler
from functools import wraps
from collections.abc import Iterable
from typing import Callable, Any

from sympy.utilities.iterables import iterable

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


class BadResult:
    ...


BAD = BadResult()


def _normalize_exc(value, default=()):
    if value is None:
        return tuple(default) if default else ()
    if isinstance(value, type) and issubclass(value, BaseException):
        return (value,)
    items = value if isinstance(value, Iterable) else (value,)
    return tuple(
        exc for exc in items
        if isinstance(exc, type) and issubclass(exc, BaseException)
    ) or default


def exception_logger(
        func: Callable=None,
        *,
        catch: tuple[Exception]=(Exception,),
        exclude: tuple[Exception]=(),
        limit: int=5,
        reraise:int=1,
        fallback: Any=BAD):
    """
    打印异常日志
    :param func:
    :param limit: 堆栈打印深度，-1表示全部，0表示打印异常信息
    :param catch: 捕获这些异常
    :param exclude: 排除这些异常
    :param reraise: 是否再次引发被捕获的异常，>0引发，0不引发，<0执行exit(-1)
    :param fallback: 异常兜底，当异常被捕获时执行，用于兜底返回
    :return:
    """
    catch = _normalize_exc(catch, (Exception,))
    exclude = _normalize_exc(exclude, ())
    limit = int(limit or 5)
    reraise = int(reraise or 1)

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except catch as e:
                # 打印异常日志
                if not any(isinstance(e, exc) for exc in exclude):
                    if limit > 0:
                        logger.exception(traceback.format_exc(limit=limit))
                    elif 0 == limit:
                        logger.exception(f"{type(e).__name__}: {e}")
                    else:
                        logger.exception(traceback.format_exc())

                # 再次引发，或终止程序
                if reraise > 0: raise
                elif reraise < 0: SystemExit(-1)

                # 兜底处理
                if callable(fallback):
                    try:
                        return fallback(*args, **kwargs)
                    except Exception as ex:
                        logger.exception(f"An exception occurred during the fallback processing\n{type(ex).__name__}: {ex}")
                        SystemExit(-1)
                return fallback
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