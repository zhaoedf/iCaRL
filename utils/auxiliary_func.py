
import os
import logging
import tqdm


# def setup_logger(name, formatter, log_file, level=logging.INFO):
#     """To setup as many loggers as you want"""

#     handler = logging.FileHandler(log_file)        
#     handler.setFormatter(formatter)

#     logger = logging.getLogger(name)
#     logger.setLevel(level)
#     logger.addHandler(handler)

#     return logger


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)  

# @defeng [checked]
def setup_logger(log_path, mlflow_runid):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    
    # 输出到console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG) # 指定被处理的信息级别为最低级DEBUG，低于level级别的信息将被忽略
    
    # 输出到file
    fh = logging.FileHandler(os.path.join(log_path, mlflow_runid), mode='w', encoding='utf-8')  # a指追加模式,w为覆盖模式
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 实现terminal和命令行同时输出
    logger.addHandler(ch)
    logger.addHandler(fh)
    # logger.addHandler(TqdmLoggingHandler())
    
    return logger