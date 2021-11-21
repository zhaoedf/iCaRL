
import os
import logging
import tqdm

import numpy as np


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
    fh = logging.FileHandler(os.path.join(log_path, mlflow_runid+'.log'), mode='w', encoding='utf-8')  # a指追加模式,w为覆盖模式
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 实现terminal和命令行同时输出
    logger.addHandler(ch)
    logger.addHandler(fh)
    # logger.addHandler(TqdmLoggingHandler())
    
    return logger


def get_class_mean(
    y: np.ndarray,
    # t: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    """Herd the samples whose features is the closest to their class mean.
    :param y: Labels of the data. shape:[nb_samples, ]
    :param t: Task ids of the data.
    :param features: Features of shape (nb_samples, nb_dim).
    :return: The class prototype vector.
    """
    # print(features.shape, y.shape)
    assert features.shape[0] == y.shape[0]
    if len(features.shape) != 2:
        raise ValueError(f"Expected features to have 2 dimensions, not {len(features.shape)}d.")
    # indexes = []

    means = []
    for class_id in np.unique(y):
        # print(len(np.unique(y)))
        class_indexes = np.where(y == class_id)[0]
        class_features = features[class_indexes] # from y.shape(above), it is easy to know the shape of "feature" is [nb_samples, feature_dim]
        # print(class_features.shape, features.shape)
        class_mean = np.mean(class_features, axis=0) # , keepdims=True # *row vector*
        # print(class_mean.shape)
        # print(class_mean.shape)

        means.append(class_mean)
        # dist_to_mean = np.linalg.norm(class_mean - class_features, axis=1) # *row vector*
        # print(dist_to_mean.shape)
        # tmp_indexes = dist_to_mean.argsort()[:nb_per_class]
        
        # indexes.append(class_indexes[tmp_indexes])
    return np.array(means)


