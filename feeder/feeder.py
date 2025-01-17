import numpy as np
import tensorflow as tf


class Feeder:
    def __init__(self, data_path, label_path):
        self.data_path = data_path
        self.label_path = label_path

    def get_dataset(self):
        # 加载 Numpy 数据
        data = np.load(self.data_path)
        label = np.load(self.label_path)

        # 使用 from_tensor_slices 创建 tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices({"data": data, "label": label})
        return dataset
