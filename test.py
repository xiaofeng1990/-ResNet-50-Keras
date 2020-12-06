import tensorflow as tf
from utils import load_image, get_dataset, ResNet50, accuracy_score, precision_score, recall_score
from time import time
import cv2
import numpy as np

model_path_h5 = "models/ResNet-model.h5"
model_path_pb = "models/pb"
file_path = "datasets/test/cat/cat.94.jpg"
cat_label = 0
dog_label = 1
image_size = 224
test_path = "datasets\\test"


def predict_by_pb(model_path, file_path):
    """
    使用pb文件测试一张图片
    :param model_path: pb文件路径
    :param file_path: 单张图片路径
    :return:
    """
    # 读取文件
    image = cv2.imread(file_path)

    # 图像resize到模型的输入尺寸
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    # 将图像转换到rgb
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0

    # 将数据转换为tf.float32类型
    image = tf.cast(image, dtype=tf.float32)
    image = np.expand_dims(image, axis=0)
    # 加载pb文件
    model = tf.saved_model.load(model_path)
    y_pred = model(image)
    print(y_pred)


def calculate(model_path, test_path):
    """
    预测测试集合，计算精度，召回率，准确度
    :param model_path:
    :param test_path:
    :return:
    """
    seq = ('TP', 'FP', 'FN', 'TN')
    dict_predict = dict.fromkeys(seq, 0)
    model = ResNet50(model_path)
    dataset = get_dataset(test_path)
    start_time = time()
    for data in dataset:
        label = model.predict(data.path)
        if data.label == cat_label:
            if label == data.label:
                dict_predict["TP"] += 1
            else:
                dict_predict["FN"] += 1
                print(data.path)
        elif data.label == dog_label:
            if label == data.label:
                dict_predict["TN"] += 1
            else:
                dict_predict["FP"] += 1
                print(data.path)
    end_time = time()
    accuracy = accuracy_score(dict_predict)
    recall = recall_score(dict_predict)
    precision = precision_score(dict_predict)
    total_time = end_time - start_time
    print("total time: ", total_time)
    print("predict one image time: ", total_time / len(dataset))
    print("accuracy: ", accuracy)
    print("recall: ", recall)
    print("precision: ", precision)


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices(device_type="GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)

    # predict_by_pb(model_path_h5, file_path)

    calculate(model_path_h5, test_path)



