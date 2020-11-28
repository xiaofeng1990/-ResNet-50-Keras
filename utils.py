import tensorflow as tf
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

cat_label = 0
dog_label = 1


def accuracy_score(dict_predict):
    TP = dict_predict["TP"]
    FP = dict_predict["FP"]
    FN = dict_predict["FN"]
    TN = dict_predict["TN"]

    accuracy = (TP + TN) / (TP + TN + FP + FN)

    return accuracy


def recall_score(dict_predict):
    TP = dict_predict["TP"]
    FP = dict_predict["FP"]
    FN = dict_predict["FN"]
    TN = dict_predict["TN"]

    recall = TP / (TP + FN)
    return recall


def precision_score(dict_predict):
    TP = dict_predict["TP"]
    FP = dict_predict["FP"]
    FN = dict_predict["FN"]
    TN = dict_predict["TN"]

    precision = TP / (TP + FP)

    return precision


class ImageClass():
    """Stores image data, and image label"""

    def __init__(self, path, label):
        self.path = path
        self.label = label

    def show(self):
        image_string = tf.io.read_file(self.path)
        image = tf.image.decode_jpeg(image_string)

        image = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)
        label = "cat" if self.label == 0 else "dog"
        cv2.imshow(label, image)
        cv2.waitKey(0)


def get_dataset(path):
    dataset = []

    cats_dir = os.path.join(path, "cat")
    dogs_dir = os.path.join(path, "dog")

    cat_filenames = [os.path.join(cats_dir, filename) for filename in os.listdir(cats_dir)]
    dog_filenames = [os.path.join(dogs_dir, filename) for filename in os.listdir(dogs_dir)]

    for cat_path, dog_path in zip(cat_filenames, dog_filenames):
        dataset.append(ImageClass(cat_path, cat_label))
        dataset.append(ImageClass(dog_path, dog_label))

    return dataset


class ResNet50():
    def __init__(self, model_path):
        assert os.path.exists(model_path), "{} is not exists".format(model_path)

        self.model = load_model(model_path)
        self.model.summary()
        self.size = self.model.input_shape[1]

    def predict(self, path):
        image = self._path_to_tensor(path)
        y = self.model.predict(image)
        lable = np.argmax(y)

        return lable

    def _path_to_tensor(self, path):
        image_string = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image_string)
        image = tf.image.resize_with_pad(image, self.size, self.size) / 255.0
        image = np.expand_dims(image.numpy(), axis=0)

        return image


def load_image(path, batch, size):
    def _decode_and_resize(filename, label):
        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image_resized = tf.image.resize_with_pad(image, size, size) / 255.0
        return image_resized, label

    cats_dir = path + "/cat/"
    dogs_dir = path + "/dog/"

    cat_filenames = tf.constant([cats_dir + filename for filename in os.listdir(cats_dir)])
    dog_filenames = tf.constant([dogs_dir + filename for filename in os.listdir(dogs_dir)])
    filenames = tf.concat([cat_filenames, dog_filenames], axis=-1)
    labels = tf.concat([
        tf.zeros(cat_filenames.shape, dtype=tf.int32),
        tf.ones(dog_filenames.shape, dtype=tf.int32)],
        axis=-1)

    datasets = tf.data.Dataset.from_tensor_slices((filenames, labels))
    datasets = datasets.map(
        map_func=_decode_and_resize,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    datasets = datasets.shuffle(buffer_size=batch * 100)
    datasets = datasets.batch(batch)
    datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)

    return datasets


if __name__ == "__main__":
    # data_dir = "datasets/train"
    # dataset = load_image(data_dir, 64, 150)
    # for images, labels in dataset:
    #     print(images.shape)
    #     print(labels)
    #     plt.imshow(images[0].numpy())
    #     plt.show()

    # file_name = "datasets/test/cat/cat.94.jpg"
    # image = ImageClass(file_name, 0)
    # image.show()

    gpus = tf.config.list_physical_devices(device_type="GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)

    model_path = "models/ResNet-model.h5"
    file_name = "images/test3.jpg"
    model = ResNet50(model_path)
    label = model.predict(file_name)
    print(label)
