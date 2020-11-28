import tensorflow as tf
from utils import load_image, get_dataset, ResNet50,accuracy_score,precision_score,recall_score
from time import time

model_path = "models/ResNet-model.h5"

file_name = "datasets/test/cat/cat.94.jpg"
cat_label = 0
dog_label = 1

test_path = "datasets\\test"

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices(device_type="GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)

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
    recall =  recall_score(dict_predict)
    precision = precision_score(dict_predict)
    total_time = end_time-start_time
    print("total time: ", total_time)
    print("predict one image time: ", total_time/len(dataset))
    print("accuracy: ",accuracy)
    print("recall: ", recall)
    print("precision: ", precision)

