from nets.resnet import ResNet50
from tensorflow.keras.utils import plot_model
from utils import load_image
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers

import tensorflow as tf

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices(device_type="GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)

    train_dir = "datasets/train"
    val_dir = "datasets/val"
    batch_size = 2048
    image_size = 224
    model = ResNet50(input_shape=(image_size, image_size, 3), classes=2)
    # model.load_weights("ResNet50.h5")
    model.summary()
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=[tf.keras.metrics.sparse_categorical_accuracy])


    plot_model(model, to_file='models/ResNet50.png',show_shapes=True, show_layer_names=True)

    logging = TensorBoard(log_dir="logs")
    checkpoint = ModelCheckpoint("models" + "/ResNet_{epoch:04d}-{val_loss:.4f}.h5", monitor="val_loss", mode="min",
                                 verbose=1, save_weights_only=True, save_best_only=True, period=3)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1)

    train_dataset = load_image(train_dir, batch_size, image_size)
    val_dataset = load_image(val_dir, batch_size, image_size)
    model.fit(
        train_dataset,
        epochs=2,
        callbacks=[reduce_lr, logging, checkpoint, early_stopping],
        validation_data=val_dataset,
        initial_epoch=0,
        max_queue_size=20,
        workers=4)

    # 保存keras特有的h5文件模型，这个可以直接加载
    model.save("models" + "/ResNet-model.h5")
    # 保存keras的权重文件，这个需要先搭建模型然后加载权重文件
    model.save_weights("models" + "/ResNet-weights.h5")
    # 保存pb文件，用于tf直接加载模型
    tf.saved_model.save(model, "models/pb")
