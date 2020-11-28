from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, Input, ZeroPadding2D, MaxPooling2D, \
    AveragePooling2D, Flatten, Dense, Dropout

from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform


def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block
    :param X: input tensor of shape (m, height_prev, width_prev, channel_prev)
    :param f: integer, specifying the shape of the middle CONV's window for the main path
    :param filters: python list of integers, defining the number of filters in the CONV layers of the main path
    :param stage: integer, used to name the layers, depending on their position in the network
    :param block: string/character, used to name the layers, depending on their position in the network
    :return: X -- output of the identity block, tensor of shape (n, height, width, channel)
    """
    # defining name basis
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path 卷积核尺寸为1， 步进为1，功能为压缩channel
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding="valid", name=conv_name_base + "2a",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = Activation("relu")(X)

    # Second component of main path 使用padding 为same， f*f的卷积核，strides为1，用于特征提取
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same", name=conv_name_base + "2b",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path 卷积核尺寸为1， 步进为1，功能为压缩channel
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block
    :param X: input tensor of shape (m, height_prev, width_prev, channel_prev)
    :param f: integer, specifying the shape of the middle CONV's window for the main path
    :param filters: python list of integers, defining the number of filters in the CONV layers of the main path
    :param stage: integer, used to name the layers, depending on their position in the network
    :param block: string/character, used to name the layers, depending on their position in the network
    :param s: Integer, specifying the stride to be used
    :return: X -- output of the identity block, tensor of shape (n, height, width, channel)
    """
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    # First component of main path strides=s，图片尺寸/s
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path 特征提取
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), name=conv_name_base + '2b', padding='same',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines) 压缩 channel
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def ResNet50(input_shape=(64, 64, 3), classes=6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    :param input_shape: shape of the images of the dataset
    :param classes: integer, number of classes
    :return: a Model() instance in Keras
    """
    # Define the input as a tensor with shape input_shape
    X_input = Input(shape=input_shape)

    # Zero-Padding (3, 3)，第一个3 表示在height的两变各加上3列0，第二个3表示在width的两边各加3行0 (4,4,3)-->(10, 10, 3)
    X = ZeroPadding2D(padding=(3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block="a", s=1)
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block="b")
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block='b')
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block='c')
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='b')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='c')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='d')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='e')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block='b')
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block='c')

    X = AveragePooling2D((2, 2), name='avg_pool')(X)

    # output layer 展平层
    X = Flatten()(X)
    X = Dropout(0.3)(X)
    Y = Dense(classes, activation="softmax", name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    model = Model(inputs=X_input, outputs=Y, name="ResNet50")

    return model
