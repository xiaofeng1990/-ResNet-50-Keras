# 处理文件/目录模块
import os
# shutil 是高级的文件，文件夹，压缩包处理模块
import shutil
from tqdm import tqdm


def mkdir(dirname):
    """Create  a directory. Delete if there is one, then create it"""
    #  determine if the directory exists
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)


ROOT_TRAIN_DIR = "./train_total/"

if __name__ == '__main__':

    # 构建训练数据集合目录
    train_dir = "train"
    mkdir(train_dir)
    train_dog_dir = os.path.join(train_dir, "dog")
    os.mkdir(train_dog_dir)
    train_cat_dir = os.path.join(train_dir, "cat")
    os.mkdir(train_cat_dir)

    # 构建验证集合目录
    val_dir = "val"
    mkdir(val_dir)
    val_dog_dir = os.path.join(val_dir, "dog")
    os.mkdir(val_dog_dir)
    val_cat_dir = os.path.join(val_dir, "cat")
    os.mkdir(val_cat_dir)

    # 构建测试集合目录
    test_dir = "test"
    mkdir(test_dir)
    test_dog_dir = os.path.join(test_dir, "dog")
    os.mkdir(test_dog_dir)
    test_cat_dir = os.path.join(test_dir, "cat")
    os.mkdir(test_cat_dir)

    path_list = [ROOT_TRAIN_DIR + path for path in os.listdir(ROOT_TRAIN_DIR)]

    dogs_list = [path for path in path_list if "dog" in path]
    cats_list = [path for path in path_list if "cat" in path]

    dogs_number = len(dogs_list)
    cats_number = len(cats_list)
    assert dogs_number == cats_number, "dog and cat 数量不相等"

    # 训练数据集合 0.7
    train_len = int(dogs_number * 0.7)
    print("train len = ", train_len )
    # 验证数据集合 0.2
    val_len = int(dogs_number * 0.2)
    print("val len = ", val_len)
    # 测试数据集合 0.1
    test_len = int(dogs_number * 0.1)
    print("test len = ", test_len)

    print("******************** 复制训练图片 ************************ \n")
    for path in tqdm(dogs_list[:train_len], desc='copy train dog'):
        shutil.copy(path, train_dog_dir)
    for path in tqdm(cats_list[:train_len], desc='copy train cat'):
        shutil.copy(path, train_cat_dir)

    print("******************** 复制验证图片 ************************ \n")
    for path in tqdm(dogs_list[train_len:train_len+val_len], desc='copy val dog'):
        shutil.copy(path, val_dog_dir)
    for path in tqdm(cats_list[train_len:train_len+val_len], desc='copy val cat'):
        shutil.copy(path, val_cat_dir)

    print("******************** 复制测试图片 ************************ \n")
    for path in tqdm(dogs_list[train_len+val_len:], desc='copy test dog'):
        shutil.copy(path, test_dog_dir)
    for path in tqdm(cats_list[train_len+val_len:], desc='copy test cat'):
        shutil.copy(path, test_cat_dir)


