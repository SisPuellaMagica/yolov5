import os
import random


# 划分训练集和验证集
def split_train_val(img_dir, train_percent=0.9):
    print('正在划分训练集和验证集')
    img_list = os.listdir(img_dir)
    random.shuffle(img_list)
    img_num = len(img_list)
    train_num = int(img_num * train_percent)
    train_filepath = os.path.dirname(img_dir) + '/' + 'train.txt'
    val_filepath = os.path.dirname(img_dir) + '/' + 'val.txt'
    with open(train_filepath, 'w', encoding='utf-8') as f:
        for i in range(0, train_num):
            img_filepath = img_dir + '/' + img_list[i]
            f.write(img_filepath)
            if i < train_num - 1:
                f.write('\n')
    with open(val_filepath, 'w', encoding='utf-8') as f:
        for i in range(train_num, img_num):
            img_filepath = img_dir + '/' + img_list[i]
            f.write(img_filepath)
            if i < img_num - 1:
                f.write('\n')
    print('划分训练集和验证集完成')


if __name__ == '__main__':
    split_train_val(
        img_dir='',
        train_percent=0.9
    )
