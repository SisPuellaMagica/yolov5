import os
from shutil import move


# 分离背景图
def split_background(img_dir, xml_dir):
    print('正在分离背景图')
    background_dir = os.path.dirname(img_dir) + '/' + 'backgrounds'
    if not os.path.exists(background_dir):
        os.makedirs(background_dir)
    for img_filename in os.listdir(img_dir):
        img_filepath = img_dir + '/' + img_filename
        xml_filepath = xml_dir + '/' + os.path.splitext(img_filename)[0] + '.xml'
        if not os.path.exists(xml_filepath):
            move(img_filepath, background_dir + '/' + img_filename)
    print('分离背景图完成')


if __name__ == '__main__':
    split_background(
        img_dir='',
        xml_dir=''
    )
