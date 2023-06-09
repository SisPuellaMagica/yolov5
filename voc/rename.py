import os
import random
import time
from shutil import move


# 创建临时目录
def create_temp_dir(father_dir):
    temp_dir = father_dir + '/' + 'temp_' + str(int(time.time())) + str(random.randint(0, 2147483647))
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    return temp_dir


# 重命名
def rename(img_dir, xml_dir, new_name):
    print('正在重命名')
    temp_img_dir = create_temp_dir(os.path.dirname(img_dir))
    temp_xml_dir = create_temp_dir(os.path.dirname(xml_dir))
    i = 0
    for xml_filename in os.listdir(xml_dir):
        i += 1
        src_xml = xml_dir + '/' + xml_filename
        dst_xml = temp_xml_dir + '/' + new_name + '_' + str(i) + '.xml'
        move(src_xml, dst_xml)
        src_img = img_dir + '/' + os.path.splitext(xml_filename)[0] + '.jpg'
        dst_img = temp_img_dir + '/' + new_name + '_' + str(i) + '.jpg'
        move(src_img, dst_img)
    i = 0
    for img_filename in os.listdir(img_dir):
        i += 1
        src_background = img_dir + '/' + img_filename
        dst_background = temp_img_dir + '/' + 'background' + '_' + str(i) + '.jpg'
        move(src_background, dst_background)
    os.rmdir(img_dir)
    os.rmdir(xml_dir)
    time.sleep(1)
    os.rename(temp_img_dir, img_dir)
    os.rename(temp_xml_dir, xml_dir)
    print('重命名完成')


if __name__ == '__main__':
    rename(img_dir='', xml_dir='', new_name='')
