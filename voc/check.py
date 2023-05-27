import os
from PIL import Image
import xml.etree.ElementTree as ElementTree


# 检查图片文件
def check_img_file(img_filepath):
    try:
        Image.open(img_filepath)
        if os.path.splitext(img_filepath)[1].lower() != '.jpg':
            os.rename(img_filepath, os.path.splitext(img_filepath)[0] + '.jpg')
        return True
    except:
        return False


# 检查图片目录
def check_img_dir(img_dir):
    flag = True
    for img_filename in os.listdir(img_dir):
        img_filepath = img_dir + '/' + img_filename
        if not check_img_file(img_filepath):
            flag = False
            print(img_filepath, '不是图片')
    return flag


# 检查xml
def check_xml_file(xml_filepath):
    if os.path.splitext(xml_filepath)[1].lower() != '.xml':
        return False
    try:
        root = ElementTree.parse(xml_filepath).getroot()
        if root.tag != 'annotation':
            return False
        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)
        object_nodes = root.findall('object')
        if not len(object_nodes) >= 1:
            return False
        for object_node in object_nodes:
            name = object_node.find('name').text
            if len(name.strip()) == 0:
                return False
            xmin = int(object_node.find('bndbox').find('xmin').text)
            ymin = int(object_node.find('bndbox').find('ymin').text)
            xmax = int(object_node.find('bndbox').find('xmax').text)
            ymax = int(object_node.find('bndbox').find('ymax').text)
            if not (0 <= xmin and xmin < xmax and xmax <= width and 0 <= ymin and ymin < ymax and ymax <= height):
                return False
        return True
    except:
        return False


# 检查xml目录
def check_xml_dir(xml_dir):
    flag = True
    for xml_filename in os.listdir(xml_dir):
        xml_filepath = xml_dir + '/' + xml_filename
        if not check_xml_file(xml_filepath):
            flag = False
            print(xml_filepath, '不是合法的xml标注')
    return flag


# 检查voc数据集
def check(img_dir, xml_dir):
    print('正在检查voc数据集')
    flag1 = check_img_dir(img_dir)
    flag2 = check_xml_dir(xml_dir)
    flag3 = True
    for xml_filename in os.listdir(xml_dir):
        img_filepath = img_dir + '/' + os.path.splitext(xml_filename)[0] + '.jpg'
        xml_filepath = xml_dir + '/' + xml_filename
        if check_img_file(img_filepath) and check_xml_file(xml_filepath):
            img = Image.open(img_filepath)
            size_node = ElementTree.parse(xml_filepath).getroot().find('size')
            if not (img.width == int(size_node.find('width').text) and img.height == int(size_node.find('height').text)):
                flag3 = False
                print(img_filepath, xml_filepath, '宽高不匹配')
        elif (not check_img_file(img_filepath)) and check_xml_file(xml_filepath):
            flag3 = False
            print(xml_filepath, '缺失对应图片')
        else:
            flag3 = False
    flag = flag1 and flag2 and flag3
    if flag:
        print('检查通过')
    return flag


if __name__ == '__main__':
    check(
        img_dir='',
        xml_dir=''
    )
