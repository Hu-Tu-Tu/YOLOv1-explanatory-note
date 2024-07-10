from curses import endwin
import xml.etree.ElementTree as ET
import os
import random

"""
这个文件的主要功能是读取文件夹内所有的xml的文件以及信息，将这些信息(name,bbox,class)写入一个txt文件中，并且按照7:3划分训练集和测试集
"""

VOC_CLASSES = (  # 定义所有的类名
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')   # 使用其他训练集需要更改

# VOC_CLASSES=[]# 自己的数据集

# 切换成当前路径-需要修改
# os.chdir('/root/workspace/YOLOV1-pytorch/')

# 定义一些参数
train_set = open('voctrain.txt', 'w')
test_set = open('voctest.txt', 'w')
# Annotations = 'VOCdevkit//VOC2007//Annotations//'
Annotations='VOCdevkit/VOC2007/Annotations/'
xml_files = os.listdir(Annotations)
random.shuffle(xml_files)  # 打乱数据集
train_num = int(len(xml_files) * 0.7)  # 训练集数量
train_lists = xml_files[:train_num]   # 训练列表
test_lists = xml_files[train_num:]    # 测测试列表
# 输出一些信息
print("train_lists:",len(train_lists))
print("test_lists:",len(test_lists))


def parse_rec(filename):  # 输入xml文件名
    """
    读取xml文件信息，在"object"目录下查看"difficult"值是否为1，若不为1则在名为"obj_struct"的字典中存入"bbox"和"name"的信息，
    再将这个字典作为名为"objects"的列表的元素，最终输出这个列表。所以这个名为"objects"的列表中的每一个元素都是一个字典。
    """
    tree = ET.parse(filename)# 生成一个总目录名为tree
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        difficult = int(obj.find('difficult').text)
        if difficult == 1:  # 若为1则跳过本次循环
            continue
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text)),
                              int(float(bbox.find('ymax').text))]
        objects.append(obj_struct)

    return objects


def write_txt():
    count = 0
    for train_list in train_lists: # 生成训练集txt
        count += 1
        image_name = train_list.split('.')[0] + '.jpg'  # 图片文件名
        results = parse_rec(Annotations + train_list)
        if len(results) == 0:
            print(train_list)
            continue
        train_set.write(image_name)
        for result in results:
            class_name = result['name']
            # # 添加类别名字
            # if class_name not in VOC_CLASSES:
            #     VOC_CLASSES.append(class_name)
            
            bbox = result['bbox']
            class_name = VOC_CLASSES.index(class_name)
            train_set.write(' ' + str(bbox[0]) +
                            ' ' + str(bbox[1]) +
                            ' ' + str(bbox[2]) +
                            ' ' + str(bbox[3]) +
                            ' ' + str(class_name))
        train_set.write('\n')
    train_set.close()

    for test_list in test_lists:   # 生成测试集txt
        count += 1
        image_name = test_list.split('.')[0] + '.jpg'  # 图片文件名
        results = parse_rec(Annotations + test_list)
        if len(results) == 0:
            print(test_list)
            continue
        test_set.write(image_name)
        for result in results:
            class_name = result['name']

            # # 添加类别名字
            # if class_name not in VOC_CLASSES:
            #     VOC_CLASSES.append(class_name)

            bbox = result['bbox']
            class_name = VOC_CLASSES.index(class_name)
            test_set.write(' ' + str(bbox[0]) +
                            ' ' + str(bbox[1]) +
                            ' ' + str(bbox[2]) +
                            ' ' + str(bbox[3]) +
                            ' ' + str(class_name))
        test_set.write('\n')
    test_set.close()

"""
if __name__ == "__main__": 的作用
在Python中，每个Python文件（模块）都可以作为脚本直接运行，也可以被其他文件导入。__name__ 是一个特殊变量，
当文件被直接运行时，__name__ 的值被设置为 "__main__"。如果文件是被导入的，则 __name__ 的值会被设置为该模块的名字。
if __name__ == "__main__": 这行代码的作用是判断该文件是否作为主程序运行。如果是，则执行该条件语句块下的代码。
这种方式通常用于提供一个可执行的入口点给该文件，同时也允许该文件中的函数和类被其他文件导入而不会自动执行这些代码。
"""
if __name__ == '__main__':
    write_txt()
    print(VOC_CLASSES)# 类别名称
