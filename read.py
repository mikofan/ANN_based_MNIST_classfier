#定义一个读取文件的函数模块
#输入，数据和标签

import struct
import numpy as np

def read_datasets(datasets_images, datasets_lables):
    # 读取标签数据集
    '''
    标签的数据格式：
    第0 ~ 3字节:    是32位整型数据，取值为0x00000801(2049)，即用幻数2049记录文件数据格式，这里的格式为文本格式。

    第4 ~ 7个字节:  是32位整型数据，取值为60000（训练集时）或10000（测试集时），用来记录标签数据的个数；

    第8个字节 ~ ):  是一个无符号型的数，取值为对应0~9 之间的标签数字，用来记录样本的标签。
    '''
    with open(datasets_lables, 'rb') as lbpath:
        labels_magic, labels_num = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    
    # 读取图片数据集
    '''
    图片的数据格式
    第0 ~ 3字节，       是32位整型数据，取值为0x00000803(2051)，即用幻数2051记录文件数据格式，这里的格式为图片格式。

    第4～7个字节，      是32位整型数据，取值为60000（训练集时）或10000（测试集时），用来记录图片数据的个数；

    第8～11个字节，     是32位整型数据，取值为28，用来记录图片数据的高度；

    第12～15个字节，    是32位整型数据，取值为28，用来记录图片数据的宽度；

    第16个字节 ~ ），   是一个无符号型的数，取值为0~255之间的灰度值，用来记录图片按行展开后得到的灰度值数据，其中0表示背景（白色），255表示前景（黑色）。
    '''
    with open(datasets_images, 'rb') as imgpath:
        images_magic, images_num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(images_num, rows * cols) 
    
    return (images, labels)


'''
struct.unpack('>II', lbpath.read(8))
表示，从lbpath中读8个字节
每个’I’代表一个无符号整数，占4字节
'''
    