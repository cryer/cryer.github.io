---
layout: post
title: Tensorflow目标检测API
description: Tensorflow目标检测API
---


# 简介

Tensorflow 目标检测API其实是官方提供的一组样例，里面不仅包含目标检测，还有OCR，GAN，自编码器，img2txt等等，感兴趣的可以自己慢慢研究。
链接如下[Tensorflow APIs](https://github.com/tensorflow/models)，找到目标检测目录，大致结构如下：
```
├─.ipynb_checkpoints
├─card_inference_graph
│  └─saved_model
│      └─variables
├─data
├─images
│  ├─test
│  └─train
├─models
│  └─__pycache__
├─test_ckpt
├─test_data
├─test_images
├─training
```
不常用的我都删除了，上面是几个比较重要的目录。其中training，images，card_inference_graph目录是我自己新建的，用来存放自定义训练的东西，
因为我们重点就是训练自己的数据集，而不是运行官方的模型，你可以首先测试object_detection_tutorial.ipynb，查看你的环境配置是否成功。

# 环境配置

我这里不会说的太详细的，而且每个人的问题可能都有区别，具体根据每一步去官网或者谷歌查找。

* 首先就是Tensorflow的安装，建议安装1.4.0及以上版本，但是要注意cuda和cudnn的配套，1.7貌似只支持cuda9.0.CPU版本的就别装了，
用CPU跑目标检测绝对会让你发疯的。
* GPU的tensorflow安装好了之后，下一步就要把上面的官方API仓库下载到本地，可以下载zip解压或者git clone。
* 安装和配置Protobuf，一种轻便高效的结构化数据存储格式,平台无关、语言无关、可扩展,可用于通讯协议和数据存储等领域。直接从谷歌官方下载，
[下载地址](https://github.com/google/protobuf/releases)，我选择的是windows版本。解压后bin目录下的rotoc.exe放到C:\Windows，这是为了能够直接
在命令行运行。
* 进入下载的API目录``` ...\models\research\object_detection\protos\```下，shift+鼠标右键，选择在当前目录打开命令提示符，然后运行
```protoc *.proto --python_out=. ```，提示找不到proto文件的话就直接一个一个输入proto文件的全名，多次运行，把所有的Proto文件都转化成py文件。
* 环境变量的配置，models/research/ 及 models/research/slim 加入PYTHONPATH路径，PYTHONPATH路径没有的新建一个。
* 最后测试自带的demo，object_detection_tutorial.ipynb，看看环境配置是否有问题。

# 测试自己的数据集

## 数据准备

你可以选择直接从百度谷歌爬取相应的图片，如果有数据集的话跳过

## 数据集的标注

* 对数据集的每一张图片进行标注，包括需要检测物体的坐标，以及所属的类别，这是很耗时间的一部分，推荐使用[labelImg](https://github.com/tzutalin/labelImg)
这种类似的软件进行标注，标注完每个图片对应一个xml文件，里面记录了类别，坐标等信息。

* 下一步就是讲xml转化成tensorflow固有的格式TFRcord，可以用你喜欢的方式进行，这里是先将xml转成csv，再转成record。

转成csv：

```
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

os.chdir('G:\\__TF_examples\\models\\research\\object_detection\\image_car\\test')
path = 'G:\\__TF_examples\\models\\research\\object_detection\\image_car\\test'

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    image_path = path
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('card_test.csv', index=None)
    print('Successfully converted xml to csv.')


main()
```

转成record：

```
import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

os.chdir('G:\\__TF_examples\\models\\research\\object_detection\\')

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def class_text_to_int(row_label):
    if row_label == 'card':
        return 1
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(os.getcwd(), 'images/test')
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
```

细节不多说的，有问题可以直接留言，仔细看代码，将对应的地址部分好好修改。

## 数据存放

将最后的两个record文件，一个train.record，一个test.record，放到data目录下，另外我们需要创建一个映射，是类别到标签的映射，也在data目录下，
后缀名为pbtxt，你可以发现已经提供了一些了，随便复制一个，改个名字，修改内容和你的数据集匹配。
```
item {
  id: 1
  display_name: "card"
}
```
比如我是要做车牌号的目标检测，只有一类，就定义如上格式就好了，几类定义几个，id从1开始。

## 选择模型

数据准备ok了，下一步就是选择模型了，进入到object_detection目录下```samples\configs```下，你可以看到很多的模型配置文件，SSD和Faster-RCNN都有，
每一种都有不同网络结构的实现，我这里选择了最简单的，所以也是速度最快的ssd_mobilenet_v1_coco.config，mobilenet+ssd。把它复制粘贴到前面的training目录下，
自己创建的一个目录，然后编辑修改配置文件，根据我的要求，检测车牌，只有一个类别。

所以：
* 9行，改成num_classes: 1
* 141行，改成batch_size: 1，你也可以改大点，因为我的机器内存和显存不高，所以我设为1
* 174行，改为input_path: "data/card_train.record"，训练集输入地址
* 176行，改为label_map_path: "data/card.pbtxt"，这是之前步骤中创建的自定义映射，下面测试部分也是这个
* 188行，改为input_path: "data/card_test.record"，测试集输入地址
* 190行，改为label_map_path: "data/card.pbtxt"，标签映射

```
fine_tune_checkpoint: "ssd_mobilenet_v1_coco_11_06_2017/model.ckpt"
from_detection_checkpoint: true
```
关于这两行代码，是微调官方训练好的模型的，如果你想从头训练，就删除，微调的话，需要下载这个模型的ckpt文件。

# 训练

目录切换到object_detection根目录，运行：
```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_coco.config  
```
训练一段时间后，可以ctrl+c中断训练，下次在运行上面命令会自动加载最近的ckpt的，可以继续训练，所以不用担心需要从头训练。
可以运行```tensorboard --logdir=training  ```查看一些损失之类的图表信息。

参数保存点都在training目录下，可以进去看一下，数字最大的就是最新的。

# 测试

我在20张训练集上训练了1个小时左右，保存了第9000多步的模型参数，接下来利用这个参数进行预测，不过可想而知，效果肯定是很差的，
因为训练的样本数太少，训练的时间也太短。我只是做一个测试：

* 首先是导出计算图

```
python export_inference_graph.py \ --input_type image_tensor \ --pipeline_config_path training/ssd_mobilenet_v1_coco.config \ 
--trained_checkpoint_prefix training/model.ckpt-9378 \
--output_directory card_inference_graph  
```
其中基本有两处需要修改：
* model.ckpt-9378，9378就是你training’目录下最大数字的那个
* output_directory需要修改，也就是计算图输出的目录，我选择的是自己新建的一个card_inference_graph目录

这样计算图就导出到了card_inference_graph中了，下面就可以开始测试了。
测试还是利用官方的测试demo，object_detection_tutorial.ipynb，只要稍作修改。

* MODEL_NAME = 'card_inference_graph'，改成自己的计算图目录
* PATH_TO_LABELS = os.path.join('data', 'card.pbtxt')，换成自己的映射
* 修改NUM_CLASSES = 1，因为我只有一类
* 删除下载模型的部分代码
* PATH_TO_TEST_IMAGES_DIR = 'test_images'，根据这个吧测试的图片放到test_images目录下，最好修改名称为image+数字.jpg，
然后把测试代码索引1-3改成你测试图片的索引，比如你有3张测试图片名字为image3.jpg，image4.jpg，image5.jpg，就
把索引改成3-6.

我的测试效果如下：

![](https://github.com/cryer/cryer.github.io/raw/master/image/44.png)

* 如果你的测试图片上面没有框的话，有几种可能，第一，你的标签没有统一，就是labelimg软件标注的类别，和映射以及生成tfrecord中的标签名要一致。
当然最大的可能其实是你的样本数太少或者训练时间太短，这是为什么呢？因为了解目标检测的人肯定知道，我们一般对每张图片生成数百上千张的框，那么多框肯定不可
能都有效的，我们首先排除框的手法就是删除置信度低的，一般50%以下的删除，其次是根据IOU，利用NMS非极大值抑制来减少。我们现在关注前一个，我们进入画框的
程序里看一下，也就是utils里visualization_utils.py文件，往下找到画框的函数，发现果然有一个阈值的min_score_thresh参数，而且默认正是0.5，
也就是说置信度低于50%的框我们是不画出来的。那就很明显了，正因为我们样本不足或者训练时间太少，模型训练不充分，导致预测时每个框的置信度都很低，
从我的测试图中可以看出，基本只有2%左右，也就是0.02，远小于0.5，自然也就不会画框了。所以你需要调低这个阈值，就可以显示框了，当然最好还是训练时间长一点。














