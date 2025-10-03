---
layout: post
title: 车牌检测
description: 印度车牌检测
---

## 介绍

上一篇博客中介绍了如何利用TensorFlow目标检测API检测自己的数据集，并且随便用了20张百度的图片来做了一个实例，最后训练的效果也当然很差，
所以这一次我打算好好的训练一次车牌检测。没有看上一篇博客的建议先去看，这篇博客不涉及具体细节，都在上篇博客中说到。

## 模型

鉴于上一次选择的模型相对简单，这次选择了一个稍微复杂的模型，faster_rcnn_inception_v2，并且不从头开始训练了，下载了官方的预训练模型，
是在coco数据集上训练好的模型，[链接](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)，
我也给出链接，可以自行下载解压到object_detection目录下。我们将在这个模型上进行微调。

## 数据集

采用的是印度车牌的数据集，一共训练集682张，测试集336张，可以说很多了，对于这个目标检测任务来说足够了。相对而言数据集还是蛮大的，如果你们不想自己弄数据集，
或者不想漫长的标注，可以留言，我会上传。图片的标注是我的一个同学完成，很辛苦233。

## TFRecord

当然图片数据集要转化成TensorFlow喜欢的TFRecord格式，具体方法我就不多说了，见上一篇博客，一定要注意标签要修改，与你的目标对应，我这里就是只有一类，
叫plate。

## 注意点

其实我训练时是经历过失败的，也就是训练了几万步发现最后测试时只有1%不到的分数，这明显是和没训练一样，我推测原因可能是config文件中的路径没有使用绝对路径，
使用相对路径可能无法识别。后来都是用了绝对路径之后，训练了8千步，效果就已经可以了，虽然微调预训练模型的功劳很大。

## 测试

区区一小时不到，训练了8000多步，导出模型后测试效果如下：

![](https://github.com/cryer/cryer.github.io/raw/master/images/1.jpg)

![](https://github.com/cryer/cryer.github.io/raw/master/images/2.jpg)

![](https://github.com/cryer/cryer.github.io/raw/master/images/3.jpg)

可以看到效果真的很完美了，甚至测试别的国家的车牌，比如中国的车牌，效果也很不错：

![](https://github.com/cryer/cryer.github.io/raw/master/images/4.jpg)

我们知道，测试集和训练集开发级分布不一致的时候，测试的效果一般不会太好的，但是中国的车牌和印度的车牌不管是背景颜色还是内容都有不小差别，
测试的结果依然很不错，这就说明这个模型的训练效果是很不错的了。

