---
layout: post
title: 目标检测安卓部署
description: 目标检测安卓部署

---

### 导入
主要是今年的最新yolo模型，也就是yolo v8在安卓手机上的部署，利用了腾讯的ncnn高性能推理框架。[ncnn链接](https://github.com/Tencent/ncnn)。ncnn是用纯C++编写的推理框架，虽然是跨平台的，但是主要用途还是手机端，针对手机端加强优化，支持IOS和安卓。

### demo

#### 简单的demo演示：

![](https://github.com/cryer/cryer.github.io/raw/master/image/10.gif)
