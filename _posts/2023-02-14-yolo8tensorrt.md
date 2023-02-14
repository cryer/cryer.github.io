---
layout: post
title: 如何提高YOLO帧率
description: 如何提高 YOLO帧率

---

### 导入

YOLO系列目标检测发展到现在已经是v8版本了，而我们使用YOLO也不像以前那样麻烦了，`ultralytics`库可以非常方便的使用所有版本的YOLO，不管是自定义数据集，自己训练，还是使用官方权重，都很方便。但是很多开发者可能会发现使用YOLO的帧率不高。这篇博客简单介绍下如何提高本地运行YOLO的帧率。



### 正文

首先先给出使用YOLO的基本样例

```python

from ultralytics import YOLO
import cv2
import math
import time

cap = cv2.VideoCapture("xx.mp4") 
model = YOLO("yolov8s.pt")

prev_frame_time = 0
new_frame_time = 0

while cap.isOpened():
    new_frame_time = time.time()
    success, img = cap.read()
    if not success:
      break
   results = model(img, stream=True, verbose=False)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:          
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0]) 

            cv2.putText(img, f'{r.names[cls]} {conf}',
                        (max(0, x1), max(35, y1)),
                        fontFace= cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=0.7,
                        color=(255, 255, 0),
                        thickness= 1,
                        )
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(img, f'fps:{fps:.1f}',
                       (20, 20),
                        fontFace= cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=0.8,
                        color=(0, 0, 0),
                        thickness= 1,
                        )

    cv2.imshow("Image", img)
    cv2.waitKey(1)
    
```

- 选择合适的模型

    针对你的项目，选择合适大小的模型，模型越大，精度越高但是帧率也越低，因此需要选择精度符合你的要求，但是模型尽量小的。每个模型都包括`n,s,m,l,x`，模型参数大小依次递增

- 修改图片尺寸

    对输入到模型的尺寸进行一个缩小，比如宽高缩小一半，帧率可以有效提升，在`ultralytics`中，只需要修改model的参数`imgsz`即可，比如`model(img, imgsz=(w/2,h/2), stream=True, verbose=False)`

- 调整batch size，如果显存足够，可以增加每次处理的批大小
  
  比如`model(img, batch = 32, stream=True, verbose=False)`

- 使用半精度FP16
  
  `model(img, half= True, stream=True, verbose=False)`

- 选择GPU推理，以及使用更好的GPU
  
  选择GPU推理而不是CPU推理，原因相信不用多说，另外，GPU的性能影响非常大，好的GPU，尤其是现在带TensorCore的GPU对神经网络的推理性能领先非常多，如果自身显卡不太好，既可以考虑使用colab来推理，colab的T4GPU效果还不错。

- 使用TensorRT加速推理

    TensorRT是Nvidia官方的推理框架，可以有效加速模型的推理，不单单是YOLO，任何的神经网络都可以加速推理。基本的步骤：首先正常使用深度学习框架训练模型，比如使用pytorch训练，生成pt模型参数。然后导入为ONNX的开放交换格式，pytorch自身就支持导出ONNX模型，然后把ONNX模型进一步转换成TensorRT模型，使用第三方库tensorrt库即可转换，然后利用tensorrt推理TensorRT模型即可。

方便的是，我们不需要这么复杂，因为`ultralytics`本身就支持tensorrt模型的推理，因此我们只需要转换成tensorrt模型即可。

#### 使用TensorRT具体步骤

```shell
pip install tensorrt
pip install tensorrt_lean
pip install tensorrt_dispatch
pip install onnx onnxsim onnxruntime-gpu
```

安装必要的第三方库，tensorRT提供多种运行时库，包括libnvinfer.so/.dll，libnvinfer_lean.so/.dll,libnvinfer_dispatch.so/.dll 对应 default runtime（默认库），lean runtime（小型库），dispatch runtime（小型 shim 库）三个运行时库。

然后在命令行中使用：

```shell
yolo export model=yolov8x.pt format=engine half=True device=0 dynamic=True

```

`format=engine`表示模型转换成TensorRT模型，`half=True`表示使用半精度，`device=0`表示使用GPU转换，`dynamic=True`表示支持动态输入尺寸

然后其实在命令行就可以使用转换后的模型进行推理了，比如：

```shell
yolo detect predict model=yolov8x.engine source="bus.jpg" device=0
```

如果需要在代码中调用，对于`ultralytics`来说，本身就支持tensorrt的推理，因此直接把`yolov8s.pt`替换成`yolov8s.engine`即可。



### 结果

因为我写这篇博客的时候在外地，只能用GPU（1050Ti）很老的笔记本进行推理，本地YOLO帧率不加任何优化的情况下，使用s模型，解析1K分辨率物体密集的视频，平均帧只有28帧。仅仅使用tensorrt模型，在colab上推理之后，帧率就可以提高到超过100帧。如果进一步使用imgsz缩小输入尺寸为原来的一半，帧率更是可以逼近150帧，使用更好的GPU的话，突破200帧是很简单的。


