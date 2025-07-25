---
layout: post
title: AI产品级部署
description: AI产品级部署

---

### 导入

几年前写过一篇博客讲的是如何发布产品，提到了docker的基本用法，也用一个简单的例子讲解了docker的多阶段构建。今天就详细展开说说如何利用docker产品级，工业级的部署一个AI项目。

我个人其实对AI的工程研究程度进行了简单的分级，入门阶段是使用非常干净的数据集，比如机器学习入门可能接触的第一个数据集，iris鸢尾花数据集，直接拿来用，利用最基本的机器模型或者自己定义的稍微复杂一点的机器模型来学习，这个阶段还称不上工程。稍微深入一点的阶段，是有一定的数据工程能力，能够对现实的总是不那么干净数据集进行分析和清理，并且可以对数据集进行训练集验证集测试集的划分，掌握交叉验证，然后利用更复杂的模型，比如深层网络模型或者近年来流行的LLM进行训练和微调，还要考虑一定的模型鲁棒性。更加深入一点的阶段，除了有更强的数据工程能力，还需要掌握以下几点：

- logging日志的高效使用

- 一定的test测试能力，指的不是基本的单元测试，而是生产部署的测试

- 容器的封装和使用

- API的设计和暴露

我觉得到这个阶段就算是一个比较合格的AI工程师了，涵盖了开发，测试，发布，接下去只要在维护和协作上深入，就可以变得更强。



### 训练模型

接下来就以一个最基础的AI项目，图像分类来说说该如何进行工业级部署。就用大家研究深度学习MNIST数据集之后接触的第二个数据集CIFAR-10数据集来举例。随便写一个CNN模型就行：

```python
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*8*8, 10)
    
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 32*8*8)
        x = self.fc1(x)
        return x


```

简单到不需要任何解释的代码，如果是更复杂一点的模型的话，要生产部署的话，我们或许要考虑半精度FP16，所以可以加一个函数：

```python
def convert_to_fp16(model):
    model.half()  # 转换权重为FP16
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()  # BN层保持FP32
    return model
```

这里只是举个例子，实际上，pytorch1.6之后，amp模块可以很方便进行混合精度的训练，不需要开发者过多操心。顺便贴一下训练的代码：

```python
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CNN

# 数据加载
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

# 初始化模型和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 训练循环
for epoch in range(2):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print(f'Epoch {epoch+1}, Batch {i+1}: loss {running_loss/200:.3f}')
            running_loss = 0.0

# 保存模型
torch.save(model.state_dict(), 'model.pth')
print('Training finished and model saved')

```

同样没什么好说的，简单训练2个epoch做个样子，然后保存模型，之后就不需要这个训练的代码了。这个例子没有什么日志的必要性。

### docker测试部署

接下来把需要的文件单独放到项目目录下，进入docker容器的测试部署，其实就是来到了测试阶段。

```
app/

|---app.py

|---model.pth

|---model.py

|---requirements.txt

|---Dockerfile
```

大体就是这个结构，app.py其实就涉及到了上面说的，API的设计和暴露，我们通过web应用框架flask来暴露API，API的设计可以按照自己想法设计，这里就是post方式，用上传表单数据的方式，访问部署服务器5000端口下的predict目录，返回json格式的`{'class': pred.item()}`,完整代码如下：

```python
from flask import Flask, request
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import CNN, convert_to_fp16

app = Flask(__name__)
model = CNN().cuda()
model.load_state_dict(torch.load('model.pth'))
model = convert_to_fp16(model).eval()  # 启用FP16模式

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return {'error': 'No image uploaded'}, 400
    
    image = Image.open(request.files['image']).convert('RGB')
    with torch.no_grad(), torch.cuda.amp.autocast():  # FP16自动混合精度
        tensor = transform(image).unsqueeze(0).cuda()
        output = model(tensor.half())  # 输入转为FP16
        pred = torch.argmax(output.float(), dim=1)  # 输出转FP32计算
    
    return {'class': pred.item()}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

```

正因为是测试阶段，所以直接用flask启动一个web服务，这当然是不能在生产中使用的，因为是这个flask自带的web服务即使开启多线程模式，也只是低并发服务器，且没有安全防护机制，不能用于生产部署。

测试使用了FP16半精度推理，加速推理速度和减少显存消耗。

至于`requirements.txt`就包含我们用到的第三方库即可，比如torchvision和flask。然后我们编写`Dockerfile`，生成一个测试镜像：

```

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

运行`docker build -t pytorch-classifier:fp16 .`构建测试镜像，`pytorch`因为我们只需要运行，不用来开发，所以选择runtime版本即可，即使这样，这个基础镜像也有3GB的大小，如果遇到网络问题就使用镜像源或者找到资源下载到本地后，使用`docker load -i` 进行本地加载镜像。镜像构建后有将近14个GB的大小：

![](C:\Users\kurumi\Desktop\post\113.jpg)

镜像生成后，启动容器：

`docker run --gpus all -p 5000:5000 --rm pytorch-classifier:fp16`

这里需要注意，要安装好`Nvidia Container Toolkit`才能在容器中使用cuda，具体安装方法见官网即可，基于wsl2的windows docker desktop的话，只需要配置文件加个runtime块就行。

启动容器后本地直接用`curl -X POST`就能测试，比如：

`curl -X POST -F "image=@test.jpg" http://localhost:5000/predict`就能将图片用表单数据的方式post给服务器，一般post传输图片有2种方式，一种就是这种用表单数据的方式，另一种就是base64编码的形式。为了能看到返回结果，可以再上面curl命令后加上`| jq`工具来查看json格式数据。

我的测试图片是一张小狗图片：

![](C:\Users\kurumi\Desktop\post\test.jpg)

测试结果：

![](C:\Users\kurumi\Desktop\post\112.jpg)

```
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
```

CIFAR10的类别如上所示，类别5就是dog。

测试要测试多个维度，不单单是模型的准确性，还要考虑模型的鲁棒性，抗干扰和异常的能力，还有延迟和吞吐量，不同部署终端硬件性能发挥等等。

```python
import requests
import time

def benchmark(url, image_path, rounds=100):
    total_time = 0
    with open(image_path, 'rb') as f:
        file = {'image': f}
        for _ in range(rounds):
            start = time.time()
            r = requests.post(url, files=file)
            total_time += time.time() - start
    print(f'Average latency: {total_time/rounds*1000:.2f}ms')

benchmark('http://localhost:5000/predict', 'test.jpg')

```

继续用这个测试程序可以进一步测试模型预测的平均延迟，更多的测试需要开发者在实际生产项目中全面测试。



### 生产部署

docker测试镜像如果测试的没有问题了，就可以开始真正的部署docker容器了，需要注意以下的改变：

- 实际生产中模型为了提高鲁棒性，需要增加更多的校验和异常处理，甚至可能多模型融合

- 更合理的API设计

- 用工业级WSGI服务器运行flask，比如uWSGI，Gunicorn

- 用nginx反向代理和负载均衡，由nginx暴露最终端口

- 使用opencv处理图像，而不是PIL，因为opencv速度更快且支持cuda加速

- 使用多阶段构建docker，构建阶段和生产运行阶段分离

- 要对项目持续的监控，持续维护，比如增加Prometheus监控指标

#### 使用Gunicorn和nginx

app.py还是原有逻辑即可，只是不需要开启自己的测试服务器了，还有API返回值的设计可以更加合理一点，比如：

```
{
  "predicted_class": 3,
  "class_name": "cat",
  "confidence": 0.92,
  "all_probabilities": {
    "0": 0.01, "1": 0.02, ..., "9": 0.001
  }
}
```

可以这样回复更多的信息，让调用者有更灵活的操作空间。

配置一个服务启动shell，同时启动Gunicorn和nginx服务：

```
#!bin/bash
# 启动Gunicorn
gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 600 app:app &
# 启动Nginx
nginx -g "daemon off;"
```



因为是在容器中运行，所以nginx不需要后台运行，改在前台运行。gunicorn接管flask的运行，`app:app`分别表示`flask文件名:flask实例名`,启用4个worker处理并发。

然后配置nginx配置文件示例：

```
 {
    listen 80;
    server_name localhost;

    location / {
        proxy_pass http://localhost:5000;
        ...
    }

    # 静态文件缓存
    location /static/ {
        alias /app/static/;
        ...
    }
        ...
}


```

监听80端口，也就是暴露给调用方的接口端口，内部5000端口除了开发部署者，无从得知，更加安全，且还可以配置负载均衡和反向代理。



#### 分段构建

```

# 基础镜像
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# 生产镜像
FROM nvidia/cuda:11.7.1-base
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app .
COPY . .

# 安装Nginx和Gunicorn
RUN apt-get update && apt-get install -y nginx && rm -rf /var/lib/apt/lists/*
RUN pip install gunicorn==20.1.0

# 配置环境
ENV PATH=/root/.local/bin:$PATH
EXPOSE 80
CMD ["sh", "start_server.sh"]

```

pytorch运行仅需：

- CUDA Runtime API（`libcudart.so`）

- cuDNN加速库（`libcudnn.so`）

- NVIDIA驱动兼容层‌

而基础镜像`pytorch/pytorch`除了包含上面，还包含：

- 开发工具链（gcc, cmake）

- 训练用依赖库（如OpenMPI）

- 调试工具‌ (gdb)

而生产镜像是Nvidia官方基础镜像，已经包含：

- GPU驱动兼容层

- CUDA Runtime API

- 必要的GLIBC库‌

因此只需要运行文件和包括pytorch在内的python第三方库，再加上nginx和Gunicorn就可以完整部署这个项目了。

所以使用`COPY --from=builder /root/.local /root/.local`可以将包括pytorch在内的python第三方库的二进制文件拷贝到运行环境下，不包含构建工具。

```

/root/.local/
├── bin/         # 用户级可执行文件
├── lib/         # Python库安装目录
│    └── python3.8/
│        └── site-packages/  # 所有pip安装的包
 └── share/       # 资源文件
```

而使用`COPY --from=builder /app .`则会将运行程序，服务启动文件，配置文件等拷贝到新环境。接着新环境安装Gunicorn和nginx之后，启动服务即可。

经过分段构建后的镜像体积可以缩小`70%`，然后正常启动容器即可完成产品级的部署。



### 其他

- 如果是多机或者集群推理的话，可能需要使用到kubernetes

- 推理不一定非要依赖pytorch，可以转换模型为onnx，然后生产部署使用onnxruntime-gpu推理
