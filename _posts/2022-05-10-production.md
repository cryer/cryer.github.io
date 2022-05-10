---
layout: post
title: AI产品级部署
description: AI 产品级部署

---

### 导入

几年前写过一篇博客讲的是如何发布产品，提到了docker的基本用法，也用一个简单的例子讲解了docker的多阶段构建。今天就详细展开说说如何利用docker产品级，工业级的部署一个AI项目。

我个人其实对AI的工程研究程度进行了简单的分级，入门阶段是使用非常干净的数据集，比如机器学习入门可能接触的第一个数据集，iris鸢尾花数据集，直接拿来用，利用最基本的机器模型或者自己定义的稍微复杂一点的机器模型来学习，这个阶段还称不上工程。稍微深入一点的阶段，是有一定的数据工程能力，能够对现实的总是不那么干净数据集进行分析和清理，并且可以对数据集进行训练集验证集测试集的划分，掌握交叉验证，然后利用更复杂的模型，比如深层网络模型或者近年来流行的LLM进行训练和微调，还要考虑一定的模型鲁棒性。更加深入一点的阶段，除了有更强的数据工程能力，还需要掌握以下几点：

- logging日志的高效使用，不仅是开发日志，还包括运行时候的日志

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

同样没什么好说的，简单训练2个epoch做个样子，然后保存模型，之后就不需要这个训练的代码了。
对于这个简单的例子来说，开发时的logging确实没什么必要，但是运行时的logging还是需要的，下面会简单介绍如何维护运行时日志。

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

![](https://github.com/cryer/cryer.github.io/raw/master/image/113.jpg)

镜像生成后，启动容器：

`docker run --gpus all -p 5000:5000 --rm pytorch-classifier:fp16`

这里需要注意，要安装好`Nvidia Container Toolkit`才能在容器中使用cuda，具体安装方法见官网即可，基于wsl2的windows docker desktop的话，只需要配置文件加个runtime块就行。

启动容器后本地直接用`curl -X POST`就能测试，比如：

`curl -X POST -F "image=@test.jpg" http://localhost:5000/predict`就能将图片用表单数据的方式post给服务器，一般post传输图片有2种方式，一种就是这种用表单数据的方式，另一种就是base64编码的形式。为了能看到返回结果，可以再上面curl命令后加上`| jq`工具来查看json格式数据。

我的测试图片是一张小狗图片：

![](https://github.com/cryer/cryer.github.io/raw/master/image/test.jpg)

测试结果：

![](https://github.com/cryer/cryer.github.io/raw/master/image/112.jpg)

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

app.py还是原有逻辑即可，只是API返回值的设计可以更加合理一点，比如：

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

配置一个服务启动shell，同时启动Gunicorn和nginx服务，`start_server.sh`：

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
 server {
    listen 80;
    server_name localhost;
    client_max_body_size 20M;  # 允许大文件上传
    proxy_read_timeout 300s;‌

    location / {
        proxy_pass http://localhost:5000;
        ... # 如果你需要其他配置，比如proxy_set_header之类
    }
}

```

监听80端口，也就是暴露给调用方的接口端口，内部5000端口除了开发部署者，无从得知，更加安全，且还可以配置负载均衡等。这里只配置了通用的根目录，可以捕获到所有形式的API请求，但是后台flask框架只会处理/predict就是，如果需要更多的功能，修改配置文件增加更多的配置或者location即可，比如location /static/之类，或者更强壮的API，增加不同模型的接口，比如location /object_detection/配置目标检测的API接口，当然对应的flask中需要增加/object_detection的处理，内部调用目标检测的模型进行推理，而不是图像分类模型。
上面只是http的server中的配置，为了生产中更好的处理API请求，还需要配置`worker_connections , worker_processes, use epoll`等，`worker_connections`表示工作进程数（通常等于CPU核心数），需要配置在nginx配置文件的全局位置，`worker_processes, use epoll`表示单个工作进程并发连接上限和使用epoll端口复用，需要配置在`events`块中，大致如下：
```
# /etc/nginx/nginx.conf 主配置文件

# [全局配置域] - 直接位于配置文件顶部
worker_processes 4;           # 工作进程数
worker_rlimit_nofile 65535;   # 进程最大文件描述符数

# [events块] - 必须显式声明
events {
    worker_connections 4096;  # 单个工作进程并发连接上限 
    use epoll;                # epoll端口复用
    multi_accept on;          # 一次性接受所有新连接 
}
http{
...
server{
...
}
...
}
```

具体更多的nginx配置就不再赘述，可以自行了解，整套系统主要的流程如下：
```
 Client->>Nginx: POST /predict (image)
  Nginx->>Gunicorn: 转发至 5000 端口
  Gunicorn->>Flask: 调用预测接口
  Flask->>PyTorch: 执行推理
  PyTorch-->>Flask: 返回预测结果
  Flask-->>Gunicorn: JSON响应
  Gunicorn-->>Nginx: 传回代理
  Nginx-->>Client: 返回分类结果
```
应该足够清晰了。

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

经过分段构建后的镜像体积可以缩小`70%`，然后正常启动容器即可完成产品级的部署，启动容器时需要将目录下的nginx配置文件挂载到镜像容器中的`/etc/nginx`目录下，比如` -v $(pwd)/nginx-conf:/etc/nginx/conf.d`, 挂载自定义配置。

### 监控
项目部署完成后，基本生产部署是完成了，但是整体的生产工作只完成了一半，还需要对项目进行持续的维护，维护首先就需要监控。可以借助开源的工业级监控库prometheus，对于python而言，就是第三方库`prometheus-client`,但是因为我们是使用flask提供API接口，则可以使用另一个专门用于flask的集成库`prometheus-flask-exporter`，prometheus有多种监控类型，比如计数，仪表盘，直方图，摘要等，可以监控多种指标，比如http的基本连接情况等，而prometheus社区维护的开源库`node-exporter`则可以监控服务器的cpu，内存，磁盘占用等硬件信息，两者可以配合使用。

**这个监控和我前面说的运行时日志还不太一样，可以认为是运维的两个部分，前者主要是监控，而运行时的日志主要是记录。监控主要监控硬件指标和我们预定义的特定软件和网络指标，不关心程序本身的运行情况，而运行时日志就负责记录程序运行状态，如果出错也会记录下来，方便后期排查程序性错误。运行时日志就使用标准库logging即可，每个模块设置一个Logger记录器和多个Handler处理器，比如警告WARNING以上日志记录到文件处理器，重大错误CRITICAL日志也可以使用SMTP处理器直接发送邮件。logging和监控都是必要的，要配合使用，一起维护项目的生产运行。关于logging这里就不再多说，知道其重要性即可，下面主要说明监控**

简单举例：
```
from flask import Flask
from prometheus_flask_exporter import PrometheusMetrics

app = Flask(__name__)
metrics = PrometheusMetrics(app)
```
`PrometheusMetrics(app)`自动将默认的HTTP指标绑定到app上,默认指标主要就是http连接的延迟和连接的总数等。并且会自动创建一个 /metrics 路由，因此只要访问`http://localhost:5000/metrics`就能访问这些默认的监控指标，这样我们就实现了一个简单的监控。我们可以添加自定义的指标，以计数为例，`metrics.counter('classification_count', 'Number of classifiedimages').inc()`来监控图像分类的已分类图片数量等等，可以设置任何自己想要的监控指标。
关于更多的prometheus的用法自行了解即可，为了部署带有监控能力的镜像，我们就需要对之前的生产镜像进行修改，首先就是app.py，也就是flask接口文件，添加上述的默认或者自定义指标，然后把`prometheus_flask_exporter`库加入`requirements.txt`中即可。
除此之外，因为是AI项目，所以我们还想监控服务器的硬件信息，也就是利用`node-exporter`，要添加如此多的内容，之前的简单的docker构建单一容器镜像的方式显然已经不适用，我们需要利用`docker compose`，组合多个容器，提高镜像的鲁棒性和可维护性。并且既然已经使用compose了，那之前的Nginx也顺便要单独拉取镜像进行组合，而不是在生产镜像中`apt install`下载，而gunicorn因为和flask其实是统一的，应用服务器和应用框架，没有分离的必要，因此也不大需要单独拉取镜像。
#### 编写配置文件
- prometheus.yml
接下来需要编写prometheus的配置文件，用来设置监控时间间隔，指明监控任务，包括`node-exporter`。
```
 global:
     scrape_interval: 15s # 每15秒拉取一次指标
scrape_configs:
    # 任务一：抓取Flask应用的指标
     -job_name: 'flask-app'
           static_configs:
             - targets: ['app:5000'] # 'app'是docker-compose中的应用服务名
    # 任务二：抓取服务器硬件的指标
         - job_name: 'node-exporter'
           static_configs:
             - targets: ['node-exporter:9100'] # 'node-exporter'是其服务名
```
- docker-compose.yml
  然后就是编写docker compose的配置文件,组合多个容器，包括flask应用容器，nginx容器，prometheus容器，node-exporter容器：
  ```
    version: '0.1'
    services:
       # flask应用
       app:
         build: .
         container_name: my_image_classifier_app
         restart: always
         command: gunicorn --workers 4 --bind 0.0.0.0:5000 app:app
        expose:
          - 5000
   
      # Nginx反向代理
      nginx:
        image: nginx:latest
        container_name: nginx_proxy
        restart: always
        ports:
          - "80:80" # 将主机的80端口映射到Nginx
        volumes:
          - ./nginx.conf:/etc/nginx/nginx.conf:ro # 挂载Nginx配置
        depends_on:
          - app
  
      # Prometheus服务
      prometheus:
        image: prom/prometheus:latest
        container_name: prometheus_server
        restart: always
        ports:
          - "9090:9090"
        volumes:
          - ./prometheus.yml:/etc/prometheus/prometheus.yml
        depends_on:
          - app
          - node-exporter
   
      # Node Exporter服务
      node-exporter:
        image: prom/node-exporter:latest
        container_name: node_exporter_metrics
        restart: always
        command:
          - '--path.rootfs=/host'
        pid: host
        volumes:
          - '/:/host:ro,rslave' # 将主机根文件系统只读挂载，以获取磁盘信息
        expose:
          - 9100
  ```
接下来启动所有docker compose的服务即可完成部署，使用命令`docker-compose up -d`，当然还有一点，之前的flask的Dockerfile中的CMD启动服务shell，同时启动Gunicorn和nginx服务的命令就不需要了，因为compose配置文件中已经执行了，注释掉即可。

**访问API**
  - `http://localhost`即可访问flask应用，因为http默认端口就是80，和nginx监听端口一致，配合POST方法传输图片表单就可以进行图像分类并获取结果
  -  Prometheus:` http://localhost:9090`，Prometheus UI中，进入 "Status" -> "Targets"，能看到 flask-app 和 node-exporter 两个任务，状态都是 "UP"

#### 进一步
前面使用Prometheus只用到了**自动监控指标 + 可视化查看指标**，而Prometheus还有一个功能，那就是**报警功能**，设置某一指标数据异常的时候进行报警，既可以简单在报警的端口查看报警信息，也可以设置邮件形式发送报警信息到对应邮箱。关于Prometheus报警的配置方法，简单说明，不深入讲解。
主要就是2点：
- 设置报警规则
- 配置报警管理器，Alertmanager也是一个容器
  
首先就是在`prometheus.yml`配置文件中，增加报警规则的路径设置和报警管理器的配置：
```
# 报警规则文件路径
    rule_files:
       - /etc/prometheus/alert.rules.yml

     # 报警管理器配置
    alerting:
          alertmanagers:
            - static_configs:
                - targets:
                  - alertmanager:9093 # 指向Alertmanager容器
```
类似这样，然后创建具体的报警规则，编写`alert.rules.yml`配置文件：
```
rules:
    - alert: HighCpuUsage
    expr: 100 - (avg by (instance)
      (rate(node_cpu_seconds_total{mode="idle"}[1m])) * 100) > 80
    for: 5m
    labels:
        severity: warning
    annotations:
        summary: "Instance {{ $labels.instance }} CPU usage high"
        description: "CPU usage is above 80% for 5 minutes."
```
这就是一个基本的报警规则编写，当服务器持续5分钟CPU占用超过80%时，触发报警。
继续编写`config.yml`报警管理器的配置文件,同样简单示例：

```
     global:
    resolve_timeout: 5m

    route:
     group_by: ['alertname', 'cluster', 'service']
     group_wait: 30s
     group_interval: 5m
    repeat_interval: 3h
    receiver: 'null' # 默认接收者

 receivers:
     - name: '通知的邮箱' 
```

最后在docker compose配置文件中，加上Alertmanager服务：

```
 alertmanager:
    image: prom/alertmanager:latest
    container_name: alertmanager
    restart: always
    ports:
     - "9093:9093"
    volumes:
      - ./config.yml:/etc/alertmanager/config.yml
```

**API**
Alertmanager: `http://localhost:9093`，在这里看到触发的报警。

为了监控指标显示信息更加的直观，还可以配合使用Grafana，Grafana是一个可视化的观测平台，可以接受Prometheus的监控指标数据。Grafana的使用很简单，也有官方的docker镜像，只需要在docker compose配置文件中添加,拉取官方镜像，配置端口映射和挂载卷，然后依赖prometheus即可，比如：
  ```
   # Grafana服务
      grafana:
      image: grafana/grafana:latest
      container_name: grafana_dashboard
      restart: always
      ports:
      - "3000:3000"
      volumes:
           - grafana_data:/var/lib/grafana # 持久化Grafana数据
      depends_on:
          - prometheus
volumes:
    grafana_data: {}
```
**API**
接着使用默认用户名/密码: admin/admin，登录Grafana: `http://localhost:3000`，UI界面中添加Prometheus数据源，地址选择Prometheus的地址`http://prometheus:9090`，然后保存。最后选择你想要显示的仪表盘的样式，可以从grafana.com查找，就可以非常直观的用导入的仪表盘样式，查看监控数据了。

### 总体架构
所以现在的总体架构是这样的：

- Flask App : AI项目的内部API接口，使用Gunicorn运行，集成 prometheus_flask_exporter，通过 /metrics端点暴露HTTP请求相关的指标
- Node Exporter: 作为一个独立的容器运行，暴露服务器的CPU、内存、磁盘等硬件指标
- Prometheus: 核心监控服务器。定期从Flask应用和Node Exporter的 /metrics  拉取（scrape）指标数据并存储
- Grafana:数据可视化平台。连接到Prometheus数据源，通过丰富的仪表盘（Dashboard）展示监控数据
- Alertmanager: 报警处理中心。Prometheus根据预设的报警规则将报警发送给Alertmanager，后者负责去重、分组并发送通知
- Nginx: 作为反向代理，和整个AI项目的外部暴露API接口，将外部请求转发给Gunicorn运行的Flask应用

### 其他

- 如果是多机或者集群推理的话，可能需要使用到kubernetes，kubernetes配合prometheus也属于是工业级的标准

- 推理不一定非要依赖pytorch，可以转换模型为onnx，然后生产部署使用onnxruntime-gpu推理

- 虽然上面从开发到测试到部署到监控，都是手动完成的，但是实际上生产中都会设计成pipline自动流水线，并且开发流水线和部署流水线分开，也就是`CI/CD`持续集成和持续部署的思想。`CI`简单来说就是每次`push`以及协作开发者的`PR`都会自动触发集成流水线，包含自动拉取本次的提交内容，自动运行测试（包括代码质量测试，比如pylint测试，单元测试，比如unittest和pytest，集成测试，也就是整体功能和性能的测试），通过后自动合并代码，代码会严格分类成主分支，开发分支，测试分支等。然后`CD`持续部署流水线则是自动从主分支拉取最新项目代码，自动根据Dockerfile编译生成最新镜像，然后运行一些基本的部署的测试，最后自动推送到远程镜像库中。`Github Actions`就是一个很方便的不需要借助第三方的`CI`工作流工具，感兴趣的读者可以去仔细了解下。


