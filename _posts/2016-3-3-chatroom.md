---
layout: post
title: 简单匿名聊天室的实现
description: 简单匿名聊天室的实现

---

### 导入

聊天室的架构还是常见的C/S架构，匿名聊天室的特点主要有三点：

- 匿名性：连接时分配随机ID（如"用户123"）

- 实时性：采用I/O多路复用技术

- 轻量级：单线程处理所有连接

为什么不采用多线程？因为多线程的开销很大，对于小项目无所谓，如果是针对“C10K”级别甚至以上的连接的大项目，多线程很难实现。即使采用线程池也是一样，线程池只是节省了线程创建和销毁的开销，但是节省不了线程切换的开销，而线程切换恰恰是线程模型性能损耗比较大的地方。

而多路IO复用，比如select，poll，epoll相比线程模型来说就轻量又实时，特点就是高并发非常擅长，因为是内核事件驱动。但是也有缺点，就是连接活跃度高，如果是消息传输频率非常高的场景，多线程或许更好，因为线程模型虽然并发低，但是连接活跃度很高。

### 实现

#### 服务器端

服务器端只需要监听socket描述符，然后针对不同的事件做不同的处理，事件主要是POLLIN事件，代表输入或者说写事件，服务器端socket描述符发生写事件，代表新连接，接收新连接，分配匿名，加入监听描述符列表即可。客户端socket描述如发生POLLIN写事件代表客户端数据传入，服务器只需要接收消息，加上匿名前缀，然后广播即可。



```c
#include <stdio.h>
#include <sys/socket.h>
#include <poll.h>
#include <netinet/in.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define MAXCLIENTS 10
#define PORT 8888

typedef struct {
    int fd;
    char name[16];
}Client;
Client clients[MAXCLIENTS];

void random_name(char *name){
    const char charset[] = "abcdefghijklmnopqrstuvwxyz123456789";
    for (int i=0;i<5;i++){
        name[i] = charset[rand() % (sizeof(charset) - 1)];
    }
    name[5]='\0';
    char prefix[] = "User-";
    strcat(prefix, name);
    strcpy(name, prefix);
}


int main()
{
    srand(time(NULL));
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr = {AF_INET, htons(PORT), 0};
    bind(sockfd,(struct sockaddr*)&addr, sizeof(addr));
    listen(sockfd, MAXCLIENTS);

    struct pollfd fds[MAXCLIENTS + 1];
    fds[0].fd = sockfd;
    fds[0].events = POLLIN;
    int nfds = 1;

    while(1){
        char buffer[256] = {0};
        char msg[300] = {0};
        int ready = poll(fds, nfds, 50000);
        for(int i=0; i<nfds; ++i){
            if (fds[i].revents & POLLIN){
                if (fds[i].fd == sockfd){
                    int clientfd = accept(sockfd, NULL, NULL);
                    clients[nfds-1].fd = clientfd;
                    random_name(clients[nfds-1].name);
                    fds[nfds].fd = clientfd;
                    fds[nfds++].events = POLLIN;
                    sprintf(msg, "%s join in...", clients[nfds-2].name);
                    for (int j=1; j<nfds; ++j){
                       send(fds[j].fd, msg, 300, 0);
                    }
                }else{
                    if (recv(fds[i].fd, buffer, 255, 0) <= 0){
                        sprintf(msg, "%s leave out...", clients[i-1].name);
                        close(fds[i].fd);
                        clients[i-1] = clients[nfds-2];
                        fds[i] = fds[--nfds];
                        for (int j=1; j<nfds; ++j){
                           send(fds[j].fd, msg, 300, 0);
                        }
                        break;
                    }
                    sprintf(msg, "%s:%s", clients[i-1].name, buffer);
                    for (int j=1; j<nfds; ++j){
                       send(fds[j].fd, msg, 300, 0);
                    }
                }
            }
        }
    }
    return 0;
}


```

#### 客户端

客户端就更简单，只需要监听stdin标准输入和服务器的socket描述符即可，标准输入发生POLLIN事件，代表你输入了消息，读取并把消息发给服务器即可，同样，服务器socket描述符发生了POLLIN事件，代表接收到服务器消息，直接打印在标准输出就行。

```c
#include <stdio.h>
#include <sys/socket.h>
#include <poll.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>

#define PORT 8888

int main()
{
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr = {AF_INET, htons(PORT), 0};
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr); 
    connect(sockfd, (struct sockaddr*)&addr, sizeof(addr));

    struct pollfd fds[2];
    fds[0].fd = sockfd;
    fds[0].events = POLLIN;
    fds[1].fd = 0;
    fds[1].events = POLLIN;
    int nfds = 2;

    while(1){
        char buffer[256] = {0};
        int ready = poll(fds, nfds, 50000);
        if (fds[1].revents & POLLIN){
            read(0, buffer, 255);
            send(sockfd, buffer, 255, 0);
        }else if (fds[0].revents & POLLIN){
            if (recv(sockfd, buffer, 255, 0) == 0){
                return 0;
            }
            printf("%s\n", buffer);
        }
    }
    return 0;
}

printf("%s\n", buffer);
        }
    }
    return 0;
}


```

如果服务器端部署在公网IP下的服务器上，使用对应的公网ip即可，只是需要注意开放防火墙对应端口。



### 浏览器聊天室

既然写到匿名聊天时了，就想着更加实用一点，每个客户端都要一个专门可执行文件也不太方便，索性直接部署到浏览器上吧，浏览器直接访问，也更加方便。针对浏览器有比socket更加方便的协议，那就是websocket，websocket基于TCP协议，但是利用了HTTP协议的握手部分，所以可以兼容，但是websocket和HTTP除此之外并没有什么关系，一些读者可能会认为websocket协议是基于HTTP协议的，其实不然。

HTTP是请求响应形式的短连接，一次请求，一次响应，虽然后面的标准可以多次响应，但是还是属于短连接，而websocket是持久连接。HTTP数据推送是轮询的方式，是半双工，而websocket最大的特点是全双工，且服务器可以主动向客户端推送数据，而HTTP只能响应请求，无法主动推送。

websocket就简单介绍到这，利用websocket实现服务器大致是这样：

```python

import asyncio
import websockets
import random
import json

connected_clients = {}

COLORS = ['#FF5733', '#33FF57', '#3357FF', '#F033FF', '#33FFF5']

async def handle_client(websocket, path):
    username = f"用户{random.randint(1000,9999)}"
    color = random.choice(COLORS)
    connected_clients[websocket] = {"username": username, "color": color}
    
    try:
        async for message in websocket:
            msg_data = {
                "username": username,
                "color": color,
                "message": message
            }
            print(f"Received: {msg_data}")
            await broadcast(json.dumps(msg_data))
    finally:
        del connected_clients[websocket]

async def broadcast(message):
    if connected_clients:
        tasks = [asyncio.create_task(client.send(message)) 
                for client in connected_clients]
        await asyncio.gather(*tasks)

start_server = websockets.serve(handle_client, "0.0.0.0", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()



```

利用python的asyncio异步配合websocket非常方便。

客户端因为浏览器访问，索然自然使用js来编写，且直接嵌入到html中，可以直接打开访问或者服务器利用nginx之类部署http服务 。

```html
<DOCTYPE html>
<html>
<head>
    <title>Chat Room</title>
    <style>
        body { font-family: Arial; max-width: 600px; margin: 0 auto; }
        #messages { height: 300px; border: 1px solid #ccc; overflow-y: scroll; padding: 10px; }
        #messageInput { width: 80%; padding: 8px; }
        #sendButton { padding: 8px 15px; }
        .message { display: flex; margin: 5px 0; align-items: center; }
        .avatar { width: 30px; height: 30px; border-radius: 50%; margin-right: 10px; }
        .username { font-weight: bold; margin-right: 5px; }
    </style>
</head>
<body>
    <h1>匿名聊天室</h1>
    <div id="messages"></div>
    <input type="text" id="messageInput" placeholder="Type your message...">
    <button id="sendButton">Send</button>

    <script>
        const ws = new WebSocket(`ws://${window.location.hostname}:8765`);
        const messages = document.getElementById('messages');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');

        ws.onmessage = (event) => {
            const msgData = JSON.parse(event.data);
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message';
            
            const avatar = document.createElement('div');
            avatar.className = 'avatar';
            avatar.style.backgroundColor = msgData.color;
            
            const username = document.createElement('span');
            username.className = 'username';
            username.textContent = msgData.username + ':';
            
            const content = document.createElement('span');
            content.textContent = msgData.message;
            
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(username);
            messageDiv.appendChild(content);
            messages.appendChild(messageDiv);
            messages.scrollTop = messages.scrollHeight;
        };

        sendButton.onclick = () => {
            if (messageInput.value) {
                ws.send(messageInput.value);
                messageInput.value = '';
            }
        };

        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendButton.click();
            }
        });
    </script>
</body>
</html>


```

一个非常简单的html界面，配合js使用websocket监听键盘事件，然后发送给websocket服务器，服务器接收消息并且广播。

然后你就可以得到这样的效果：

![](https://github.com/cryer/cryer.github.io/raw/master/image/1.jpg)

### 总结

IO多路复用，以及websocket协议的特点和使用。


