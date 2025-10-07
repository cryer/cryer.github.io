---
layout: post
title: 实现一个文件传输工具
description: 实现一个文件传输工具

---

### 导入

实现一个简单的双向跨平台文件传输工具，使用自定义的简单文件传输协议，具有如下特性：

- 双向传输，客户端和服务端集成在一起

- 跨平台，支持windows和类unix系统

- 大文件分块传输

- 支持目录传输

- 支持断点续传

**传输协议**：先发送文件名（最大255字节），再发送文件大小（8字节，`long long`），然后发送文件内容。



简单说明部分思路：

- 目录传输
  
  - 客户端检测参数是否为目录
  
  - 若是目录，则调用 `tar -cf archive.tar 目录名` 打包 (windows需要先下载tar)
  
  - 传输 `archive.tar` 文件
  
  - 服务器接收后，若文件名以 `.dir.tar` 结尾，则自动调用 `tar -xf` 解压
  
  - 删除临时 `.tar` 文件（可选）

- 断点重续
  
  - 客户端连接后，先发送文件名
  
  - 服务器检查本地是否存在同名文件，不存在 → 返回 0，存在但大小 ≥ 客户端文件大小 → 返回 -1（视为已完成），存在但大小 < 客户端文件大小 → 返回当前大小（断点位置）
  
  - 客户端从断点位置开始发送数据

### 完整代码

```c
//file_transfer.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #include <direct.h>
    #pragma comment(lib, "ws2_32.lib")
    typedef int socklen_t;
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <sys/wait.h>
    #define SOCKET int
    #define INVALID_SOCKET -1
    #define SOCKET_ERROR -1
    #define closesocket close
#endif

#define BUFFER_SIZE 8192
#define _FILENAME_MAX 512
#define DIR_SUFFIX ".dir.tar"

void error(const char* msg) {
    perror(msg);
    exit(1);
}

#ifdef _WIN32
void cleanup_winsock() {
    WSACleanup();
}
#endif

// 执行系统命令（用于 tar 打包/解包）
int execute_command(const char* cmd) {
#ifdef _WIN32
    return system(cmd);
#else
    int status = system(cmd);
    return WEXITSTATUS(status);
#endif
}

// 判断路径是否为目录
int is_directory(const char* path) {
    struct stat statbuf;
    if (stat(path, &statbuf) != 0) return 0;
    return S_ISDIR(statbuf.st_mode);
}

// 获取文件大小
long long get_file_size(const char* filepath) {
    FILE* fp = fopen(filepath, "rb");
    if (!fp) return -1;
    fseek(fp, 0, SEEK_END);
    long long size = ftell(fp);
    fclose(fp);
    return size;
}

// 服务器：获取已接收的文件大小（用于断点续传）
long long get_existing_file_size(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) return 0; // 文件不存在
    fseek(fp, 0, SEEK_END);
    long long size = ftell(fp);
    fclose(fp);
    return size;
}

// 服务器模式：接收文件（支持断点续传 + 目录自动解压）
void server_mode(int port) {
    SOCKET server_fd, client_fd;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    char buffer[BUFFER_SIZE];
    char filename[_FILENAME_MAX];
    long long filesize, received = 0, resume_from = 0;
    FILE* fp;

#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        error("WSAStartup failed");
    }
    atexit(cleanup_winsock);
#endif

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET) {
        error("socket failed");
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt))) {
        error("setsockopt");
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        error("bind failed");
    }

    if (listen(server_fd, 3) < 0) {
        error("listen");
    }

    printf("Server listening on port %d...\n", port);

    if ((client_fd = accept(server_fd, (struct sockaddr*)&address, &addrlen)) < 0) {
        error("accept");
    }

    printf("Client connected.\n");

    // 接收文件名
    if (recv(client_fd, filename, _FILENAME_MAX, 0) <= 0) {
        error("Failed to receive filename");
    }
    printf("Receiving file: %s\n", filename);

    // 接收文件大小
    if (recv(client_fd, (char*)&filesize, sizeof(filesize), 0) <= 0) {
        error("Failed to receive filesize");
    }
    printf("File size: %lld bytes\n", filesize);

    // 断点续传：检查本地是否存在部分文件
    resume_from = get_existing_file_size(filename);
    if (resume_from > 0 && resume_from < filesize) {
        printf("Resuming from byte: %lld\n", resume_from);
    } else if (resume_from >= filesize) {
        printf("File already complete. Skipping.\n");
        long long zero = 0;
        send(client_fd, (char*)&zero, sizeof(zero), 0); // 通知客户端无需传输
        closesocket(client_fd);
        closesocket(server_fd);
        return;
    }

    // 发送断点位置给客户端
    if (send(client_fd, (char*)&resume_from, sizeof(resume_from), 0) < 0) {
        error("Failed to send resume position");
    }

    // 打开文件（追加模式）
    fp = fopen(filename, resume_from > 0 ? "ab" : "wb");
    if (!fp) {
        perror("fopen");
        closesocket(client_fd);
        closesocket(server_fd);
        exit(1);
    }

    received = resume_from;

    // 接收文件内容
    while (received < filesize) {
        int to_receive = (filesize - received > BUFFER_SIZE) ? BUFFER_SIZE : (int)(filesize - received);
        int n = recv(client_fd, buffer, to_receive, 0);
        if (n <= 0) {
            printf("\nConnection closed or error. Received %lld/%lld bytes.\n", received, filesize);
            break;
        }
        fwrite(buffer, 1, n, fp);
        received += n;
        printf("\rReceived: %lld / %lld bytes (%.1f%%)", received, filesize, received * 100.0 / filesize);
        fflush(stdout);
    }
    printf("\nFile received successfully: %s\n", filename);

    fclose(fp);
    closesocket(client_fd);
    closesocket(server_fd);

    // 如果是目录压缩包，自动解压
    if (strlen(filename) > strlen(DIR_SUFFIX) &&
        strcmp(filename + strlen(filename) - strlen(DIR_SUFFIX), DIR_SUFFIX) == 0) {
        printf("Detected directory archive. Extracting...\n");
        char cmd[1024];
        snprintf(cmd, sizeof(cmd), "tar -xf \"%s\"", filename);
        if (execute_command(cmd) != 0) {
            fprintf(stderr, "Failed to extract directory. Please extract manually.\n");
        } else {
            printf("Directory extracted successfully.\n");
            // 可选：删除临时 tar 文件
            remove(filename);
        }
    }
}

// 客户端：发送文件（支持断点续传 + 目录打包）
void client_mode(const char* ip, int port, const char* filepath) {
    SOCKET sock = 0;
    struct sockaddr_in serv_addr;
    char buffer[BUFFER_SIZE];
    long long filesize, sent = 0, resume_from = 0;
    FILE* fp;
    char filename[_FILENAME_MAX];
    char temp_tar[_FILENAME_MAX] = {0};

    int is_dir = is_directory(filepath);

    // 如果是目录，先打包
    if (is_dir) {
        const char* last_slash = strrchr(filepath, '/');
        const char* last_backslash = strrchr(filepath, '\\');
        const char* dirname = filepath;
        if (last_slash && last_backslash) {
            dirname = (last_slash > last_backslash) ? last_slash + 1 : last_backslash + 1;
        } else if (last_slash) {
            dirname = last_slash + 1;
        } else if (last_backslash) {
            dirname = last_backslash + 1;
        }
        snprintf(temp_tar, sizeof(temp_tar), "%s%s", dirname, DIR_SUFFIX);
        printf("Packing directory '%s' into '%s'...\n", filepath, temp_tar);

        char cmd[1024];
        snprintf(cmd, sizeof(cmd), "tar -cf \"%s\" \"%s\"", temp_tar, filepath);
        if (execute_command(cmd) != 0) {
            error("Failed to create tar archive");
        }
        filepath = temp_tar; // 后续操作此 tar 文件
        printf("Directory packed successfully.\n");
    }

#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        error("WSAStartup failed");
    }
    atexit(cleanup_winsock);
#endif

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        error("Socket creation error");
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);

    if (inet_pton(AF_INET, ip, &serv_addr.sin_addr) <= 0) {
        error("Invalid address / Address not supported");
    }

    if (connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        error("Connection Failed");
    }

    // 提取文件名
    const char* last_slash = strrchr(filepath, '/');
    const char* last_backslash = strrchr(filepath, '\\');
    const char* fname = filepath;
    if (last_slash && last_backslash) {
        fname = (last_slash > last_backslash) ? last_slash + 1 : last_backslash + 1;
    } else if (last_slash) {
        fname = last_slash + 1;
    } else if (last_backslash) {
        fname = last_backslash + 1;
    }
    strncpy(filename, fname, _FILENAME_MAX - 1);
    filename[_FILENAME_MAX - 1] = '\0';

    printf("Sending file: %s\n", filename);

    // 发送文件名
    if (send(sock, filename, strlen(filename) + 1, 0) < 0) {
        error("Failed to send filename");
    }

    // 获取文件大小
    filesize = get_file_size(filepath);
    if (filesize < 0) {
        perror("Cannot get file size");
        closesocket(sock);
        exit(1);
    }
    printf("File size: %lld bytes\n", filesize);

    // 发送文件大小
    if (send(sock, (char*)&filesize, sizeof(filesize), 0) < 0) {
        error("Failed to send filesize");
    }

    // 接收服务器返回的断点位置
    if (recv(sock, (char*)&resume_from, sizeof(resume_from), 0) <= 0) {
        error("Failed to receive resume position");
    }

    if (resume_from == -1 || resume_from >= filesize) {
        printf("File already exists on server. Skipping transfer.\n");
        closesocket(sock);
        if (is_dir) remove(temp_tar); // 清理临时文件
        return;
    }

    if (resume_from > 0) {
        printf("Resuming from byte: %lld\n", resume_from);
    }

    // 打开文件
    fp = fopen(filepath, "rb");
    if (!fp) {
        perror("fopen");
        closesocket(sock);
        exit(1);
    }

    // 跳转到断点位置
    if (resume_from > 0) {
        if (fseek(fp, resume_from, SEEK_SET) != 0) {
            perror("fseek");
            fclose(fp);
            closesocket(sock);
            exit(1);
        }
        sent = resume_from;
    }

    // 发送文件内容
    while (sent < filesize) {
        size_t to_send = (filesize - sent > BUFFER_SIZE) ? BUFFER_SIZE : (size_t)(filesize - sent);
        size_t n = fread(buffer, 1, to_send, fp);
        if (n == 0) break;

        if (send(sock, buffer, n, 0) < 0) {
            error("Failed to send file data");
        }
        sent += n;
        printf("\rSent: %lld / %lld bytes (%.1f%%)", sent, filesize, sent * 100.0 / filesize);
        fflush(stdout);
    }
    printf("\nFile sent successfully.\n");

    fclose(fp);
    closesocket(sock);

    // 清理临时 tar 文件（如果是目录）
    if (is_dir) {
        remove(temp_tar);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage:\n");
        printf("  Server: %s -s <port>\n", argv[0]);
        printf("  Client: %s -c <ip> <port> <file_or_directory_path>\n", argv[0]);
        printf("\nFeatures: Resume transfer, Directory transfer (auto tar/untar)\n");
        return 1;
    }

    if (strcmp(argv[1], "-s") == 0 && argc == 3) {
        int port = atoi(argv[2]);
        if (port <= 0 || port > 65535) {
            fprintf(stderr, "Invalid port number\n");
            return 1;
        }
        server_mode(port);
    }
    else if (strcmp(argv[1], "-c") == 0 && argc == 5) {
        const char* ip = argv[2];
        int port = atoi(argv[3]);
        const char* filepath = argv[4];
        if (port <= 0 || port > 65535) {
            fprintf(stderr, "Invalid port number\n");
            return 1;
        }
        client_mode(ip, port, filepath);
    }
    else {
        printf("Invalid arguments.\n");
        printf("Usage:\n");
        printf("  Server: %s -s <port>\n", argv[0]);
        printf("  Client: %s -c <ip> <port> <file_or_directory_path>\n", argv[0]);
        return 1;
    }

    return 0;
}
```

### 编译/运行/测试

编译：

- 类Unix系统：`gcc -o file_transfer file_transfer.c`

- windows系统（MinGW）：`gcc -o file_transfer.exe file_transfer.c -lws2_32`

测试发送目录（windows下为例）：

- 接收方：`file_transfer.exe -s 1234`

- 发送方（发送目录）：`file_transfer.exe  -c 127.0.0.1 1234 111` 111为目录名

结果：

- 发送方：

```
Packing directory '111' into '111.dir.tar'...
Directory packed successfully.
Sending file: 111.dir.tar
File size: 5014528 bytes
Sent: 5014528 / 5014528 bytes (100.0%)
File sent successfully.
```

- 接收方：

```
Server listening on port 1234...
Client connected.
Receiving file: 111.dir.tar
File size: 5014528 bytes
Received: 5014528 / 5014528 bytes (100.0%)
File received successfully: 111.dir.tar
Detected directory archive. Extracting...
Directory extracted successfully.
```



测试发送文件和断点重续（windows下为例）：

- 接收方：`file_transfer.exe -s 1234`

- 发送方：`file_transfer.exe -c 127.0.0.1 1234 test.pdf` 

结果：

- 发送方(`CTRL C中断传输`)：

```
Sending file: test.pdf
File size: 110306771 bytes
Sent: 73580544 / 110306771 bytes (66.7%)^C
```

然后继续传输`file_transfer.exe -c 127.0.0.1 1234 test.pdf`：

```
Sending file: test.pdf
File size: 110306771 bytes
Resuming from byte: 73580544
Sent: 110306771 / 110306771 bytes (100.0%)
File sent successfully.
```

- 接收方：

```
Server listening on port 1234...
Client connected.
Receiving file: test.pdf
File size: 110306771 bytes
Received: 73580544 / 110306771 bytes (66.7%)
Connection closed or error. Received 73580544/110306771 bytes.

File received successfully: test.pdf 
 
重新接收后： 
Server listening on port 1234...
Client connected.
Receiving file: test.pdf
File size: 110306771 bytes
Resuming from byte: 73580544
Received: 110306771 / 110306771 bytes (100.0%)
File received successfully: test.pdf
```

然后接收到的pdf文件可以正常打开。

### 注意点和改进方向

- 在判断是否为目录的函数中，利用了`<sys/stat.h>`头文件，这个头文件是类Unix系统中的头文件，这里windows因为使用的时MinGW，所以可以正常编译运行，如果想要使用MSVC的编译器，即CL编译器的话，这里就需要改进一下，比如：

```c
#ifdef _WIN32
#include <windows.h>

int is_directory(const char* path) {
    DWORD attrs = GetFileAttributesA(path);
    if (attrs == INVALID_FILE_ATTRIBUTES) {
        return 0; // 文件不存在或出错
    }
    return (attrs & FILE_ATTRIBUTE_DIRECTORY) != 0;
}

#else // Unix-like systems

#include <sys/stat.h>

int is_directory(const char* path) {
    struct stat statbuf;
    if (stat(path, &statbuf) != 0) return 0;
    return S_ISDIR(statbuf.st_mode);
}

#endif
```

CL编译命令：`cl file_transfer.c ws2_32.lib`



- 文件在传输过程中可能因网络抖动、缓冲区错误、磁盘写入失败等原因导致数据损坏,即使文件看似100%传输完成，也可能已经损坏，因此最好需要加上校验，用来验证文件完整性，如果检查文件不完整，则提示文件损坏，要求重传。比如利用`MD5校验`的简单思路：客户端计算本地MD5->发送服务器，服务器接收文件计算本地MD5->和客户端对比，断点重续则校验已有的数据部分。MD5有很多轻量级开源库实现，可以随意选择即可。

- 本传输工具采用明文传输，有安全风险，可以考虑使用`OpenSSL`加密传输

- 工具是一对一传输，可以考虑使用多线程同时接受多个线程的并发传输文件，既可以是多个客户端，也可以是同一个客户端但是多线程，比如一个大文件分割成线程数量的块，每个线程负责一个块的传输，可以充分利用多核的优势。



