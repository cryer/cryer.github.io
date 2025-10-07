---
layout: post
title: 实现一个简单的深度学习框架
description: 实现一个简单的深度学习框架

---

### 导入

之前博客介绍过，利用C++ cuda和Cython编写python扩展，实现了手动写矩阵乘法的kernel进而`cuBLAS`库中矩阵相乘函数`cublasSgemm`的调用，并且对比了执行速度。

现在就再进一步，使用`cudnn`和`cuBLAS`一起实现一个简单的深度学习库，包含卷积层，池化层，softmax+损失层，reLu激活层，全连接层的GPU端的前向和反向过程，其中全连接层就是一个矩阵乘法，直接利用`cuBLAS`的`cublasSgemm`通用矩阵乘法函数即可，其他层就利用`cudnn`提供的函数进行封装。

主要流程还是和之前python扩展一样：

- C++ cuda编写cu文件，然后nvcc编译成共享库

- cython编写`pyx`文件，封装cu文件中的函数，提供python接口，编写`setup.py`，然后生成cpp文件并且编译成共享库

- python调用函数，搭建网络，加载数据集，训练和测试模型即可。

nvcc单独编译成共享库，然后再cython封装可以，直接一起写到`setup.py`中也可以，上次博客采用的是前者，那这次就用后者。

还是主要提供代码，不过多讲解。

### 代码

**项目结构：**

```
├─cpp 
│ ├─layers.cu 
│ └─layers_wrapper.h 
└─python 
  ├─cnn_layers.pyx 
  ├─mnist_data 
  │ ├─t10k-images-idx3-ubyte.gz 
  │ ├─t10k-labels-idx1-ubyte.gz 
  │ ├─train-images-idx3-ubyte.gz 
  │ └─train-labels-idx1-ubyte.gz 
  ├─setup.py 
  ├─train_mnist.py 
  └─utils.py 
```

数据集就用简单的`MNIST`，cpp目录是后端，提供核心cuda编程和头文件，python目录是前端，封装cuda后端代码成共享库，提供接口，然后调用接口，搭建模型，加载数据，训练测是。

**环境要求：**

- NVIDIA GPU
- CUDA Toolkit (推荐 11.x 或更高版本)
- cuDNN 库 (需要与 CUDA 版本匹配)
- C++ 编译器 (如 g++)
- Python 3.x
- Cython (`pip install cython`)
- NumPy (`pip install numpy`)

#### 后端代码

- 核心cuda代码

```cpp
//layers.cu
#include <stdio.h>
#include <stdlib.h> 
#include <cudnn.h>
#include <cublas_v2.h>

#define THREADS_PER_BLOCK 256

#define cudaSafeCall(err)      __cudaSafeCall(err, __FILE__, __LINE__)
inline void __cudaSafeCall(cudaError_t err, const char *file, const int line) {
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

#define cudnnSafeCall(err)      __cudnnSafeCall(err, __FILE__, __LINE__)
inline void __cudnnSafeCall(cudnnStatus_t err, const char *file, const int line) {
    if (CUDNN_STATUS_SUCCESS != err) {
        fprintf(stderr, "cuDNN Error at %s:%d - %s\n", file, line, cudnnGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

const char* __cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "Unknown cuBLAS error";
}

#define cublasSafeCall(err)      __cublasSafeCall(err, __FILE__, __LINE__)
inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line) {
    if (CUBLAS_STATUS_SUCCESS != err) {
        fprintf(stderr, "cuBLAS Error at %s:%d - %s\n", file, line, __cublasGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


cudnnHandle_t& get_cudnn_handle() {
    static cudnnHandle_t cudnn_handle;
    static bool initialized = false;
    if (!initialized) {
        cudnnSafeCall(cudnnCreate(&cudnn_handle));
        // 注册一个函数，在程序正常终止时调用，清理资源
        atexit([]{ cudnnDestroy(cudnn_handle); });
        initialized = true;
    }
    return cudnn_handle;
}

cublasHandle_t& get_cublas_handle() {
    static cublasHandle_t cublas_handle;
    static bool initialized = false;
    if (!initialized) {
        cublasSafeCall(cublasCreate(&cublas_handle));
        atexit([]{ cublasDestroy(cublas_handle); });
        initialized = true;
    }
    return cublas_handle;
}

// =================================================================================
// ** 卷积层 (Convolutional Layer) **
// =================================================================================

extern "C" void conv_forward_gpu(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K) {
    cudnnHandle_t cudnn = get_cudnn_handle(); // 获取句柄
    float alpha = 1.0f, beta = 0.0f;

    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnFilterDescriptor_t k_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnSafeCall(cudnnCreateTensorDescriptor(&x_desc));
    cudnnSafeCall(cudnnCreateTensorDescriptor(&y_desc));
    cudnnSafeCall(cudnnCreateFilterDescriptor(&k_desc));
    cudnnSafeCall(cudnnCreateConvolutionDescriptor(&conv_desc));

    cudnnSafeCall(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W));
    cudnnSafeCall(cudnnSetFilter4dDescriptor(k_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, M, C, K, K));
    cudnnSafeCall(cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
    cudnnSafeCall(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));

    int out_B, out_C, out_H, out_W;
    cudnnSafeCall(cudnnGetConvolution2dForwardOutputDim(conv_desc, x_desc, k_desc, &out_B, &out_C, &out_H, &out_W));
    cudnnSafeCall(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_B, out_C, out_H, out_W));

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    size_t workspace_bytes = 0;
    cudnnSafeCall(cudnnGetConvolutionForwardWorkspaceSize(cudnn, x_desc, k_desc, conv_desc, y_desc, algo, &workspace_bytes));
    void *d_workspace{nullptr};
    if (workspace_bytes > 0) {
        cudaSafeCall(cudaMalloc(&d_workspace, workspace_bytes));
    }

    cudnnSafeCall(cudnnConvolutionForward(cudnn, &alpha, x_desc, x, k_desc, k, conv_desc, algo, d_workspace, workspace_bytes, &beta, y_desc, y));

    if (d_workspace) cudaFree(d_workspace);
    cudnnSafeCall(cudnnDestroyTensorDescriptor(x_desc));
    cudnnSafeCall(cudnnDestroyTensorDescriptor(y_desc));
    cudnnSafeCall(cudnnDestroyFilterDescriptor(k_desc));
    cudnnSafeCall(cudnnDestroyConvolutionDescriptor(conv_desc));
}

extern "C" void conv_backward_gpu(float *dx, float *dk, const float *dy, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K) {
    cudnnHandle_t cudnn = get_cudnn_handle();
    float alpha = 1.0f, beta = 0.0f;

    cudnnTensorDescriptor_t x_desc, dy_desc, dx_desc;
    cudnnFilterDescriptor_t k_desc, dk_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&x_desc);
    cudnnCreateTensorDescriptor(&dy_desc);
    cudnnCreateTensorDescriptor(&dx_desc);
    cudnnCreateFilterDescriptor(&k_desc);
    cudnnCreateFilterDescriptor(&dk_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W);
    cudnnSetFilter4dDescriptor(k_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, M, C, K, K);
    cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);
    cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH);

    int out_B, out_C, out_H, out_W;
    cudnnGetConvolution2dForwardOutputDim(conv_desc, x_desc, k_desc, &out_B, &out_C, &out_H, &out_W);
    cudnnSetTensor4dDescriptor(dy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_B, out_C, out_H, out_W);
    cudnnSetTensor4dDescriptor(dx_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W);
    cudnnSetFilter4dDescriptor(dk_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, M, C, K, K);

    cudnnConvolutionBwdFilterAlgo_t algo_filter = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

    size_t workspace_bytes_filter = 0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn, x_desc, dy_desc, conv_desc, dk_desc, algo_filter, &workspace_bytes_filter);
    void *d_workspace_filter{nullptr};
    if (workspace_bytes_filter > 0) {
        cudaMalloc(&d_workspace_filter, workspace_bytes_filter);
    }

    cudnnConvolutionBackwardFilter(cudnn, &alpha, x_desc, x, dy_desc, dy, conv_desc, algo_filter, d_workspace_filter, workspace_bytes_filter, &beta, dk_desc, dk);

    if (d_workspace_filter) cudaFree(d_workspace_filter);

    cudnnConvolutionBwdDataAlgo_t algo_data = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;

    size_t workspace_bytes_data = 0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, k_desc, dy_desc, conv_desc, dx_desc, algo_data, &workspace_bytes_data);
    void *d_workspace_data{nullptr};
    if (workspace_bytes_data > 0) {
        cudaMalloc(&d_workspace_data, workspace_bytes_data);
    }

    cudnnConvolutionBackwardData(cudnn, &alpha, k_desc, k, dy_desc, dy, conv_desc, algo_data, d_workspace_data, workspace_bytes_data, &beta, dx_desc, dx);

    if (d_workspace_data) cudaFree(d_workspace_data);

    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(dy_desc);
    cudnnDestroyTensorDescriptor(dx_desc);
    cudnnDestroyFilterDescriptor(k_desc);
    cudnnDestroyFilterDescriptor(dk_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
}


// =================================================================================
// ** 最大池化层 (Max Pooling Layer) **
// =================================================================================

extern "C" void max_pool_forward_gpu(float *y, const float *x, int *pool_idx, const int B, const int C, const int H, const int W, const int K) {
    cudnnHandle_t cudnn = get_cudnn_handle();
    float alpha = 1.0f, beta = 0.0f;
    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnPoolingDescriptor_t pool_desc;
    cudnnCreateTensorDescriptor(&x_desc);
    cudnnCreateTensorDescriptor(&y_desc);
    cudnnCreatePoolingDescriptor(&pool_desc);
    cudnnSetPooling2dDescriptor(pool_desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, K, K, 0, 0, K, K);
    cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W);
    int out_B, out_C, out_H, out_W;
    cudnnGetPooling2dForwardOutputDim(pool_desc, x_desc, &out_B, &out_C, &out_H, &out_W);
    cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_B, out_C, out_H, out_W);
    cudnnPoolingForward(cudnn, pool_desc, &alpha, x_desc, x, &beta, y_desc, y);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyPoolingDescriptor(pool_desc);
}

extern "C" void max_pool_backward_gpu(float *dx, const float *dy, const int *pool_idx, const int B, const int C, const int H, const int W, const int K) {}

// =================================================================================
// ** ReLU 激活函数 (ReLU Activation) **
// =================================================================================

__global__ void relu_forward_kernel(float *y, const float *x, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = x[idx] > 0.0f ? x[idx] : 0.0f;
    }
}

extern "C" void relu_forward_gpu(float *y, const float *x, const int N) {
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    relu_forward_kernel<<<blocks, THREADS_PER_BLOCK>>>(y, x, N);
    cudaSafeCall(cudaGetLastError());
}

__global__ void relu_backward_kernel(float *dx, const float *dy, const float *x, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dx[idx] = x[idx] > 0.0f ? dy[idx] : 0.0f;
    }
}

extern "C" void relu_backward_gpu(float *dx, const float *dy, const float *x, const int N) {
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    relu_backward_kernel<<<blocks, THREADS_PER_BLOCK>>>(dx, dy, x, N);
    cudaSafeCall(cudaGetLastError());
}

// =================================================================================
// ** 全连接层 (Fully Connected Layer) **
// =================================================================================

extern "C" void fully_connected_forward_gpu(float *y, const float *x, const float *W, const int B, const int In, const int Out) {
    cublasHandle_t cublas = get_cublas_handle();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSafeCall(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                               Out, B, In,
                               &alpha,
                               W, Out,
                               x, In,
                               &beta,
                               y, Out));
}

extern "C" void fully_connected_backward_gpu(float *dx, float *dW, const float *dy, const float *x, const float *W, const int B, const int In, const int Out) {
    cublasHandle_t cublas = get_cublas_handle();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSafeCall(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                               Out, In, B,
                               &alpha,
                               dy, Out,
                               x, In,
                               &beta,
                               dW, Out));
    cublasSafeCall(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                               In, B, Out,
                               &alpha,
                               W, In,
                               dy, Out,
                               &beta,
                               dx, In));
}

// =================================================================================
// ** Softmax + 损失函数 (Softmax + Loss) **
// =================================================================================

extern "C" void softmax_loss_gpu(float *loss, float *probs, const float *scores, const int *labels, const int B, const int C) {
    cudnnHandle_t cudnn = get_cudnn_handle();
    float alpha = 1.0f, beta = 0.0f;
    cudnnTensorDescriptor_t desc;
    cudnnCreateTensorDescriptor(&desc);
    cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, 1, 1);
    cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, desc, scores, &beta, desc, probs);
    cudnnDestroyTensorDescriptor(desc);
}

__global__ void softmax_backward_kernel(float *d_scores, const float *probs, const int *labels, int B, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B) { 
        int label = labels[idx];
        for (int j = 0; j < C; ++j) {
            int sample_idx = idx * C + j;
            float indicator = (j == label) ? 1.0f : 0.0f;
            d_scores[sample_idx] = (probs[sample_idx] - indicator) / B;
        }
    }
}

extern "C" void softmax_backward_gpu(float *d_scores, const float *probs, const int *labels, const int B, const int C) {
    int blocks = (B + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    softmax_backward_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_scores, probs, labels, B, C);
    cudaSafeCall(cudaGetLastError());
}
```

- 头文件提供接口，cython的pyx文件中这个接口头文件

```cpp
//layers_wrapper.h
#ifndef LAYERS_WRAPPER_H
#define LAYERS_WRAPPER_H

// C-style interface for CUDA functions
#ifdef __cplusplus
extern "C" {
#endif

// ** 卷积层 (Convolutional Layer)
void conv_forward_gpu(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K);
void conv_backward_gpu(float *dx, float *dk, const float *dy, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K);

// ** 最大池化层 (Max Pooling Layer)
void max_pool_forward_gpu(float *y, const float *x, int *pool_idx, const int B, const int C, const int H, const int W, const int K);
void max_pool_backward_gpu(float *dx, const float *dy, const int *pool_idx, const int B, const int C, const int H, const int W, const int K);

// ** ReLU 激活函数 (ReLU Activation)
void relu_forward_gpu(float *y, const float *x, const int N);
void relu_backward_gpu(float *dx, const float *dy, const float *x, const int N);

// ** 全连接层 (Fully Connected Layer)
void fully_connected_forward_gpu(float *y, const float *x, const float *W, const int B, const int In, const int Out);
void fully_connected_backward_gpu(float *dx, float *dW, const float *dy, const float *x, const float *W, const int B, const int In, const int Out);

// ** Softmax 损失函数 (Softmax Loss)
void softmax_loss_gpu(float *loss, float *probs, const float *scores, const int *labels, const int B, const int C);
void softmax_backward_gpu(float *d_scores, const float *probs, const int *labels, const int B, const int C);

#ifdef __cplusplus
}
#endif

#endif // LAYERS_WRAPPER_H
```

#### 前端代码

- cython封装C++函数接口

```python
#cnn_layers.pyx

import numpy as np
cimport numpy as np

# ** C++ 函数声明 **
cdef extern from "../cpp/layers_wrapper.h":
    void conv_forward_gpu(float *y, const float *x, const float *k, int B, int M, int C, int H, int W, int K) noexcept nogil
    void conv_backward_gpu(float *dx, float *dk, const float *dy, const float *x, const float *k, int B, int M, int C, int H, int W, int K) noexcept nogil
    void max_pool_forward_gpu(float *y, const float *x, int *pool_idx, int B, int C, int H, int W, int K) noexcept nogil
    void max_pool_backward_gpu(float *dx, const float *dy, const int *pool_idx, int B, int C, int H, int W, int K) noexcept nogil
    void relu_forward_gpu(float *y, const float *x, int N) noexcept nogil
    void relu_backward_gpu(float *dx, const float *dy, const float *x, int N) noexcept nogil
    void fully_connected_forward_gpu(float *y, const float *x, const float *W, int B, int In, int Out) noexcept nogil
    void fully_connected_backward_gpu(float *dx, float *dW, const float *dy, const float *x, const float *W, int B, int In, int Out) noexcept nogil
    void softmax_loss_gpu(float *loss, float *probs, const float *scores, const int *labels, int B, int C) noexcept nogil
    void softmax_backward_gpu(float *d_scores, const float *probs, const int *labels, int B, int C) noexcept nogil


# ** Python 包装函数 **
# -------------------
# 卷积层
# -------------------
def conv_forward(np.ndarray[np.float32_t, ndim=4] x, np.ndarray[np.float32_t, ndim=4] k):
    cdef int B = x.shape[0]
    cdef int C = x.shape[1]
    cdef int H = x.shape[2]
    cdef int W = x.shape[3]
    cdef int M = k.shape[0]
    cdef int K = k.shape[2]
    cdef int out_H = H - K + 1 + 2 * 1
    cdef int out_W = W - K + 1 + 2 * 1
    cdef np.ndarray[np.float32_t, ndim=4] y = np.zeros((B, M, out_H, out_W), dtype=np.float32)
    with nogil:
        conv_forward_gpu(<float*> &y[0,0,0,0], <const float*> &x[0,0,0,0], <const float*> &k[0,0,0,0], B, M, C, H, W, K)
    return y, (x, k)

def conv_backward(np.ndarray[np.float32_t, ndim=4] dy, cache):
    cdef np.ndarray[np.float32_t, ndim=4] x, k
    x, k = cache
    cdef int B = x.shape[0]
    cdef int C = x.shape[1]
    cdef int H = x.shape[2]
    cdef int W = x.shape[3]
    cdef int M = k.shape[0]
    cdef int K = k.shape[2]
    cdef np.ndarray[np.float32_t, ndim=4] dx = np.zeros_like(x)
    cdef np.ndarray[np.float32_t, ndim=4] dk = np.zeros_like(k)
    with nogil:
        conv_backward_gpu(<float*> &dx[0,0,0,0], <float*> &dk[0,0,0,0], <const float*> &dy[0,0,0,0], <const float*> &x[0,0,0,0], <const float*> &k[0,0,0,0], B, M, C, H, W, K)
    return dx, dk

# -------------------
# ReLU 激活
# -------------------
def relu_forward(np.ndarray[np.float32_t] x):
    cdef np.ndarray[np.float32_t] y = np.zeros_like(x)
    cdef int N = x.size
    with nogil:
        relu_forward_gpu(<float*> y.data, <const float*> x.data, N)
    return y, x

def relu_backward(np.ndarray[np.float32_t] dy, cache):
    cdef np.ndarray[np.float32_t] x = cache
    cdef np.ndarray[np.float32_t] dx = np.zeros_like(x)
    cdef int N = x.size
    with nogil:
        relu_backward_gpu(<float*> dx.data, <const float*> dy.data, <const float*> x.data, N)
    return dx

# -------------------
# 最大池化层
# -------------------
def max_pool_forward(np.ndarray[np.float32_t, ndim=4] x, int pool_size):
    cdef int B = x.shape[0]
    cdef int C = x.shape[1]
    cdef int H = x.shape[2]
    cdef int W = x.shape[3]
    cdef int K = pool_size
    cdef int out_H = H // K
    cdef int out_W = W // K
    cdef np.ndarray[np.float32_t, ndim=4] y = np.zeros((B, C, out_H, out_W), dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=4] pool_idx = np.zeros_like(y, dtype=np.int32)
    with nogil:
        max_pool_forward_gpu(<float*> &y[0,0,0,0], <const float*> &x[0,0,0,0], <int*> &pool_idx[0,0,0,0], B, C, H, W, K)
    return y, (x, pool_size, pool_idx)

def max_pool_backward(np.ndarray[np.float32_t, ndim=4] dy, cache):
    cdef np.ndarray[np.float32_t, ndim=4] x
    cdef int pool_size
    cdef np.ndarray[np.int32_t, ndim=4] pool_idx
    x, pool_size, pool_idx = cache
    cdef int B = x.shape[0]
    cdef int C = x.shape[1]
    cdef int H = x.shape[2]
    cdef int W = x.shape[3]
    cdef int K = pool_size
    cdef np.ndarray[np.float32_t, ndim=4] dx = np.zeros_like(x)
    with nogil:
        max_pool_backward_gpu(<float*> &dx[0,0,0,0], <const float*> &dy[0,0,0,0], <const int*> &pool_idx[0,0,0,0], B, C, H, W, K)
    return dx

# -------------------
# 全连接层
# -------------------
def fully_connected_forward(np.ndarray[np.float32_t, ndim=2] x, np.ndarray[np.float32_t, ndim=2] W):
    cdef int B = x.shape[0]
    cdef int In = x.shape[1]
    cdef int Out = W.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] y = np.zeros((B, Out), dtype=np.float32)
    with nogil:
        fully_connected_forward_gpu(<float*> y.data, <const float*> x.data, <const float*> W.data, B, In, Out)
    return y, (x, W)

def fully_connected_backward(np.ndarray[np.float32_t, ndim=2] dy, cache):
    cdef np.ndarray[np.float32_t, ndim=2] x, W
    x, W = cache
    cdef int B = x.shape[0]
    cdef int In = x.shape[1]
    cdef int Out = W.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] dx = np.zeros_like(x)
    cdef np.ndarray[np.float32_t, ndim=2] dW = np.zeros_like(W)
    with nogil:
        fully_connected_backward_gpu(<float*> dx.data, <float*> dW.data, <const float*> dy.data, <const float*> x.data, <const float*> W.data, B, In, Out)
    return dx, dW

# -------------------
# Softmax 损失
# -------------------
def softmax_loss(np.ndarray[np.float32_t, ndim=2] scores, np.ndarray[np.int32_t, ndim=1] labels):
    cdef int B = scores.shape[0]
    cdef int C = scores.shape[1]
    cdef np.ndarray[np.float32_t, ndim=2] probs = np.zeros_like(scores)
    cdef np.ndarray[np.float32_t, ndim=1] loss_array = np.zeros(1, dtype=np.float32)
    with nogil:
        softmax_loss_gpu(<float*> loss_array.data, <float*> probs.data, <const float*> scores.data, <const int*> labels.data, B, C)
    loss = -np.sum(np.log(probs[np.arange(B), labels] + 1e-9)) / B
    d_scores = probs.copy()
    d_scores[np.arange(B), labels] -= 1
    d_scores /= B
    return loss, d_scores
```

- 编译CUDA和CYTHON，编译完成后可以生成CUDA的共享库和`cnn_layers.cpython-3xx-x86_64-linux-gnu.so`这样的共享库，后者可以直接python ` import cnn_layers`导入，然后调用cython封装后的函数

```python
# setup.py
import os
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

print("--- 开始CUDA代码编译 (Linux) ---")
# --- 1. 路径 ---
setup_dir = os.path.dirname(os.path.abspath(__file__))
cpp_dir = os.path.abspath(os.path.join(setup_dir, "..", "cpp"))
build_dir = os.path.join(cpp_dir, "build")
cuda_source = os.path.join(cpp_dir, "layers.cu")

lib_name = "gpucnn"
lib_filename = f"lib{lib_name}.so"
lib_path = os.path.join(build_dir, lib_filename)

# --- 2. 创建编译输出目录 ---
if not os.path.exists(build_dir):
    print(f"创建编译输出目录: {build_dir}")
    os.makedirs(build_dir)

# --- 3. 构建NVCC编译命令 ---
compile_command = (
    f'nvcc --shared -Xcompiler -fPIC -std=c++11 '
    f'-o "{lib_path}" '
    f'"{cuda_source}" '
    f'-gencode=arch=compute_70,code=sm_70 '
    f'-gencode=arch=compute_75,code=sm_75 '
    f'-gencode=arch=compute_80,code=sm_80 '
    f'-gencode=arch=compute_86,code=sm_86 '
    f'-gencode=arch=compute_89,code=sm_89 '
    f'-lcudnn -lcublas'
)

print(f"即将执行CUDA编译命令:\n{compile_command}")

# --- 4. 执行编译 ---
compile_status = os.system(compile_command)
if compile_status != 0:
    print("\n错误: CUDA编译失败！")
    print("请检查:")
    print("1. NVIDIA CUDA Toolkit (nvcc) 是否已安装并配置在系统的PATH环境变量中。")
    print("2. cuDNN和cuBLAS库是否位于CUDA的安装目录中。")
    raise RuntimeError("CUDA compilation failed.")

print("--- CUDA代码编译成功 ---")

# ==============================================================================
# ** 定义Cython扩展 (Linux版本) **
# ==============================================================================

print("\n--- 开始Cython扩展编译 ---")

extensions = [
    Extension(
        "cnn_layers",
        ["cnn_layers.pyx"],
        language="c++",
        include_dirs=[
            numpy.get_include(),
            cpp_dir
        ],
        library_dirs=[build_dir],
        # 链接器会在库目录中查找 libgpucnn.so
        libraries=[lib_name],
        extra_compile_args=["-std=c++11", "-O2"],
        # -Wl,-rpath: 将库的运行时搜索路径嵌入到扩展中。
        extra_link_args=[f'-Wl,-rpath,{build_dir}']
    )
]

setup(
    name="CUDNN CNN Layers",
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level' : "3"}
    ),
    zip_safe=False,
)

print("---" " Cython扩展编译完成 ---")
```

- 工具模块，加载数据集

```python
#utils.py

import numpy as np
import os
import gzip

# ==============================================================================
# ** MNIST 数据集加载器 **
# ==============================================================================

def load_mnist(path, kind='train'):
    """ 
    从指定路径加载MNIST数据集。

    参数:
    - path: 存放MNIST文件的目录路径。
    - kind: 'train' 表示加载训练集, 't10k' 表示加载测试集。
    """
    # 构造标签和图像文件的完整路径
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')

    # 使用gzip模块读取标签文件
    with gzip.open(labels_path, 'rb') as lbpath:
        # 从文件偏移量8的位置开始读取，并转换为numpy数组
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    # 使用gzip模块读取图像文件
    with gzip.open(images_path, 'rb') as imgpath:
        # 从文件偏移量16的位置开始读取，并转换为numpy数组
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16)
        # 将一维数组重塑为 (样本数, 通道数, 高, 宽) 的4D张量
        # MNIST是灰度图，所以通道数为1。图像尺寸为28x28。
        images = images.reshape(len(labels), 1, 28, 28)

    return images, labels


# ==============================================================================
# ** 迭代器和数据处理 **
# ==============================================================================

def create_batches(data, labels, batch_size):
    """
    将数据集分割成多个批次(batch)。

    参数:
    - data: 图像数据 (NumPy数组)。
    - labels: 标签数据 (NumPy数组)。
    - batch_size: 每个批次的大小。

    返回:
    - 一个包含多个 (data_batch, label_batch) 元组的列表。
    """
    num_samples = data.shape[0]
    # 创建一个索引数组并打乱顺序，以实现随机抽样
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    batches = []
    # 按照batch_size步长进行迭代
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        # 获取当前批次的打乱后的索引
        batch_indices = indices[start_idx:end_idx]

        # 根据索引提取数据和标签
        data_batch = data[batch_indices]
        label_batch = labels[batch_indices]

        # 将数据类型转换为float32并进行归一化 (将像素值从[0, 255]缩放到[0, 1])
        data_batch = data_batch.astype(np.float32) / 255.0
        # 将标签类型转换为int32
        label_batch = label_batch.astype(np.int32)

        batches.append((data_batch, label_batch))

    return batches
```

- 训练和测试代码

```python
#train_mnist.py 

import numpy as np
from cnn_layers import (
    conv_forward, conv_backward,
    relu_forward, relu_backward,
    max_pool_forward, max_pool_backward,
    fully_connected_forward, fully_connected_backward,
    softmax_loss
)
from utils import load_mnist, create_batches
import time

class SimpleCNN:
    """
    一个简单的卷积神经网络模型: CONV -> RELU -> POOL -> FC -> SOFTMAX
    """
    def __init__(self, num_classes=10, num_filters=16, filter_size=3):
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.filter_size = filter_size

        fc_input_dim = num_filters * 14 * 14
        self.W1 = np.random.randn(num_filters, 1, filter_size, filter_size).astype(np.float32) * np.sqrt(2.0 / (num_filters * filter_size * filter_size))
        self.W2 = np.random.randn(num_classes, fc_input_dim).astype(np.float32) * np.sqrt(2.0 / fc_input_dim)

    def forward(self, X):
        conv_out, self.conv_cache = conv_forward(X, self.W1)
        relu_out, self.relu_cache = relu_forward(conv_out)
        pool_out, self.pool_cache = max_pool_forward(relu_out, pool_size=2)

        B, C, H, W = pool_out.shape
        fc_in = pool_out.reshape(B, -1)
        self.reshape_cache = pool_out.shape

        fc_out, self.fc_cache = fully_connected_forward(fc_in, self.W2)
        return fc_out

    def backward(self, d_scores, lr=0.001):
        dfc_in, dW2 = fully_connected_backward(d_scores, self.fc_cache)
        dpool_in = dfc_in.reshape(self.reshape_cache)
        drelu_in = max_pool_backward(dpool_in, self.pool_cache)
        dconv_in = relu_backward(drelu_in, self.relu_cache)
        dX, dW1 = conv_backward(dconv_in, self.conv_cache)

        self.W1 -= lr * dW1
        self.W2 -= lr * dW2

if __name__ == "__main__":
    EPOCHS = 5
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01

    # --- 加载数据 ---
    print("正在加载MNIST数据集...")
    X_train, y_train = load_mnist('./mnist_data', kind='train')
    X_test, y_test = load_mnist('./mnist_data', kind='t10k')
    print("数据集加载完毕。")

    # --- 初始化模型 ---
    model = SimpleCNN(num_classes=10, num_filters=16, filter_size=3)

    # --- 训练循环 ---
    print("开始训练...")
    try:
        for epoch in range(EPOCHS):
            start_time = time.time()
            batches = create_batches(X_train, y_train, BATCH_SIZE)
            total_loss = 0

            for i, (X_batch, y_batch) in enumerate(batches):
                scores = model.forward(X_batch)
                loss, d_scores = softmax_loss(scores, y_batch)
                total_loss += loss
                model.backward(d_scores, lr=LEARNING_RATE)

                if (i + 1) % 100 == 0:
                    print(f"Epoch {epoch+1}/{EPOCHS}, Batch {i+1}/{len(batches)}, Loss: {loss:.4f}")

            epoch_time = time.time() - start_time
            avg_loss = total_loss / len(batches)
            print(f"Epoch {epoch+1} 完成。平均损失: {avg_loss:.4f}, 耗时: {epoch_time:.2f}s")

        # --- 测试模型 ---
        print("\n开始测试...")
        test_batches = create_batches(X_test, y_test, BATCH_SIZE)
        correct_predictions = 0
        total_predictions = 0

        for X_batch, y_batch in test_batches:
            scores = model.forward(X_batch)
            predictions = np.argmax(scores, axis=1)
            correct_predictions += np.sum(predictions == y_batch)
            total_predictions += X_batch.shape[0]

        accuracy = correct_predictions / total_predictions
        print(f"测试集准确率: {accuracy:.4f} ({correct_predictions}/{total_predictions})")

    finally:
        print("程序结束。GPU资源将由C++库自动释放。")
```

**编译运行**

```
cd python
python setup.py build_ext --inplace
python train_mnist.py
```

### 总结
只是一个简单的深度学习库，距离完善的深度学习框架还有很多需要做的，比如至少还需要实现计算图和自动求导。这里只是一个简单的例子，主要展示python高层和gpu底层矩阵运算是如何连接起来的，可以对深度学习框架的底层构造有一个更深的理解。



