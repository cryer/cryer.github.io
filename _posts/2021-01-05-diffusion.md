---
layout: post
title: diffusion扩散模型
description: diffusion扩散模型

---
<script>
  MathJax = {
    tex: {
      inlineMath: [['\$', '\$']],
      displayMath: [['\$', '\$']]
    }
  };
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>


### 导入

去年提出的扩散模型（也就是`DDPM(Denoising Diffusion Probabilistic Models)`）确实是生成模型中的一大突破，个人觉得比`GAN`更有前途。我不想仔细讲解扩散模型的原理和公式推导，而是希望直接通过代码给出更加直观清晰的视角，然后再配合简单的说明。

### 正文

扩散模型最核心的步骤就是2步，正向加噪和反向扩散。所谓正向加噪，就是拿一个清晰的原图，然后不断加入均值0方差1的高斯分布噪声（以特定的系数），重复T次之后，图像就几乎完全变成了噪声图像，分辨不出任何的原图信息。而反向扩散就反过来，从T时刻开始，用噪声图像来预测T-1时刻的图像，然后不断往前预测，不断减少噪声，最后恢复原图。

这是扩散模型的基本原理和理解，但是实际使用时，和理解会有一定偏差。比如加噪过程不会一步一步添加，实际上是可以直接计算出任意t时刻的噪声图像的；比如反向扩散，也不需要一步步往前训练，而是使用随机时间步训练，我们不关心采样顺序，1000步只是为了**覆盖完整噪声范围**。只有采样的时候才一步步往前采样，因为采样是一个马尔可夫过程，必须完整走完1000步；比如实际预测的并不是前一步的图像，而是前一步加的噪声等，后面代码就可以看得很清楚。

下面就直接进入代码环节，不过思来想去，还是在最后会进行一些原理性的说明。


噪声2个系数`α`和`β`的变化，每一步都是固定的，计算方式如代码所示。下面的所有代码，我都会写上详细的注释，本次就以基本的MNIST作为数据集，因此有的尺寸会直接以MNIST的图像尺寸标注，也就是`1*28*28`。

```python
class DiffusionModel:
    """
    扩散过程实现
    """
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02,device='cuda'):
        """
        初始化扩散模型参数
        参数:
            T: 扩散总步数
            beta_start: 初始噪声系数
            beta_end: 最终噪声系数
        """
        self.T = T  # 扩散总步数
        
        # 线性调度噪声系数β，从beta_start到beta_end
        self.betas = torch.linspace(beta_start, beta_end, T,device=device)
        
        # α = 1 - β，表示保留原始数据的比例
        self.alphas = 1. - self.betas
        
        # α的累积乘积，用于计算任意时刻的噪声图像
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def forward_diffuse(self, x0, t, device):
        """
        正向扩散过程：对输入图像添加噪声
        参数:
            x0: 原始图像(batch_size, 1, 28, 28)
            t: 时间步(0到T-1的整数)
        返回:
            xt: 加噪后的图像
            noise: 实际添加的噪声
        """
        # 生成与输入同形状的标准高斯噪声
        t = t.to(device) 
        noise = torch.randn_like(x0, device=device)
        
        # 获取当前时间步的α_bar值，并调整形状以匹配输入维度
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1).to(device)
        
        # 计算加噪图像：√ᾱx0 + √(1-ᾱ)ε
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        
        return xt, noise
```

扩散模型的架构是基于`U-Net`的，先下采样，然后上采样。完整的应该包括跳跃连接，也就是上采样和下采样之间的残差连接，以及时间步的编码嵌入，这个后面会给出。现在先以一个简单的演示架构来说明，不加入跳跃连接和时间步位置编码，重点先关注扩散模型的正向加噪和反向扩散的过程。

```python
class UNet(nn.Module):
    """
    U-Net结构的噪声预测网络
    输入: 噪声图像
    输出: 预测的噪声
    """
    def __init__(self):
        super().__init__()
        
        # 下采样部分第一层
        self.down1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),  # 输入通道1，输出64，3x3卷积
            nn.ReLU(),                       # 激活函数
            nn.Conv2d(64, 64, 3, padding=1), # 保持通道数不变
            nn.ReLU()
        )
        
        # 下采样部分第二层(带降采样)
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),                 # 2x2最大池化，尺寸减半
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        
        # 上采样部分
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),  # 转置卷积实现上采样
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        
        # 输出层(1x1卷积将通道数降为1)
        self.out = nn.Conv2d(64, 1, 1)
        
    def forward(self, x):
        """
        前向传播
        参数:
            x: 输入噪声图像(batch_size, 1, 28, 28)
        返回:
            预测的噪声(batch_size, 1, 28, 28)
        """
        # 下采样路径
        x1 = self.down1(x)  # -> (batch_size, 64, 28, 28)
        x2 = self.down2(x1) # -> (batch_size, 128, 14, 14)
        
        # 上采样路径
        x = self.up1(x2)    # -> (batch_size, 64, 28, 28)
        
        # 输出预测噪声
        return self.out(x)
```

然后就是训练代码：

```python
def train(model, diffusion, dataloader, epochs=10, device='cuda'):
    """
    训练噪声预测模型
    参数:
        model: UNet模型实例
        diffusion: DiffusionModel实例
        dataloader: 数据加载器
        epochs: 训练轮数
        device: 训练设备(cpu/cuda)
    """
    model = model.to(device)
    
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 使用均方误差损失
    criterion = nn.MSELoss()
    
    # 训练循环
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x0, _) in enumerate(dataloader):
            x0 = x0.to(device)
            
            # 随机采样时间步(0到T-1)
            t = torch.randint(0, diffusion.T, (x0.size(0),), device=device)
            
            # 正向扩散：加噪
            xt, noise = diffusion.forward_diffuse(x0, t, device)
            
            # 预测噪声
            pred_noise = model(xt, t)
            
            # 计算损失(预测噪声与真实噪声的MSE)
            loss = criterion(pred_noise, noise)
            total_loss += loss.item()
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 每100个batch打印一次损失
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}')
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1} completed | Avg Loss: {avg_loss:.4f}')
```

代码非常简单，随机取一个时间步，先正向加噪计算t步的噪声图像，并且返回噪声作为标签，然后根据网络输出和噪声标签计算损失，反向传播更新参数即可。



然后是采样代码，也就是预测，利用随机噪声，然后从T时刻开始一步步往前反向扩散，慢慢移除噪声，最终从XT一直采样到X0，获取原图。关于加噪的原因和反向扩散公式，文章末尾我会简单说明。

```python
def sample(model, diffusion, n_samples=16, device='cuda'):
    """
    从纯噪声生成图像
    参数:
        model: 训练好的UNet模型
        diffusion: DiffusionModel实例
        n_samples: 生成样本数量
        device: 计算设备
    返回:
        生成的图像样本(cpu tensor)
    """
    model.eval()  # 设置为评估模式
    
    with torch.no_grad():  # 禁用梯度计算
        # 初始化为随机噪声
        x = torch.randn(n_samples, 1, 28, 28, device=device)
        
        # 反向扩散过程(从T到0)
        for t in reversed(range(diffusion.T)):
            # 创建当前时间步的张量
            t_tensor = torch.full((n_samples,), t, device=device)
            
            # 预测噪声
            pred_noise = model(x, t_tensor)
            
            # 获取当前时间步的参数
            alpha_t = diffusion.alphas[t]
            alpha_bar_t = diffusion.alpha_bars[t]
            beta_t = diffusion.betas[t]
            
            # 计算去噪后的图像
            if t > 0:
                noise = torch.randn_like(x)  # 添加随机噪声
            else:
                noise = torch.zeros_like(x)  # 最后一步不加噪声
                
            # 反向扩散公式计算（实则就是目标分布q(xt-1 | xt,x0)的均值部分）
            x = (1 / torch.sqrt(alpha_t)) * (
                x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise
            ) + torch.sqrt(beta_t) * noise
    
    return x.cpu()  # 返回CPU上的结果
```

然后我们就准备MNIST数据，初始化扩散模型，初始化U-Net模型，训练模型，然后采样即可。

下图是训练5个epoch后，采样的16张图片：

![](https://github.com/cryer/cryer.github.io/raw/master/image/mnist1.png)


#### 模型改进

演示模型效果不好是正常的，缺少了时间步编码和残差连接，接下来给出相对更加合理的U-Net模型：

```python
# 模块1: 正弦位置嵌入 (Sinusoidal Position Embeddings)
class SinusoidalPositionEmbeddings(nn.Module):
    """
    正弦位置嵌入模块。
    这个模块用于将时间步（timestep）t 编码成一个高维向量。
    在Diffusion模型中，模型需要知道当前处于哪个去噪步骤
    """
    def __init__(self, dim):
        """
        初始化方法。
        Args:
            dim (int): 编码向量的维度。这个维度需要是偶数。
        """
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        """
        前向传播方法。
        Args:
            t (torch.Tensor): 时间步张量，形状为 (B,)，其中 B 是批量大小。
        
        Returns:
            torch.Tensor: 时间编码向量，形状为 (B, dim)。
        """
        device = t.device
        half_dim = self.dim // 2
        
        # 计算嵌入的基底频率，取值范围从 1 到 10000^(-1)
        # embeddings 的形状是 (half_dim,)
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # 将时间步 t 和频率相乘
        # t[:, None] 的形状是 (B, 1)
        # embeddings[None, :] 的形状是 (1, half_dim)
        # 广播机制作用后，embeddings 的形状变为 (B, half_dim)
        embeddings = t[:, None] * embeddings[None, :]
        
        # 将sin和cos编码拼接在一起，形成最终的时间编码
        # 返回的张量形状为 (B, dim)
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


# 模块2: 残差模块 (Residual Block)
class ResidualBlock(nn.Module):
    """
    带有时间嵌入的残差模块。
    构成U-Net主体的基本单元。它包含两个卷积层、归一化层，并加入了时间嵌入信息和残差连接。
    """
    def __init__(self, in_channels, out_channels, time_emb_dim=128):
        """
        初始化方法。
        Args:
            in_channels (int): 输入特征图的通道数。
            out_channels (int): 输出特征图的通道数。
            time_emb_dim (int): 时间编码向量的维度。
        """
        super().__init__()
        # 时间嵌入处理网络：一个简单的MLP，将时间编码向量映射到与卷积特征图兼容的维度
        self.time_mlp = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels) # 分组归一化，8个组
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        # 残差连接的捷径（shortcut）
        # 如果输入和输出通道数不同，则使用1x1卷积进行匹配；否则直接恒等映射
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, t_emb):
        """
        前向传播方法。
        Args:
            x (torch.Tensor): 输入特征图，形状为 (B, in_channels, H, W)。
            t_emb (torch.Tensor): 时间编码向量，形状为 (B, time_emb_dim)。
        
        Returns:
            torch.Tensor: 输出特征图，形状为 (B, out_channels, H, W)。
        """
        # h 是主路径
        h = self.conv1(x)
        h = self.norm1(h)
        
        # 处理时间嵌入并加到特征图中
        t_emb_proj = self.time_mlp(t_emb) # (B, time_emb_dim) -> (B, out_channels)
        # 需要将 t_emb_proj 从 (B, out_channels) 扩展到 (B, out_channels, 1, 1) 以便和 (B, out_channels, H, W) 的 h 相加
        h = h + t_emb_proj.unsqueeze(-1).unsqueeze(-1)
        
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        
        # 最终输出 = 主路径输出 + 捷径输出，然后通过激活函数
        return F.silu(h + self.shortcut(x))

# 模块3: U-Net 主网络架构
class ResidualUNet(nn.Module):
    """
    基于残差块的U-Net网络。
    这是整个Diffusion模型的核心，用于预测在给定时间步t时添加到图像x中的噪声。
    网络结构包含下采样路径、瓶颈层和上采样路径，并带有跳跃连接。
    
    假设输入图像尺寸为 (B, 1, 28, 28)，下面是各层尺寸变化的注释。
    """
    def __init__(self, in_channels=1, out_channels=1, hidden_dims=[64, 128, 256]):
        """
        初始化方法。
        Args:
            in_channels (int): 输入图像的通道数
            out_channels (int): 输出噪声图的通道数 
            hidden_dims (list[int]): U-Net各层级的隐藏通道数。
        """
        super().__init__()
        time_emb_dim = 128 # 定义时间编码维度
        
        # 1. 时间编码模块 ，输出尺寸【B，128】
        self.time_embed = SinusoidalPositionEmbeddings(time_emb_dim)
        
        # 2. 初始卷积层
        # 将输入图像从in_channels映射到第一个隐藏维度
        # 【B，1，28，28】 -> 【B，64，28，28】
        self.init_conv = nn.Conv2d(in_channels, hidden_dims[0], 3, padding=1)
        
        # 3. 下采样路径 (Encoder)
        self.down_blocks = nn.ModuleList([
            ResidualBlock(hidden_dims[0], hidden_dims[0], time_emb_dim),
            ResidualBlock(hidden_dims[1], hidden_dims[1], time_emb_dim)
        ])
        self.down_pools = nn.ModuleList([
            nn.Conv2d(hidden_dims[0], hidden_dims[1], 3, stride=2, padding=1),
            nn.Conv2d(hidden_dims[1], hidden_dims[2], 3, stride=2, padding=1)
        ])
        
        # 4. 瓶颈层 (Bottleneck)
        self.bottleneck = ResidualBlock(hidden_dims[2], hidden_dims[2], time_emb_dim)
        
        # 5. 上采样路径 (Decoder)
        self.up_convs = nn.ModuleList([
            nn.ConvTranspose2d(hidden_dims[2], hidden_dims[1], 2, stride=2),
            nn.ConvTranspose2d(hidden_dims[1], hidden_dims[0], 2, stride=2)
        ])
        self.up_blocks = nn.ModuleList([
            ResidualBlock(hidden_dims[1]*2, hidden_dims[1], time_emb_dim), # *2 是因为拼接了跳跃连接
            ResidualBlock(hidden_dims[0]*2, hidden_dims[0], time_emb_dim)
        ])
        
        # 6. 输出层
        self.final_block = ResidualBlock(hidden_dims[0], hidden_dims[0], time_emb_dim)
        self.out_conv = nn.Conv2d(hidden_dims[0], out_channels, 1) # 1x1卷积，将通道数映射回out_channels

    def forward(self, x, t):
        """
        前向传播方法。
        Args:
            x (torch.Tensor): 输入的带噪图像，形状为 (B, in_channels, H, W)。
                               假设为 (B, 1, 28, 28)。
            t (torch.Tensor): 当前的时间步，形状为 (B,)。
        
        Returns:
            torch.Tensor: 预测的噪声，形状与x相同 (B, out_channels, H, W)。
        """
        # --- 时间编码 ---
        # t: (B,) -> t_emb: (B, 128)
        t_emb = self.time_embed(t)
        
        # --- 初始卷积 ---
        # x: (B, 1, 28, 28) -> (B, 64, 28, 28)
        x = self.init_conv(x)
        
        # `skips` 用于存储下采样路径的输出，以便在上采样时进行跳跃连接
        skips = [] 
        
        # --- 下采样路径 (Encoder) ---
        # Level 1
        # x_in: (B, 64, 28, 28)
        x = self.down_blocks[0](x, t_emb) # -> (B, 64, 28, 28)
        skips.append(x) # 保存跳跃连接
        x = self.down_pools[0](x) # -> (B, 128, 14, 14)
        
        # Level 2
        # x_in: (B, 128, 14, 14)
        x = self.down_blocks[1](x, t_emb) # -> (B, 128, 14, 14)
        skips.append(x) # 保存跳跃连接
        x = self.down_pools[1](x) # -> (B, 256, 7, 7)
        
        # --- 瓶颈层 ---
        # x_in: (B, 256, 7, 7)
        x = self.bottleneck(x, t_emb) # -> (B, 256, 7, 7)
        
        # --- 上采样路径 (Decoder) ---
        # Level 2 -> 1
        # x_in: (B, 256, 7, 7)
        x = self.up_convs[0](x) # -> (B, 128, 14, 14)
        skip_connection = skips.pop() # 取出 Level 2 的跳跃连接 (B, 128, 14, 14)
        x = torch.cat([x, skip_connection], dim=1) # 拼接 -> (B, 256, 14, 14)
        x = self.up_blocks[0](x, t_emb) # -> (B, 128, 14, 14)
        
        # Level 1 -> 0
        # x_in: (B, 128, 14, 14)
        x = self.up_convs[1](x) # -> (B, 64, 28, 28)
        skip_connection = skips.pop() # 取出 Level 1 的跳跃连接 (B, 64, 28, 28)
        x = torch.cat([x, skip_connection], dim=1) # 拼接 -> (B, 128, 28, 28)
        x = self.up_blocks[1](x, t_emb) # -> (B, 64, 28, 28)
        
        # --- 输出 ---
        # x_in: (B, 64, 28, 28)
        x = self.final_block(x, t_emb) # -> (B, 64, 28, 28)
        # 1x1卷积调整通道数，输出预测的噪声
        # x: (B, 64, 28, 28) -> (B, 1, 28, 28)
        return self.out_conv(x)
```

除了改进的模型，还有正弦时间步位置嵌入编码，另外为了读者看的更清晰，我特别注释了每一层输出的维度尺寸。

使用这个新模型后，训练5个epoch后，使用16个随机噪声采样后的图片效果如下：

![](https://github.com/cryer/cryer.github.io/raw/master/image/mnist2.png)

换一个更复杂的数据集`CIFAR-100`,训练20个epoch后，效果如下：

![](https://github.com/cryer/cryer.github.io/raw/master/image/cifar100.png)

效果也还可以，训练时间毕竟不算长，模型也并不是很深。

对于只需要对扩散模型的基本原理有简单的理解，并且能够代码中使用的读者，看到这里已经够了，再仔细看看代码流程就好。下面会讲解一点扩散模型偏底层一点的东西以及公式的简单说明。



### 更多

- 系数α和β，实则就是**信噪比**，α表示信号所占的权重，而β表示噪声所占的权重。每一步的取值都是固定的，其中信号权重越来越低，噪声权重越来越大。是因为前期加入噪声对图像信号的影响非常大，尤其是第一次，越往后加入噪声影响越小，因为看起来都是噪声，都分辨不出图像本身。因此为了让信噪比均衡，也就是1000步中信号和噪声的每一步影响都差不多，就需要前期加大信号的权重，减少噪声影响，后期加大噪音权重，加大噪声影响。

- 每一步加入0-1高斯噪声后的噪声图像都是服从高斯分布的。为什么呢？如果一个分布z服从均值`μ`，标准差`σ`的高斯分布，那么z减去均值除以标准差后的分布服从0-1高斯分布。把均值和标准差移相到右边，`z=μ + σ*ε`,其中`ε`服从0-1高斯分布。而我们加噪的过程就是每一步增加0-1高斯噪声，具体公式是：\$X_t = \sqrt{a_t } X_{t-1}+ \sqrt{1 - a_t} \epsilon\$ ,和上述的高斯分布形式是一致的，此时系数βt（即1-αt）就相当于方差（**记住这个，后面会说到**），前面就相当于均值。这个也被称为**重参数化**、

- 加噪公式可以进一步展开，不断递归，最终可以得到Xt可以只用X0就能计算，这也是为什么前面说正向加噪过程，不需要一步步计算的原因

- 我们加噪的过程是`q(xt | xt-1)`，可以通过加入0-1高斯噪声，甚至直接用X0计算。而反向扩散的目标是`q(xt-1 | xt)`，可惜的是，我们并不能直接计算，因此才需要使用神经网络来预测，也就是`pθ(xt-1 | xt)`来预测`q(xt-1 | xt)`。而这没有解决目标`q(xt-1 | xt)`未知的问题，因为神经网络需要目标来计算损失，而目标依然是未知的。你可能会想，正向扩散过程中，我们知道了每一步的Xt，知道每一步加的噪声ε，知道加噪声的信噪比参数α和β，那我神经网络直接设置输入为Xt时刻图像，输出为Xt-1时刻图像，真正的Xt-1时刻的标签和Xt输入这两个都可以通过X0配合参数α计算出，不是就可以计算损失，训练参数吗？这理论上当然是可行的，但是效果很差，原因如下：
  
  1. 高噪声水平下的预测不确定性问题：在逆向扩散过程中，当时间步 t 较大时（即早期阶段），Xt​ 几乎完全被高斯噪声主导，图像信息高度退化。直接预测 Xt−1需要模型从随机噪声中重建复杂结构，这会导致预测结果方差极大、训练不稳定，因为模型难以区分噪声与真实信号
  
  2. 时间步长嵌入的精细化控制需求：引入时间步长 t 允许模型根据噪声强度动态调整行为。在扩散早期（高噪声），模型应关注图像轮廓等低频特征；在后期（低噪声），则聚焦细节修复。直接预测 Xt−1 缺乏这种机制
  
     因此我们需要寻求新的神经网络建模，而正向扩散的过程实则是一个马尔科夫链，也就是t时刻的图像只受t-1时刻图像的影响，因此我们条件加上X0原始图像，并不影响分布，即`q(xt-1 | xt,x0)`不影响分布，而这个分布可以进一步计算，Xt，X0同时发生的联合概率分布乘以 Xt，X0条件下的Xt-1的条件概率分布，就等于X0，Xt-1，Xt三者同时发生的联合概率分布，因此换算一下，三者的联合概率分布除以Xt，X0的联合概率分布就是初始目标分布。然后再进一步，两个联合概率，还可以展开成条件发生的概率乘以该条件下剩下目标发生的概率。我们发现分解后分布，都是前面能计算的分布。而这三个分布都是高斯分布，经过互相乘法除法之后，总的分布也是高斯分布，并且分布的均值和方差可求。而计算后的方差其实是固定系数组成的，可以看成常数，因此我们神经网络新的建模就可以选择对分布的均值进行建模。而均值包含x0，也是未知的，所幸前面正向加噪的公式中，我们可以把X0用Xt表示，最后均值化简如下（详细推导过程自行wiki）：
  
  ![](https://github.com/cryer/cryer.github.io/raw/master/image/141.jpg)
  
  因此可以看出，对均值建模实则就是对高斯噪声建模，因为其他系数都是常数，而xt是网络的输入。
  
  这里严格来说应该是要拟合`q(Xt-1 | Xt )`分布和网络预测`pθ(Xt-1| Xt )`分布之间的相似度，也就是KL散度。但是上面转换成了对两个高斯分布的均值的相似度的拟合，会不会有疑虑？事实就是方差可以假定为常量，而KL散度均值其实化简后除了（ε - εθ ）的均方误差之外，自然是有一个蛮复杂的系数的，只不过这个系数只跟α有关，跟方差一样，可以舍去，事实上也是舍去效果更好。
  
  下个问题是我们输出的现在变成噪声了，反向扩散如何获得噪声图像呢？还是用到前面提到的`z=μ + σ*ε`，可以用0-1高斯分布表示任何的高斯分布，现在网络输出噪声，代入到上面的均值公式中，就可以计算均值，所以你对照下采样sample代码中，最后计算输出使用的是不是正是这个均值公式。那方差怎么表示？前面我让读者记住一点，那就是正向加噪过程中，系数βt（即1-αt）就相当于方差，因此这里用的就是βt。
  
  这同时也解释了另一个常见问题，为什么要添加随机扰动，也就是一个0-1高斯噪声，因为必须添加，否则就破坏了噪声图像的高斯分布了（因为`z=μ + σ*ε`才能保证高斯分布），而且丧失了一定随机性。








