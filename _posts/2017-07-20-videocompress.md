---
layout: post
title: 视频是如何压缩的
description: 实现一个简单的视频压缩程序

---

### 导入

想象一个1080P（1920x1080）的视频，简单只考虑视频数据本身，一帧应该有2,073,600，也就是将近2M的像素，假设每个像素有RGB三色数据，每个颜色数据256个级别，也就是1个字节表示，那么这一帧的画面应该有大约6MB的数据大小，假设视频一秒30帧，那么一秒就有180MB的数据，一个10秒的视频就要近2个GB，这显然是很不合常识的，我们平时可能一部电影也就2个GB。那么视频为什么可以压缩这么多？是怎么做到的呢？



首先，我们需要注意视频的特点，那就是冗余性很大，不仅是空间冗余性，还有时间冗余性。空间上相邻像素一般很相似，时间上相邻帧甚至相邻很多帧，同位置或者靠近的位置一般很相似。对于这种，我们考虑是不是可以使用差分？也就是第一帧传输原来的图像，然后后面之传输差别的地方？这确实是一个很好的思想，实际上虽然视频压缩不严格采用差分，但是确实有着差分的思想。现代视频压缩一般采用**I帧，P帧和B帧**的方式：

- **I帧**：就是关键帧，这些帧我们一般采用完整的帧画面，但是也需要经过图像压缩。这个帧既作为随机访问点，比如你随机点击一秒就可以播放那时候的视频画面，而纯粹差分需要从第一帧开始计算。同时，这个帧也作为P帧和B帧的矫正。

- **P帧**：也就是预测帧，根据I帧或者前一个P帧进行预测，但是视频压缩领域中，不使用单纯的像素差别来预测，而是使用运动估计。这个帧不会存储实际的画面像素，只会存储运动矢量和残差，所以如果不依赖其它帧，是无法得到这个帧真正的像素数据的，所谓残差就是预测像素和实际像素的差，因为预测不可能百分百准确，所以需要残差进行矫正。且同样也不是直接存储矢量和残差，同样需要进行编码压缩。

- **B帧**:和P帧的区别是，P帧是通过前面的帧预测，而B帧是通过两边的帧预测，以便是两边的P帧，预测中间的B帧，比如IPBP的帧序列，就需要先通过I帧预测第一个P帧，然后下一个要根据第一个P帧预测第2个P帧，然后再通过2个P帧预测中间的B帧，B帧本身不能作为参考帧。可以看出B帧的解码是比较复杂的，也是比较耗时的，但是压缩率是最高的。

其他像还有量化，可以通过降低数据的精度，来节省存储的空间，比如可以对残差进行量化。实际工业级的压缩算法非常复杂，还涉及到很多细节，这里本博客主要实现一个简单的视频压缩程序，来展示视频压缩的大概工作方式，程序本身属于玩具程序，不要用于专业的压缩。



主要特性如下：

- 只实现**I帧P帧**，**B帧**相对复杂，就不实现了，实际的压缩算法一般是IPPBPPIPPBPP这样的帧排列方式，I帧一般2秒左右存储一次。这里就简单使用IPPPPIPPPP每5帧一个循环的方式进行帧的排列。

- **I帧**完整存储，不依赖其他帧，只使用帧内预测。编码方式类似于 JPEG：将图像划分为块（如 4×4 或 16×16），利用空间相邻像素的相关性进行预测（例如“左边像素值”预测当前块），然后对预测残差进行变换、量化、熵编码。

- **P帧**整像素精度的全搜索运动估计

- 原始视频格式一般是**YUV**或者**RAW**,这里以YUV为例，可以从网上下载，也可以用`ffmpeg`直接从一个mp4格式中提取，使用命令`ffmpeg -i input.mp4 -pix_fmt yuv420p -f rawvideo output.yuv`提取，所以上面的I帧的左像素预测之类都是针对的YUV，就是Y通道，UV通道一般直接量化压缩即可，因为人眼对色度信息不敏感，因此也可以采用较大的量化步长，也就是较大的`q_step(代码中)`。

- 自定义二进制序列化存储，直接使用python的pickle库也是可以的

- 运动矢量采用简单的zlib压缩，实际压缩方式，比如H264采用`CABAC/CAVLC`之类的熵编码方式

### 完整代码

```python


import numpy as np
import zlib
import os
import cv2



# ==============================
# 第一部分：生成测试视频（整像素移动）
# ==============================

def generate_test_video(width=320, height=240, num_frames=15, block_size=40):
    """
    生成白色方块整像素移动的测试视频
    每帧移动 16 像素，确保运动估计完美匹配
    """
    frames = []
    for i in range(num_frames):
        y = np.zeros((height, width), dtype=np.uint8)
        u = np.full((height // 2, width // 2), 128, dtype=np.uint8)
        v = np.full((height // 2, width // 2), 128, dtype=np.uint8)
        # 每帧移动 16 像素（整数，且是 16x16 块的倍数）
        x_pos = (i * 16) % (width - block_size)
        y_pos = height // 2 - block_size // 2
        y[y_pos:y_pos + block_size, x_pos:x_pos + block_size] = 255
        frames.append((y, u, v))
    return frames


# ==============================
# 第二部分：I 帧编码/解码（Y 平面）
# ==============================

def encode_i_frame(y_plane):
    """I 帧编码：左像素预测 + 残差"""
    height, width = y_plane.shape
    pred = np.zeros_like(y_plane, dtype=np.int16)
    pred[:, 0] = 128
    if width > 1:
        pred[:, 1:] = y_plane[:, :-1]
    residual = y_plane.astype(np.int16) - pred
    return residual


def decode_i_frame(residual):
    """I 帧解码：重建 Y 平面"""
    height, width = residual.shape
    y_rec = np.zeros((height, width), dtype=np.int16)
    y_rec[:, 0] = 128 + residual[:, 0]
    for x in range(1, width):
        y_rec[:, x] = y_rec[:, x - 1] + residual[:, x]
    return np.clip(y_rec, 0, 255).astype(np.uint8)


# ==============================
# 第三部分：P 帧编码/解码（Y 平面）
# ==============================

def motion_search(current_block, ref_frame, ref_y, ref_x, search_range=8):
    """全搜索运动估计"""
    block_h, block_w = current_block.shape
    best_sad = float('inf')
    best_mv = (0, 0)
    best_pred = None

    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            cand_y, cand_x = ref_y + dy, ref_x + dx
            if cand_y < 0 or cand_x < 0:
                continue
            if cand_y + block_h > ref_frame.shape[0] or cand_x + block_w > ref_frame.shape[1]:
                continue
            pred_block = ref_frame[cand_y:cand_y + block_h, cand_x:cand_x + block_w]
            sad = np.sum(np.abs(current_block.astype(np.int16) - pred_block.astype(np.int16)))
            if sad < best_sad:
                best_sad = sad
                best_mv = (dx, dy)
                best_pred = pred_block

    if best_pred is None:
        best_pred = np.zeros_like(current_block)
    residual = current_block.astype(np.int16) - best_pred.astype(np.int16)
    return best_mv, residual


def encode_p_frame(curr_y, ref_y, block_size=16, search_range=8):
    """
    P 帧编码：返回 numpy 格式的 MV 数组
    """
    height, width = curr_y.shape
    num_blocks = (height // block_size) * (width // block_size)
    mv_array = np.zeros((num_blocks, 2), dtype=np.int16)  # dx, dy
    residual_list = []

    idx = 0
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            curr_block = curr_y[y:y + block_size, x:x + block_size]
            mv, residual = motion_search(curr_block, ref_y, y, x, search_range)
            mv_array[idx] = mv
            residual_list.append(residual)
            idx += 1
    return mv_array, residual_list


def decode_p_frame(mv_list, residual_list, ref_y, block_size=16):
    """P 帧解码：接收 MV 列表和残差列表"""
    height, width = ref_y.shape
    y_rec = np.zeros((height, width), dtype=np.int16)
    idx = 0
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            dx, dy = mv_list[idx]
            residual = residual_list[idx]
            pred_block = np.zeros_like(residual)
            ref_y_pos, ref_x_pos = y + dy, x + dx
            if (ref_y_pos >= 0 and ref_x_pos >= 0 and
                ref_y_pos + block_size <= ref_y.shape[0] and
                ref_x_pos + block_size <= ref_y.shape[1]):
                pred_block = ref_y[ref_y_pos:ref_y_pos + block_size,
                                   ref_x_pos:ref_x_pos + block_size]
            rec_block = pred_block.astype(np.int16) + residual
            y_rec[y:y + block_size, x:x + block_size] = rec_block
            idx += 1
    return np.clip(y_rec, 0, 255).astype(np.uint8)


# ==============================
# 第四部分：量化函数
# ==============================

def quantize_residual(residual, q_step=4):
    """量化残差"""
    q_residual = np.round(residual / q_step).astype(np.int16)   
    return q_residual


def dequantize_residual(q_residual, q_step=4):
    """反量化残差"""
    return (q_residual * q_step).astype(np.int16)


def quantize_uv_plane(plane, q_step=4):
    """量化 U/V 平面"""
    # 使用 round 而不是直接除法截断
    q_plane = np.round(plane / q_step).astype(np.uint8) * q_step
    return q_plane


# ==============================
# 第五部分：高效二进制压缩/解压
# ==============================

def compress_video_to_binary(frames, gop_size=5, block_size=16, q_step_y=64, q_step_uv=8, output_file="compressed.bin"):
    """
    将视频压缩为二进制文件（包含 Y/U/V 压缩）
    """
    with open(output_file, "wb") as f:
        # 文件头
        width, height = frames[0][0].shape[1], frames[0][0].shape[0]
        num_frames = len(frames)
        f.write(width.to_bytes(4, 'little'))
        f.write(height.to_bytes(4, 'little'))
        f.write(num_frames.to_bytes(4, 'little'))
        f.write(gop_size.to_bytes(4, 'little'))
        f.write(q_step_y.to_bytes(4, 'little'))
        f.write(q_step_uv.to_bytes(4, 'little'))
        
        reconstructed_cache = []
        
        for i, (y, u, v) in enumerate(frames):
            if i % gop_size == 0:
                # I 帧
                print(f"  帧 {i}: 编码为 I 帧")
                f.write((0).to_bytes(1, 'little'))  # 帧类型: 0=I, 1=P
                residual = encode_i_frame(y)
                q_residual = quantize_residual(residual, q_step_y)
                comp_residual = zlib.compress(q_residual.astype(np.int16).tobytes())
                f.write(len(comp_residual).to_bytes(4, 'little'))  # 残差长度
                f.write(comp_residual)
                
                # 量化并压缩 U/V 平面
                q_u = quantize_uv_plane(u, q_step_uv)
                q_v = quantize_uv_plane(v, q_step_uv)
                comp_u = zlib.compress(q_u.tobytes())
                comp_v = zlib.compress(q_v.tobytes())
                f.write(len(comp_u).to_bytes(4, 'little'))
                f.write(len(comp_v).to_bytes(4, 'little'))
                f.write(comp_u)
                f.write(comp_v)
                
                # 重建帧
                dq_residual = dequantize_residual(q_residual, q_step_y)
                y_rec = decode_i_frame(dq_residual)
                reconstructed_cache.append(y_rec)
                
            else:
                # P 帧
                print(f"  帧 {i}: 编码为 P 帧")
                f.write((1).to_bytes(1, 'little'))  # 帧类型: P
                ref_y = reconstructed_cache[-1]
                mv_array, residual_list = encode_p_frame(y, ref_y, block_size)
                q_residual_list = [quantize_residual(r, q_step_y) for r in residual_list]
                
                # 压缩残差
                flat = np.concatenate([r.flatten() for r in q_residual_list])
                comp_residuals = zlib.compress(flat.astype(np.int16).tobytes())
                
                # 压缩 MV
                comp_mv = zlib.compress(mv_array.tobytes())
                
                # 写入数据
                f.write(len(comp_mv).to_bytes(4, 'little'))
                f.write(comp_mv)
                f.write(len(comp_residuals).to_bytes(4, 'little'))
                f.write(comp_residuals)
                
                # 量化并压缩 U/V 平面
                q_u = quantize_uv_plane(u, q_step_uv)
                q_v = quantize_uv_plane(v, q_step_uv)
                comp_u = zlib.compress(q_u.tobytes())
                comp_v = zlib.compress(q_v.tobytes())
                f.write(len(comp_u).to_bytes(4, 'little'))
                f.write(len(comp_v).to_bytes(4, 'little'))
                f.write(comp_u)
                f.write(comp_v)
                
                # 重建帧
                dq_residual_list = [dequantize_residual(r, q_step_y) for r in q_residual_list]
                mv_list = [tuple(mv) for mv in mv_array]
                y_rec = decode_p_frame(mv_list, dq_residual_list, ref_y, block_size)
                reconstructed_cache.append(y_rec)
    
    print(f"二进制压缩数据已保存到 {output_file}")


def decompress_video_from_binary(input_file="compressed.bin"):
    """
    从二进制文件解压视频（包含 Y/U/V 解压）
    """
    with open(input_file, "rb") as f:
        # 读取文件头
        width = int.from_bytes(f.read(4), 'little')
        height = int.from_bytes(f.read(4), 'little')
        num_frames = int.from_bytes(f.read(4), 'little')
        gop_size = int.from_bytes(f.read(4), 'little')
        q_step_y = int.from_bytes(f.read(4), 'little')
        q_step_uv = int.from_bytes(f.read(4), 'little')
        
        frames = []
        last_rec_y = None
        
        for i in range(num_frames):
            frame_type = int.from_bytes(f.read(1), 'little')  # 0=I, 1=P
            
            if frame_type == 0:  # I 帧
                # 解压 Y 残差
                comp_len = int.from_bytes(f.read(4), 'little')
                comp_residual = f.read(comp_len)
                residual_bytes = zlib.decompress(comp_residual)
                residual = np.frombuffer(residual_bytes, dtype=np.int16).reshape((height, width))
                dq_residual = dequantize_residual(residual, q_step_y)
                y_rec = decode_i_frame(dq_residual)
                
                # 解压 U/V
                u_len = int.from_bytes(f.read(4), 'little')
                v_len = int.from_bytes(f.read(4), 'little')
                comp_u = f.read(u_len)
                comp_v = f.read(v_len)
                u_bytes = zlib.decompress(comp_u)
                v_bytes = zlib.decompress(comp_v)
                u = np.frombuffer(u_bytes, dtype=np.uint8).reshape((height//2, width//2))
                v = np.frombuffer(v_bytes, dtype=np.uint8).reshape((height//2, width//2))
                
                frames.append((y_rec, u, v))
                last_rec_y = y_rec
                
            else:  # P 帧
                # 解压 MV
                mv_len = int.from_bytes(f.read(4), 'little')
                comp_mv = f.read(mv_len)
                mv_bytes = zlib.decompress(comp_mv)
                mv_array = np.frombuffer(mv_bytes, dtype=np.int16).reshape(-1, 2)
                mv_list = [tuple(mv) for mv in mv_array]
                
                # 解压 Y 残差
                res_len = int.from_bytes(f.read(4), 'little')
                comp_residuals = f.read(res_len)
                res_bytes = zlib.decompress(comp_residuals)
                num_blocks = (height // 16) * (width // 16)
                block_pixels = 16 * 16
                flat = np.frombuffer(res_bytes, dtype=np.int16)
                residual_list = [flat[i*block_pixels:(i+1)*block_pixels].reshape((16,16)) for i in range(num_blocks)]
                dq_residual_list = [dequantize_residual(r, q_step_y) for r in residual_list]
                
                # 重建 Y
                y_rec = decode_p_frame(mv_list, dq_residual_list, last_rec_y, 16)
                
                # 解压 U/V
                u_len = int.from_bytes(f.read(4), 'little')
                v_len = int.from_bytes(f.read(4), 'little')
                comp_u = f.read(u_len)
                comp_v = f.read(v_len)
                u_bytes = zlib.decompress(comp_u)
                v_bytes = zlib.decompress(comp_v)
                u = np.frombuffer(u_bytes, dtype=np.uint8).reshape((height//2, width//2))
                v = np.frombuffer(v_bytes, dtype=np.uint8).reshape((height//2, width//2))
                
                frames.append((y_rec, u, v))
                last_rec_y = y_rec
    
    print(f"从 {input_file} 解压了 {num_frames} 帧")
    return frames


# ==============================
# 第六部分：保存视频为 MP4
# ==============================

def save_frames_as_mp4(frames, filename, fps=10):
    """
    将 YUV420 格式的帧列表保存为 MP4 视频
    参数：
        frames: [(Y, U, V), ...] 列表
        filename: 输出 MP4 文件名，如 "original.mp4"
        fps: 帧率（默认 10）
    """
    if not frames:
        return
    height, width = frames[0][0].shape
    # 创建 VideoWriter（使用 MP4V 编码器）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for y, u, v in frames:
        # 将 YUV420 转为 BGR（OpenCV 需要 BGR）
        # 先上采样 U/V 到 Y 的尺寸
        u_up = cv2.resize(u, (width, height), interpolation=cv2.INTER_NEAREST)
        v_up = cv2.resize(v, (width, height), interpolation=cv2.INTER_NEAREST)
        # 合并为 YUV 三通道
        yuv = np.stack([y, u_up, v_up], axis=2)
        # 转为 BGR
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        # 写入
        out.write(bgr)

    out.release()
    print(f"视频已保存为 {filename}")


def read_yuv_file(filename, width, height, num_frames, block_size=16):
    """
    读取 YUV420P 文件并裁剪到块大小的整数倍
    """
    # 计算裁剪后的尺寸（保证是 block_size 的整数倍）
    crop_width = (width // block_size) * block_size
    crop_height = (height // block_size) * block_size
    
    if crop_width != width or crop_height != height:
        print(f"  裁剪分辨率: {width}x{height} → {crop_width}x{crop_height} (块大小={block_size})")
    
    frame_size = width * height * 3 // 2
    with open(filename, 'rb') as f:
        raw_bytes = f.read()
    
    total_size = frame_size * num_frames
    if len(raw_bytes) < total_size:
        raise ValueError(f"文件大小不足，期望 {total_size}, 实际 {len(raw_bytes)}")
    
    frames = []
    for i in range(num_frames):
        start = i * frame_size
        # Y 平面
        y_raw = np.frombuffer(raw_bytes[start:start + width * height], dtype=np.uint8).reshape((height, width))
        y = y_raw[:crop_height, :crop_width]
        
        # U/V 平面（按比例裁剪）
        u_start = start + width * height
        u_raw = np.frombuffer(raw_bytes[u_start:u_start + (width//2)*(height//2)], dtype=np.uint8).reshape((height//2, width//2))
        u_crop_width = crop_width // 2
        u_crop_height = crop_height // 2
        u = u_raw[:u_crop_height, :u_crop_width]
        
        v_start = u_start + (width//2)*(height//2)
        v_raw = np.frombuffer(raw_bytes[v_start:v_start + (width//2)*(height//2)], dtype=np.uint8).reshape((height//2, width//2))
        v = v_raw[:u_crop_height, :u_crop_width]
        
        frames.append((y, u, v))
    
    print(f"从 {filename} 读取了 {num_frames} 帧，分辨率 {crop_width}x{crop_height}")
    return frames


if __name__ == "__main__":
    # === 选择输入方式 ===
    use_yuv_file = True  # 设置为 True 使用 YUV 文件，False 使用生成的测试视频
    
    if use_yuv_file:
        # 从 YUV 文件读取
        yuv_filename = "demo.yuv"  # 替换为你的 YUV 文件路径
        width, height = 852, 480         # 替换为你的视频分辨率
        num_frames = 30                  # 替换为你的视频帧数
        
        print(f"1. 从 {yuv_filename} 读取 YUV 视频...")
        original_frames = read_yuv_file(yuv_filename, width, height, num_frames)
    else:
        # 1.生成测试视频
        print("1. 生成测试视频...")
        original_frames = generate_test_video(width=320, height=240, num_frames=15, block_size=40)

        # 保存原始 YUV（无压缩）
        with open("original.yuv", "wb") as f:
            for y, u, v in original_frames:
                f.write(y.tobytes())
                f.write(u.tobytes())
                f.write(v.tobytes())

    # 2. 压缩为二进制文件（包含 Y/U/V 压缩）
    print("\n2. 压缩视频...")
    compress_video_to_binary(
        original_frames, 
        gop_size=5, 
        q_step_y=64, 
        q_step_uv=8,  # U/V 用较小的量化步长
        output_file="compressed.bin"
    )

    # 3. 解压
    print("\n3. 解压视频...")
    reconstructed_frames = decompress_video_from_binary("compressed.bin")

    # 4. 保存为 MP4
    save_frames_as_mp4(reconstructed_frames, "reconstructed.mp4", fps=5)

    # 5. 验证质量
    print("\n4. 验证重建质量...")
    orig_y0 = original_frames[0][0]
    rec_y0 = reconstructed_frames[0][0]
    error0 = np.mean(np.abs(orig_y0.astype(np.float32) - rec_y0.astype(np.float32)))
    print(f"  I 帧（第0帧）平均绝对误差: {error0:.2f}")

    orig_y_last = original_frames[-1][0]
    rec_y_last = reconstructed_frames[-1][0]
    error_last = np.mean(np.abs(orig_y_last.astype(np.float32) - rec_y_last.astype(np.float32)))
    print(f"  P 帧（最后一帧）平均绝对误差: {error_last:.2f}")

    # 6. 计算压缩率
    print("\n5. 压缩效果:")
    original_size = len(original_frames) * 320 * 240 * 1.5  # YUV420: 1.5 bytes/pixel
    compressed_size = os.path.getsize("compressed.bin")
    print(f"  原始大小: {original_size / 1024:.1f} KB")
    print(f"  压缩后: {compressed_size / 1024:.1f} KB")
    if compressed_size > 0:
        print(f"  压缩率: {original_size / compressed_size:.1f} : 1")
    else:
        print("  压缩后大小为 0，异常！")

    print("\n✅ 程序运行完成！")
```



### 效果

这里先用程序生成简单的15帧视频作为测试，输出如下：

```
1. 生成测试视频...

2. 压缩视频...
  帧 0: 编码为 I 帧
  帧 1: 编码为 P 帧
  帧 2: 编码为 P 帧
  帧 3: 编码为 P 帧
  帧 4: 编码为 P 帧
  帧 5: 编码为 I 帧
  帧 6: 编码为 P 帧
  帧 7: 编码为 P 帧
  帧 8: 编码为 P 帧
  帧 9: 编码为 P 帧
  帧 10: 编码为 I 帧
  帧 11: 编码为 P 帧
  帧 12: 编码为 P 帧
  帧 13: 编码为 P 帧
  帧 14: 编码为 P 帧
二进制压缩数据已保存到 compressed.bin

3. 解压视频...
从 compressed.bin 解压了 15 帧
视频已保存为 reconstructed.mp4

4. 验证重建质量...
  I 帧（第0帧）平均绝对误差: 0.00
  P 帧（最后一帧）平均绝对误差: 0.00

5. 压缩效果:
  原始大小: 1687.5 KB
  压缩后: 6.4 KB
  压缩率: 262.1 : 1

✅ 程序运行完成！
```

可以看到残差都是0，这是因为这个例子就是简单的白色方框右移的例子，非常好预测，根本没有误差，等下我们可以看到实际的复杂的视频，运动估计还是有误差的，不够这也是程序简单的问题，工业级的压缩算法其实预测非常准确，残差很小。同时我们看到压缩率超过了260倍。

压缩前后的视频如下：

<table>
    <tr>
        <td><img src="https://github.com/cryer/cryer.github.io/raw/master/image/video3.gif" alt="Image 1" width="500"></td>
        <td><img src="https://github.com/cryer/cryer.github.io/raw/master/image/video4.gif" alt="Image 2" width="500"></td>
    </tr>
</table>


可以看到对于简单的视频来说，因为没有误差，所以是一摸一样的。下面用一个复杂的例子再看一下（一秒的视频）：

```
1. 从 demo.yuv 读取 YUV 视频...
  裁剪分辨率: 852x480 → 848x480 (块大小=16)
从 demo.yuv 读取了 30 帧，分辨率 848x480

2. 压缩视频...
  帧 0: 编码为 I 帧
  帧 1: 编码为 P 帧
  帧 2: 编码为 P 帧
  帧 3: 编码为 P 帧
  帧 4: 编码为 P 帧
  帧 5: 编码为 I 帧
  帧 6: 编码为 P 帧
  帧 7: 编码为 P 帧
  帧 8: 编码为 P 帧
  帧 9: 编码为 P 帧
  帧 10: 编码为 I 帧
  帧 11: 编码为 P 帧
  帧 12: 编码为 P 帧
  帧 13: 编码为 P 帧
  帧 14: 编码为 P 帧
  帧 15: 编码为 I 帧
  帧 16: 编码为 P 帧
  帧 17: 编码为 P 帧
  帧 18: 编码为 P 帧
  帧 19: 编码为 P 帧
  帧 20: 编码为 I 帧
  帧 21: 编码为 P 帧
  帧 22: 编码为 P 帧
  帧 23: 编码为 P 帧
  帧 24: 编码为 P 帧
  帧 25: 编码为 I 帧
  帧 26: 编码为 P 帧
  帧 27: 编码为 P 帧
  帧 28: 编码为 P 帧
  帧 29: 编码为 P 帧
二进制压缩数据已保存到 compressed.bin

3. 解压视频...
从 compressed.bin 解压了 30 帧
视频已保存为 reconstructed.mp4

4. 验证重建质量...
  I 帧（第0帧）平均绝对误差: 54.54
  P 帧（最后一帧）平均绝对误差: 12.43

5. 压缩效果:
  原始大小: 17971.8 KB
  压缩后: 444.9 KB
  压缩率: 40.4 : 1

✅ 程序运行完成！
```

可以看到残差还是很大的，不过压缩率还是很不错的。下面看压缩前后的视频效果：

<table>
    <tr>
        <td><img src="https://github.com/cryer/cryer.github.io/raw/master/image/video1.gif" alt="Image 1" width="500"></td>
        <td><img src="https://github.com/cryer/cryer.github.io/raw/master/image/video2.gif" alt="Image 2" width="500"></td>
    </tr>
</table>

因为程序简单，所以还是有比较明显的失真和噪音的，但是整体上其实还是不错的。相信通过这个例子，读者可以对视频压缩的底层原理有更清晰的了解。




