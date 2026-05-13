---
layout: post
title: 自制简单的3D游戏引擎
description: C++开发简单的3D游戏引擎

---

## 简单的3D游戏引擎

一个轻量级、模块化、易于扩展的3D游戏引擎原型，既可作为学习参考，也能作为小型3D游戏项目的开发起点。基于C++和openGL，以及少量的第三方库，比如`ImGui,Assimp,json，stb...`等。

项目地址：[点我](https://github.com/cryer/hako/tree/master/game_start)


### 主要实现模块

- 资源管理模块
- 相机，着色器，3d模型加载模块
- 简单碰撞检测模块
- 渲染模块
- 音频模块
- 光照系统和阴影系统
- 数据驱动关卡管理模块
- 地图编辑器

![](https://github.com/cryer/hako/raw/master/docs/res/map_editor.jpg)

- 悬浮菜单

![](https://github.com/cryer/hako/raw/master/docs/res/floatting_menu.jpg)

- 游戏内调试终端

![](https://github.com/cryer/hako/raw/master/docs/res/terminal.jpg)

### 基本性能优化算法
- 视锥体剔除
- 四叉树空间划分
- 可视距离渲染剔除
- 批处理
- GPU顶点缓存优化 + 物理顺序重排
- LOD
