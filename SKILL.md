---
name: gpu-scheduler
description: |
  GPU 集群调度器 — 当你需要查看 GPU 状态、提交任务到 GPU 队列、或管理多台 GPU 服务器时使用。
  触发场景：
  - 查看所有 GPU 服务器的状态
  - 找空闲 GPU 跑训练/推理任务
  - 管理 GPU 任务队列
  - 测试 GPU 服务器 SSH 连接
---

# GPU Scheduler

轻量级多服务器 GPU 调度器。纯 SSH，零部署（GPU 服务器不需要安装任何东西）。

## 前置条件

- 开发机上已安装 `gpu-scheduler`：`pipx install gpu-scheduler`
- 配置文件 `~/.config/gpu-scheduler/config.toml` 已编辑（服务器列表、SSH key）

## 配置管理

生成示例配置：
```bash
gpu-scheduler config init
```

查看当前配置：
```bash
gpu-scheduler config show
```

测试所有服务器 SSH 连接：
```bash
gpu-scheduler config test
```

配置文件格式 (`~/.config/gpu-scheduler/config.toml`)：
```toml
[[servers]]
host = "gpu-server-1"
port = 22
user = "your-username"
key_file = "~/.ssh/id_rsa"

[[servers]]
host = "gpu-server-2"
port = 22
user = "your-username"
key_file = "~/.ssh/id_rsa"

[scheduler]
poll_interval = 5.0
```

## 核心命令

### 查看 GPU 集群状态

```bash
gpu-scheduler list
```

持续监控：
```bash
gpu-scheduler list --watch --interval 3
```

### 提交任务到队列

```bash
gpu-scheduler submit python train.py --gpus 2 --priority 0
```

提交后需要启动 daemon 来消费队列：
```bash
gpu-scheduler daemon
```

### 一键运行（提交 + 等待完成）

```bash
gpu-scheduler run python train.py --gpus 2 --wait
```

### 查看队列

```bash
gpu-scheduler queue
gpu-scheduler queue --status pending
```

### 取消任务

```bash
gpu-scheduler cancel 3
```

## 调度器 Daemon

在一个独立终端/会话中运行：
```bash
gpu-scheduler daemon
```

单次调度（处理完所有 pending 任务后退出）：
```bash
gpu-scheduler daemon --once
```

## 工作原理

1. 用户通过 `submit` 或 `run` 命令提交任务 → SQLite 队列
2. `daemon` 轮询队列，按优先级 + FIFO 取出任务
3. 通过 SSH 并行查询所有 GPU 服务器状态（`nvidia-smi --query-gpu=... --format=csv`）
4. 找到有足够空闲 GPU 的服务器，SSH 上去执行用户命令（设置 `CUDA_VISIBLE_DEVICES`）
5. 更新任务状态（完成/失败）

## 任务优先级

数字越小优先级越高（默认 0）。高优先级任务优先调度。
```bash
gpu-scheduler submit important_job.sh --priority -10
gpu-scheduler submit low_priority.sh --priority 100
```

## 多 GPU 任务

指定需要的 GPU 数量：
```bash
gpu-scheduler submit multi_gpu_train.sh --gpus 4
```

## 显存要求

设置最低显存（MB），调度器只会分配满足要求的 GPU：
```bash
gpu-scheduler submit large_model.py --gpu-memory 24000
```

## 限制

- 纯 SSH 方案：GPU 服务器不需要安装任何软件，但需要 SSH key 认证
- 仅支持 Linux GPU 服务器（依赖 nvidia-smi）
- 调度器 daemon 需要在前台运行（可配合 tmux/screen/systemd）
- 不支持跨服务器分配 GPU（一个任务的所有 GPU 必须在同一台服务器上）
- 当前版本不追踪内存中正在运行的进程（只检查 GPU 显存占用）
