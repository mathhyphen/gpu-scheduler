# GPU Scheduler

轻量级多服务器 GPU 调度器。纯 SSH，零部署 —— GPU 服务器不需要安装任何软件。

> 📺 **操作演示**：[gist.githack.com 直接预览](https://gist.githack.com/mathhyphen/87abca430dc683ebc72c3eea67ae50c2/raw/gpu-scheduler-demo.html)

## 特性

- **零部署**：GPU 服务器不需要安装任何 agent，纯 SSH + nvidia-smi
- **并行查询**：asyncio + asyncssh 同时查询所有服务器
- **任务队列**：SQLite 持久化，FIFO + 优先级调度
- **Rich 终端**：彩色 GPU 状态表格，实时刷新（`--watch`）
- **简单安装**：`pipx install gpu-scheduler`
- **DeepSeek TUI Skill**：附带 SKILL.md，在 TUI 里跟 AI 说「帮我找张空闲 GPU 跑训练」即可自动调度

## 安装

```bash
pipx install gpu-scheduler
```

或开发安装：

```bash
git clone https://github.com/mathhyphen/gpu-scheduler.git
cd gpu-scheduler
pip install -e .
```

## 快速开始

```bash
# 1. 生成配置文件
gpu-scheduler config init

# 2. 编辑配置，填入你的 GPU 服务器信息
vim ~/.config/gpu-scheduler/config.toml

# 3. 测试 SSH 连接
gpu-scheduler config test

# 4. 查看所有 GPU 状态
gpu-scheduler list

# 5. 启动调度 daemon（另一个终端）
gpu-scheduler daemon

# 6. 提交任务
gpu-scheduler submit python train.py --gpus 2 --priority 0
```

## 命令

| 命令 | 说明 |
|------|------|
| `gpu-scheduler list` | 查看所有 GPU 状态（`--watch` 实时刷新） |
| `gpu-scheduler submit <cmd>` | 提交任务到队列（`--gpus` `--priority` `--gpu-memory`） |
| `gpu-scheduler run <cmd> --wait` | 提交任务并阻塞等待完成 |
| `gpu-scheduler queue` | 查看任务队列（`--status pending` 筛选） |
| `gpu-scheduler cancel <id>` | 取消等待中的任务 |
| `gpu-scheduler daemon` | 启动调度 daemon（`--once` 单轮） |
| `gpu-scheduler config init` | 生成示例配置 |
| `gpu-scheduler config show` | 查看当前配置 |
| `gpu-scheduler config test` | 测试所有服务器 SSH 连接 |

别名：`gpu-sched` 等价于 `gpu-scheduler`。

## 配置

```toml
[[servers]]
host = "gpu-server-1"
port = 22
user = "your-username"
key_file = "~/.ssh/id_rsa"
# labels = { project = "nlp", gpu_type = "a100" }

[[servers]]
host = "gpu-server-2"
port = 22
user = "your-username"
key_file = "~/.ssh/id_rsa"

[scheduler]
poll_interval = 5.0
```

### 配置文件搜索路径（优先级从高到低）

1. `$GPU_SCHEDULER_CONFIG` 环境变量
2. 当前目录 `gpu-scheduler.toml` / `.gpu-scheduler.toml`
3. `~/.config/gpu-scheduler/config.toml`

## 工作原理

```
用户提交任务 → SQLite 队列 (PENDING)
       ↓
  Daemon 轮询队列（按优先级 + FIFO）
       ↓
  AsyncSSH 并行查询所有 GPU 服务器（nvidia-smi CSV）
       ↓
  找到同一服务器上满足显存要求的空闲 GPU
       ↓
  SSH 执行用户命令（自动设置 CUDA_VISIBLE_DEVICES）
       ↓
  更新任务状态（COMPLETED / FAILED）
```

## 项目结构

```
gpu_scheduler/
├── cli.py           # Typer CLI (8 个命令)
├── config.py        # TOML 配置管理
├── gpu/
│   ├── __init__.py  # GPUInfo / ProcessInfo 数据模型
│   └── query.py     # SSH + nvidia-smi CSV 解析 + UUID 进程映射
├── executor/
│   └── __init__.py  # SSH 远程执行 + CUDA_VISIBLE_DEVICES
├── scheduler/
│   ├── __init__.py  # Task / TaskStatus 数据模型
│   └── queue.py     # SQLite 持久化队列 + 调度循环
├── utils.py         # Rich 终端表格渲染
└── __init__.py
```

## 需求

- Python 3.10+
- Linux GPU 服务器（含 nvidia-smi）
- SSH key 认证

## 技术栈

| 组件 | 选型 | 理由 |
|------|------|------|
| CLI 框架 | Typer | 类型提示驱动，代码量最少 |
| 终端渲染 | Rich | 彩色表格、Live 刷新 |
| SSH | AsyncSSH | asyncio 原生，并行查询 |
| 任务队列 | SQLite | 零依赖持久化，WAL 模式 |
| 配置 | TOML | Python 3.11+ 内置 |
| 打包 | Hatchling + pipx | 一行安装 |

## 测试

```bash
python test_core.py
```

13 项单元测试覆盖：CSV 解析、空闲判断、GPU 分配、SQLite 队列操作、UUID 进程映射。

## 许可

MIT
