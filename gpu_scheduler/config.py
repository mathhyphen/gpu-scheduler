"""配置管理 — TOML 格式，XDG 规范路径."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "gpu-scheduler" / "config.toml"


@dataclass
class ServerConfig:
    """单台 GPU 服务器的连接配置."""

    host: str
    port: int = 22
    user: str = ""
    key_file: str = ""  # SSH 私钥路径，空则用默认 (~/.ssh/id_rsa)
    gpu_count: int = 0  # 0 = 自动检测
    labels: dict[str, str] = field(default_factory=dict)  # 自定义标签


@dataclass
class SchedulerConfig:
    """调度器配置."""

    poll_interval: float = 5.0  # 调度循环轮询间隔（秒）
    db_path: str = ""  # SQLite 数据库路径，空则用默认


@dataclass
class Config:
    """全局配置."""

    servers: list[ServerConfig] = field(default_factory=list)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


def _default_db_path() -> str:
    return str(Path.home() / ".local" / "share" / "gpu-scheduler" / "queue.db")


def _find_config_file() -> Path | None:
    """按优先级搜索配置文件."""
    candidates = [
        Path(os.environ.get("GPU_SCHEDULER_CONFIG", "")),
        Path.cwd() / "gpu-scheduler.toml",
        Path.cwd() / ".gpu-scheduler.toml",
        DEFAULT_CONFIG_PATH,
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def load_config(path: str | None = None) -> Config:
    """加载配置，失败则返回默认空配置."""
    config_path: Path | None = Path(path) if path else _find_config_file()

    if config_path is None or not config_path.exists():
        return Config()

    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))

    servers = []
    for s in raw.get("servers", []):
        servers.append(
            ServerConfig(
                host=s["host"],
                port=s.get("port", 22),
                user=s.get("user", ""),
                key_file=s.get("key_file", ""),
                gpu_count=s.get("gpu_count", 0),
                labels=s.get("labels", {}),
            )
        )

    sched_raw = raw.get("scheduler", {})
    scheduler = SchedulerConfig(
        poll_interval=sched_raw.get("poll_interval", 5.0),
        db_path=sched_raw.get("db_path", _default_db_path()),
    )

    return Config(servers=servers, scheduler=scheduler)


def generate_example_config() -> str:
    """生成示例配置文件内容."""
    return '''# GPU Scheduler 配置文件
# 默认路径: ~/.config/gpu-scheduler/config.toml

[[servers]]
host = "gpu-server-1"
port = 22
user = "your-username"
key_file = "~/.ssh/id_rsa"
# gpu_count = 8  # 可选：手动指定 GPU 数量，0 = 自动检测

[[servers]]
host = "gpu-server-2"
port = 22
user = "your-username"
key_file = "~/.ssh/id_rsa"

# [[servers]]
# host = "gpu-server-3"
# port = 2222
# user = "admin"
# key_file = "~/.ssh/gpu_key"
# labels = { project = "nlp", gpu_type = "a100" }

[scheduler]
poll_interval = 5.0  # 调度轮询间隔（秒）
# db_path = "/path/to/queue.db"
'''
