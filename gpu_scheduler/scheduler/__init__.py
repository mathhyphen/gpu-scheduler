"""调度任务数据模型."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime


class TaskStatus(str, enum.Enum):
    """任务状态."""

    PENDING = "pending"  # 等待调度
    RUNNING = "running"  # 正在执行
    COMPLETED = "completed"  # 成功完成
    FAILED = "failed"  # 执行失败
    CANCELLED = "cancelled"  # 被取消


@dataclass
class Task:
    """一个 GPU 任务."""

    id: int = 0  # 数据库自增 ID
    command: str = ""  # 要执行的命令
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0  # 数字越小优先级越高
    gpu_count: int = 1  # 需要几张 GPU
    gpu_ids: str = ""  # 分配的具体 GPU（逗号分隔，如 "0,1"）
    server: str = ""  # 分配到的服务器
    created_at: str = ""
    started_at: str = ""
    finished_at: str = ""
    exit_code: int | None = None
    output: str = ""  # 截断的 stdout/stderr

    # 用户可选的约束
    gpu_memory_min: int = 0  # 最低显存要求 (MB)
    prefer_labels: dict[str, str] = field(default_factory=dict)  # 偏好标签

    def to_row(self) -> dict:
        """转为数据库行."""
        return {
            "command": self.command,
            "status": self.status.value,
            "priority": self.priority,
            "gpu_count": self.gpu_count,
            "gpu_ids": self.gpu_ids,
            "server": self.server,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "output": self.output,
            "gpu_memory_min": self.gpu_memory_min,
        }

    @classmethod
    def from_row(cls, row: dict) -> Task:
        """从数据库行恢复."""
        return cls(
            id=row["id"],
            command=row["command"],
            status=TaskStatus(row["status"]),
            priority=row["priority"],
            gpu_count=row["gpu_count"],
            gpu_ids=row.get("gpu_ids", ""),
            server=row.get("server", ""),
            created_at=row.get("created_at", ""),
            started_at=row.get("started_at", ""),
            finished_at=row.get("finished_at", ""),
            exit_code=row.get("exit_code"),
            output=row.get("output", ""),
            gpu_memory_min=row.get("gpu_memory_min", 0),
        )
