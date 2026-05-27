"""GPU 数据模型."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProcessInfo:
    """GPU 上运行的进程信息."""

    pid: int
    name: str
    gpu_memory_mb: int
    owner: str = ""  # 通过 ps 获取


@dataclass
class GPUInfo:
    """单张 GPU 的完整状态."""

    index: int
    name: str
    server: str  # 所属服务器 host
    uuid: str = ""  # GPU UUID，用于精确映射进程

    # 显存 (MB)
    memory_total: int = 0
    memory_used: int = 0
    memory_free: int = 0

    # 利用率 (%)
    gpu_util: int = 0
    memory_util: int = 0

    # 其他
    temperature: int = 0
    power_draw: float = 0.0
    fan_speed: int = 0

    processes: list[ProcessInfo] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.processes is None:
            self.processes = []

    @property
    def is_free(self) -> bool:
        """判断 GPU 是否空闲（显存占用 < 500MB 且无进程）."""
        return self.memory_used < 500 and len(self.processes) == 0

    @property
    def memory_usage_pct(self) -> float:
        """显存使用百分比."""
        if self.memory_total == 0:
            return 0.0
        return (self.memory_used / self.memory_total) * 100
