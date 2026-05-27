"""通过 SSH + nvidia-smi CSV 查询远程 GPU 状态."""

from __future__ import annotations

import asyncio
import csv
import io
from pathlib import Path

import asyncssh

from gpu_scheduler.config import Config, ServerConfig
from gpu_scheduler.gpu import GPUInfo, ProcessInfo

# ── nvidia-smi 查询模板 ─────────────────────────────

GPU_QUERY_COLS = [
    "index",
    "uuid",
    "name",
    "temperature.gpu",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "memory.free",
    "power.draw",
    "fan.speed",
]

GPU_QUERY_CMD = (
    "nvidia-smi --query-gpu="
    + ",".join(GPU_QUERY_COLS)
    + " --format=csv,noheader,nounits"
)

COMPUTE_APPS_COLS = ["pid", "process_name", "used_memory", "gpu_uuid"]

PROCESS_QUERY_CMD = (
    "nvidia-smi --query-compute-apps="
    + ",".join(COMPUTE_APPS_COLS)
    + " --format=csv,noheader,nounits"
)

# 获取进程 owner 的命令模板
PROCESS_OWNER_CMD = "ps -o user= -p {pid} 2>/dev/null || echo ''"


def _resolve_key(key_file: str) -> str | None:
    """解析 SSH 密钥路径，展开 ~."""
    if not key_file:
        return None
    return str(Path(key_file).expanduser())


# ── 纯解析函数（可独立测试）─────────────────────────

def parse_gpu_csv(csv_text: str, server_host: str) -> list[GPUInfo]:
    """解析 nvidia-smi GPU CSV 输出为 GPUInfo 列表."""
    gpus: list[GPUInfo] = []
    reader = csv.reader(io.StringIO(csv_text))
    for row in reader:
        if len(row) < len(GPU_QUERY_COLS):
            continue
        # 列序: index, uuid, name, temp, gpu_util, mem_util, mem_used, mem_total, mem_free, power, fan
        gpu = GPUInfo(
            index=int(row[0].strip()),
            uuid=row[1].strip(),
            name=row[2].strip(),
            server=server_host,
            temperature=_parse_int(row[3]),
            gpu_util=_parse_int(row[4]),
            memory_util=_parse_int(row[5]),
            memory_used=_parse_int(row[6]),
            memory_total=_parse_int(row[7]),
            memory_free=_parse_int(row[8]),
            power_draw=_parse_float(row[9]),
            fan_speed=_parse_int(row[10]),
        )
        gpus.append(gpu)
    return gpus


def parse_compute_apps_csv(
    csv_text: str, uuid_to_index: dict[str, int] | None = None
) -> dict[int, list[ProcessInfo]]:
    """解析 nvidia-smi compute-apps CSV 输出.

    列序: pid, process_name, used_memory, gpu_uuid
    通过 uuid_to_index 映射将进程分配到正确的 GPU.
    如果未提供 uuid_to_index，则尝试从第四列直接匹配.
    """
    processes: dict[int, list[ProcessInfo]] = {}
    reader = csv.reader(io.StringIO(csv_text))
    for row in reader:
        if len(row) < 4:
            continue
        pid = int(row[0].strip())
        name = row[1].strip()
        mem_str = row[2].strip()
        mem = int(mem_str.replace(" MiB", "")) if "MiB" in mem_str else _parse_int(mem_str)
        gpu_uuid = row[3].strip() if len(row) > 3 else ""

        # 通过 UUID 映射找到 GPU index
        gpu_idx = -1
        if uuid_to_index and gpu_uuid:
            # 尝试完整匹配
            gpu_idx = uuid_to_index.get(gpu_uuid, -1)
            if gpu_idx < 0:
                # 有些 nvidia-smi 版本 UUID 前缀可能不同，尝试子串匹配
                for uid, idx in uuid_to_index.items():
                    if gpu_uuid.startswith(uid[:12]) or uid.startswith(gpu_uuid[:12]):
                        gpu_idx = idx
                        break

        if gpu_idx < 0:
            gpu_idx = 0  # fallback

        if gpu_idx not in processes:
            processes[gpu_idx] = []
        processes[gpu_idx].append(ProcessInfo(pid=pid, name=name, gpu_memory_mb=mem))

    return processes


# ── 远程查询 ────────────────────────────────────────


async def _query_one_server(
    server: ServerConfig,
) -> tuple[str, list[GPUInfo]]:
    """查询单台服务器的 GPU 状态."""
    gpus: list[GPUInfo] = []
    key_path = _resolve_key(server.key_file) or None

    try:
        async with asyncssh.connect(
            server.host,
            port=server.port,
            username=server.user or None,
            client_keys=key_path,
            known_hosts=None,
        ) as conn:
            # 并行执行 GPU 和进程查询
            gpu_result = await conn.run(GPU_QUERY_CMD, check=False)
            proc_result = await conn.run(PROCESS_QUERY_CMD, check=False)

            if gpu_result.exit_status != 0:
                raise RuntimeError(
                    f"nvidia-smi 失败: {gpu_result.stderr.strip()}"
                )

            # 解析 GPU 列表
            gpus = parse_gpu_csv(gpu_result.stdout, server.host)

            # 构建 UUID → index 映射
            uuid_to_index: dict[str, int] = {}
            for g in gpus:
                if g.uuid:
                    uuid_to_index[g.uuid] = g.index

            # 解析进程列表并分配到正确的 GPU
            if proc_result.exit_status == 0:
                processes = parse_compute_apps_csv(proc_result.stdout, uuid_to_index)
                for g in gpus:
                    g.processes = processes.get(g.index, [])

    except (OSError, asyncssh.Error) as e:
        gpus.append(
            GPUInfo(
                index=-1,
                name=f"ERROR: {e}",
                server=server.host,
            )
        )

    return server.host, gpus


async def query_all_gpus(config: Config) -> list[GPUInfo]:
    """并行查询所有服务器的 GPU 状态."""
    if not config.servers:
        return []

    tasks = [_query_one_server(s) for s in config.servers]
    results = await asyncio.gather(*tasks)

    all_gpus: list[GPUInfo] = []
    for host, gpus in results:
        all_gpus.extend(gpus)

    all_gpus.sort(key=lambda g: (g.server, g.index if g.index >= 0 else 999))
    return all_gpus


def query_all_gpus_sync(config: Config) -> list[GPUInfo]:
    """同步封装."""
    return asyncio.run(query_all_gpus(config))


# ── 解析辅助 ───────────────────────────────────────

def _parse_int(val: str) -> int:
    try:
        s = val.strip()
        if s in ("", "[Not Supported]", "N/A"):
            return 0
        return int(float(s))
    except (ValueError, TypeError):
        return 0


def _parse_float(val: str) -> float:
    try:
        s = val.strip()
        if s in ("", "[Not Supported]", "N/A"):
            return 0.0
        return float(s)
    except (ValueError, TypeError):
        return 0.0
