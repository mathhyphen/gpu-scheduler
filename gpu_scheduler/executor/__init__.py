"""SSH 执行层 — 通过 SSH 连接池在远程 GPU 服务器上执行用户命令."""

from __future__ import annotations

import asyncio

from gpu_scheduler.config import ServerConfig
from gpu_scheduler.executor.ssh_pool import get_pool, close_pool


async def run_remote(
    server: ServerConfig,
    command: str,
    gpu_ids: list[int] | None = None,
) -> tuple[int, str]:
    """在远程服务器的指定 GPU 上执行命令.

    Returns:
        (exit_code, combined_output)
    """
    env = {}
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    if env:
        env_str = " ".join(f'{k}="{v}"' for k, v in env.items())
        full_cmd = f"{env_str} {command}"
    else:
        full_cmd = command

    try:
        pool = get_pool()
        conn = await pool.get(server)
        result = await conn.run(full_cmd, check=False)
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        return result.exit_status, output[-5000:]
    except Exception as e:
        return -1, f"SSH 执行失败: {e}"


async def check_ssh(server: ServerConfig) -> tuple[bool, str]:
    """测试 SSH 连接是否正常."""
    try:
        pool = get_pool()
        conn = await pool.get(server)
        result = await conn.run("nvidia-smi -L", check=False)
        if result.exit_status == 0:
            return True, result.stdout.strip()
        return False, result.stderr.strip()
    except Exception as e:
        return False, str(e)


async def run_immediate(
    config,
    command: str,
    gpu_count: int = 1,
    gpu_memory_min: int = 0,
) -> tuple[int, str, str, list[int]]:
    """立即执行：查询 GPU → 找空闲 → SSH 执行 → 返回结果。

    不走队列，连接用完即断。

    Returns:
        (exit_code, output, server_host, gpu_ids)
    """
    from gpu_scheduler.config import Config
    from gpu_scheduler.utils import console
    from gpu_scheduler.gpu.query import query_all_gpus
    from gpu_scheduler.scheduler.queue import _find_free_gpus

    try:
        # 查询所有 GPU
        console.print("[dim]查询 GPU 状态...[/dim]")
        gpus = await query_all_gpus(config)

        # 找空闲 GPU
        server_host, gpu_ids = _find_free_gpus(gpus, gpu_count, gpu_memory_min)

        if server_host is None:
            return -1, f"找不到满足要求的空闲 GPU（需要 {gpu_count} 张，最低显存 {gpu_memory_min}MB）", "", []

        # 找服务器配置
        server_config = None
        for s in config.servers:
            if s.host == server_host:
                server_config = s
                break

        if server_config is None:
            return -1, f"找不到服务器配置: {server_host}", "", []

        console.print(
            f"[cyan]分配 GPU: {server_host} [{', '.join(str(g) for g in gpu_ids)}]"
            f"  | 执行: {command[:60]}...[/cyan]"
        )

        # 执行
        exit_code, output = await run_remote(server_config, command, gpu_ids)

        return exit_code, output, server_host, gpu_ids

    finally:
        await close_pool()
