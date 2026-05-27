"""SSH 执行层 — 通过 SSH 在远程 GPU 服务器上执行用户命令."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import asyncssh

from gpu_scheduler.config import ServerConfig


def _resolve_key(key_file: str) -> str | None:
    if not key_file:
        return None
    return str(Path(key_file).expanduser())


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

    # 构建远程命令：设置环境变量后执行
    if env:
        env_str = " ".join(f'{k}="{v}"' for k, v in env.items())
        full_cmd = f"{env_str} {command}"
    else:
        full_cmd = command

    key_path = _resolve_key(server.key_file) or None

    try:
        async with asyncssh.connect(
            server.host,
            port=server.port,
            username=server.user or None,
            client_keys=key_path,
            known_hosts=None,
        ) as conn:
            result = await conn.run(full_cmd, check=False)
            output = result.stdout
            if result.stderr:
                output += "\n[stderr]\n" + result.stderr
            return result.exit_status, output[-5000:]  # 截断过长输出

    except (OSError, asyncssh.Error) as e:
        return -1, f"SSH 连接失败: {e}"


async def check_ssh(server: ServerConfig) -> tuple[bool, str]:
    """测试 SSH 连接是否正常."""
    try:
        key_path = _resolve_key(server.key_file) or None
        async with asyncssh.connect(
            server.host,
            port=server.port,
            username=server.user or None,
            client_keys=key_path,
            known_hosts=None,
        ) as conn:
            result = await conn.run("nvidia-smi -L", check=False)
            if result.exit_status == 0:
                return True, result.stdout.strip()
            return False, result.stderr.strip()
    except (OSError, asyncssh.Error) as e:
        return False, str(e)
