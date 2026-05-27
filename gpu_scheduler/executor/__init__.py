"""SSH 执行层 — 通过 SSH 连接池在远程 GPU 服务器上执行用户命令."""

from __future__ import annotations

from gpu_scheduler.config import ServerConfig
from gpu_scheduler.executor.ssh_pool import get_pool


async def run_remote(
    server: ServerConfig,
    command: str,
    gpu_ids: list[int] | None = None,
) -> tuple[int, str]:
    """在远程服务器的指定 GPU 上执行命令 (复用连接池).

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
    """测试 SSH 连接是否正常 (复用连接池)."""
    try:
        pool = get_pool()
        conn = await pool.get(server)
        result = await conn.run("nvidia-smi -L", check=False)
        if result.exit_status == 0:
            return True, result.stdout.strip()
        return False, result.stderr.strip()
    except Exception as e:
        return False, str(e)
