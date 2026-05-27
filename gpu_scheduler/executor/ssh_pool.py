"""SSH 连接池 — 复用长连接，避免频繁握手被误判为爆破."""

from __future__ import annotations

import asyncio
from pathlib import Path

import asyncssh

from gpu_scheduler.config import ServerConfig


class SSHPool:
    """按服务器缓存 SSH 连接，复用而非每次新建.

    Usage:
        pool = SSHPool()
        conn = await pool.get(server)
        result = await conn.run("nvidia-smi")
    """

    def __init__(self):
        self._connections: dict[str, asyncssh.SSHClientConnection] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._last_used: dict[str, float] = {}

    def _key(self, server: ServerConfig) -> str:
        return f"{server.user}@{server.host}:{server.port}"

    async def get(self, server: ServerConfig) -> asyncssh.SSHClientConnection:
        """获取或创建到指定服务器的 SSH 连接."""
        key = self._key(server)

        # 防止并发时重复创建
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()

        async with self._locks[key]:
            conn = self._connections.get(key)

            # 检查现有连接是否存活
            if conn is not None:
                try:
                    # 发送一个轻量 keep-alive 确认连接存活
                    await conn.run("echo ok", check=False)
                    self._last_used[key] = asyncio.get_event_loop().time()
                    return conn
                except (OSError, asyncssh.Error):
                    # 连接已断，清理后重建
                    await self._close_conn(key)

            # 新建连接
            key_path = _resolve_key(server.key_file)
            conn = await asyncssh.connect(
                server.host,
                port=server.port,
                username=server.user or None,
                client_keys=key_path,
                known_hosts=None,
                keepalive_interval=60,  # 60s 心跳保活
                keepalive_count_max=3,
            )
            self._connections[key] = conn
            self._last_used[key] = asyncio.get_event_loop().time()
            return conn

    async def _close_conn(self, key: str) -> None:
        """关闭并移除指定连接."""
        conn = self._connections.pop(key, None)
        if conn:
            try:
                conn.close()
                await conn.wait_closed()
            except Exception:
                pass

    async def close_all(self) -> None:
        """关闭所有连接."""
        keys = list(self._connections.keys())
        for key in keys:
            await self._close_conn(key)
        self._locks.clear()
        self._last_used.clear()

    @property
    def active_connections(self) -> int:
        return len(self._connections)


def _resolve_key(key_file: str) -> str | None:
    """解析 SSH 密钥路径."""
    if not key_file:
        return None
    return str(Path(key_file).expanduser())


# 全局单例
_global_pool: SSHPool | None = None


def get_pool() -> SSHPool:
    """获取全局 SSH 连接池单例."""
    global _global_pool
    if _global_pool is None:
        _global_pool = SSHPool()
    return _global_pool


async def close_pool() -> None:
    """关闭全局连接池."""
    global _global_pool
    if _global_pool is not None:
        await _global_pool.close_all()
        _global_pool = None
