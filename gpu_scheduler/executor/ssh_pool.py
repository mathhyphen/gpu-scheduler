"""SSH 连接池 — 按需连接，用完即断。支持 async with 自动管理生命周期."""

from __future__ import annotations

import asyncio
from pathlib import Path

import asyncssh

from gpu_scheduler.config import ServerConfig


class SSHPool:
    """按服务器缓存 SSH 连接。

    两种用法：
    1. async with — 自动清理（推荐，用于单次命令）
       async with get_pool() as pool:
           conn = await pool.get(server)

    2. 手动 — daemon 长期复用
       pool = get_pool()
       conn = await pool.get(server)
       await pool.close_all()
    """

    def __init__(self):
        self._connections: dict[str, asyncssh.SSHClientConnection] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close_all()

    def _key(self, server: ServerConfig) -> str:
        return f"{server.user}@{server.host}:{server.port}"

    async def get(self, server: ServerConfig) -> asyncssh.SSHClientConnection:
        """获取或创建到指定服务器的 SSH 连接."""
        if self._closed:
            raise RuntimeError("连接池已关闭")

        key = self._key(server)
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()

        async with self._locks[key]:
            conn = self._connections.get(key)
            if conn is not None:
                try:
                    await conn.run("echo ok", check=False)
                    return conn
                except (OSError, asyncssh.Error):
                    await self._close_conn(key)

            key_path = _resolve_key(server.key_file)
            conn = await asyncssh.connect(
                server.host,
                port=server.port,
                username=server.user or None,
                client_keys=key_path,
                known_hosts=None,
            )
            self._connections[key] = conn
            return conn

    async def _close_conn(self, key: str) -> None:
        conn = self._connections.pop(key, None)
        if conn:
            try:
                conn.close()
                await conn.wait_closed()
            except Exception:
                pass

    async def close_all(self) -> None:
        """关闭所有连接."""
        self._closed = True
        keys = list(self._connections.keys())
        for key in keys:
            await self._close_conn(key)
        self._locks.clear()

    @property
    def active_connections(self) -> int:
        return len(self._connections)


def _resolve_key(key_file: str) -> str | None:
    if not key_file:
        return None
    return str(Path(key_file).expanduser())


# ── 全局单例 ──

_pool: SSHPool | None = None


def get_pool() -> SSHPool:
    global _pool
    if _pool is None or _pool._closed:
        _pool = SSHPool()
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close_all()
        _pool = None
