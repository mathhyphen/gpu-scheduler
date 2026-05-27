"""SQLite 持久化任务队列."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from gpu_scheduler.config import Config
from gpu_scheduler.executor import run_remote
from gpu_scheduler.gpu import GPUInfo
from gpu_scheduler.gpu.query import query_all_gpus
from gpu_scheduler.scheduler import Task, TaskStatus


def _get_db_path(config: Config) -> str:
    if config.scheduler.db_path:
        return config.scheduler.db_path
    p = Path.home() / ".local" / "share" / "gpu-scheduler" / "queue.db"
    return str(p)


def _ensure_dir(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def _connect(config: Config) -> sqlite3.Connection:
    db_path = _get_db_path(config)
    _ensure_dir(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db(config: Config) -> None:
    """初始化数据库表."""
    conn = _connect(config)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            command TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            priority INTEGER NOT NULL DEFAULT 0,
            gpu_count INTEGER NOT NULL DEFAULT 1,
            gpu_ids TEXT DEFAULT '',
            server TEXT DEFAULT '',
            created_at TEXT NOT NULL DEFAULT '',
            started_at TEXT DEFAULT '',
            finished_at TEXT DEFAULT '',
            exit_code INTEGER,
            output TEXT DEFAULT '',
            gpu_memory_min INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def add_task(config: Config, task: Task) -> int:
    """添加任务到队列，返回任务 ID."""
    conn = _connect(config)
    task.created_at = datetime.now(timezone.utc).isoformat()
    task.status = TaskStatus.PENDING

    row = task.to_row()
    del row["gpu_ids"]  # 初始为空
    del row["server"]
    del row["started_at"]
    del row["finished_at"]
    del row["exit_code"]
    del row["output"]
    # Ensure created_at is set
    row["created_at"] = task.created_at

    cols = ", ".join(row.keys())
    placeholders = ", ".join("?" for _ in row)
    conn.execute(f"INSERT INTO tasks ({cols}) VALUES ({placeholders})", list(row.values()))
    conn.commit()
    task_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return task_id


def get_task(config: Config, task_id: int) -> Task | None:
    """获取单个任务."""
    conn = _connect(config)
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    return Task.from_row(dict(row))


def list_tasks(
    config: Config,
    status: TaskStatus | None = None,
    limit: int = 20,
) -> list[Task]:
    """列出任务."""
    conn = _connect(config)
    if status:
        rows = conn.execute(
            "SELECT * FROM tasks WHERE status = ? ORDER BY priority ASC, created_at ASC LIMIT ?",
            (status.value, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()
    return [Task.from_row(dict(r)) for r in rows]


def get_next_pending(config: Config) -> Task | None:
    """获取队列中优先级最高的待调度任务."""
    conn = _connect(config)
    row = conn.execute(
        "SELECT * FROM tasks WHERE status = 'pending' ORDER BY priority ASC, created_at ASC LIMIT 1"
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return Task.from_row(dict(row))


def update_task(config: Config, task: Task) -> None:
    """更新任务状态."""
    conn = _connect(config)
    conn.execute(
        """UPDATE tasks SET
            status = ?, gpu_ids = ?, server = ?,
            started_at = ?, finished_at = ?,
            exit_code = ?, output = ?
        WHERE id = ?""",
        (
            task.status.value,
            task.gpu_ids,
            task.server,
            task.started_at,
            task.finished_at,
            task.exit_code,
            task.output,
            task.id,
        ),
    )
    conn.commit()
    conn.close()


def cancel_task(config: Config, task_id: int) -> bool:
    """取消一个待调度任务."""
    conn = _connect(config)
    cur = conn.execute(
        "UPDATE tasks SET status = 'cancelled', finished_at = ? WHERE id = ? AND status = 'pending'",
        (datetime.now(timezone.utc).isoformat(), task_id),
    )
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated


def _find_free_gpus(
    gpus: list[GPUInfo],
    gpu_count: int,
    gpu_memory_min: int,
) -> tuple[str | None, list[int]]:
    """从 GPU 列表中找出一台服务器上有足够空闲 GPU 的.

    Returns:
        (server_host, [gpu_indices]) 或 (None, []) 如果找不到.
    """
    # 按服务器分组
    from collections import defaultdict
    by_server: dict[str, list[GPUInfo]] = defaultdict(list)
    for g in gpus:
        if g.index >= 0:  # 排除错误条目
            by_server[g.server].append(g)

    for host, host_gpus in by_server.items():
        free = [
            g for g in host_gpus
            if g.is_free and g.memory_free >= gpu_memory_min
        ]
        if len(free) >= gpu_count:
            free.sort(key=lambda g: g.memory_free, reverse=True)
            selected = [g.index for g in free[:gpu_count]]
            return host, selected

    return None, []


async def _run_task(config: Config, task: Task) -> None:
    """执行单个任务（内部方法，由调度循环调用）."""
    # 查询所有 GPU
    gpus = await query_all_gpus(config)

    # 找空闲 GPU
    server_host, gpu_ids = _find_free_gpus(
        gpus, task.gpu_count, task.gpu_memory_min
    )

    if server_host is None:
        # 没有足够的空闲 GPU，保持 pending
        return

    # 找到对应服务器配置
    server_config = None
    for s in config.servers:
        if s.host == server_host:
            server_config = s
            break

    if server_config is None:
        task.status = TaskStatus.FAILED
        task.output = f"找不到服务器配置: {server_host}"
        task.finished_at = datetime.now(timezone.utc).isoformat()
        update_task(config, task)
        return

    # 标记为 RUNNING
    task.status = TaskStatus.RUNNING
    task.gpu_ids = ",".join(str(g) for g in gpu_ids)
    task.server = server_host
    task.started_at = datetime.now(timezone.utc).isoformat()
    update_task(config, task)

    # 执行
    exit_code, output = await run_remote(
        server_config, task.command, gpu_ids
    )

    # 更新结果
    task.status = TaskStatus.COMPLETED if exit_code == 0 else TaskStatus.FAILED
    task.exit_code = exit_code
    task.output = output
    task.finished_at = datetime.now(timezone.utc).isoformat()
    update_task(config, task)


async def scheduler_loop(config: Config, once: bool = False) -> None:
    """调度主循环 — 作为 daemon 运行.

    Args:
        config: 全局配置
        once: True = 只执行一轮就退出
    """
    from gpu_scheduler.utils import console

    init_db(config)

    console.print("[bold green]GPU Scheduler daemon 启动[/bold green]")
    console.print(f"  轮询间隔: {config.scheduler.poll_interval}s")
    console.print(f"  服务器数: {len(config.servers)}")

    while True:
        task = get_next_pending(config)
        if task is not None:
            console.print(f"[cyan]调度任务 #{task.id}: {task.command[:60]}...[/cyan]")
            await _run_task(config, task)
            if task.status == TaskStatus.COMPLETED:
                console.print(f"  [green][OK] 完成 (exit={task.exit_code})[/green]")
            elif task.status == TaskStatus.FAILED:
                console.print(f"  [red][FAIL] 失败 (exit={task.exit_code})[/red]")

        if once:
            break

        await asyncio.sleep(config.scheduler.poll_interval)
