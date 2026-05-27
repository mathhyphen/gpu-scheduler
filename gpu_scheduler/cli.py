"""CLI 入口 — Typer 应用."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer

from gpu_scheduler.config import Config, ServerConfig, SchedulerConfig, generate_example_config, load_config
from gpu_scheduler.gpu.query import query_all_gpus_sync
from gpu_scheduler.scheduler import Task, TaskStatus
from gpu_scheduler.scheduler.queue import (
    add_task,
    cancel_task,
    get_task,
    init_db,
    list_tasks,
    scheduler_loop,
)
from gpu_scheduler.utils import console, print_gpu_table, render_gpu_table, render_queue_table

app = typer.Typer(
    name="gpu-scheduler",
    help="轻量级多服务器 GPU 调度器",
    no_args_is_help=True,
)

# ── list ──────────────────────────────────────────────


@app.command("list")
def list_gpus(
    watch: bool = typer.Option(False, "--watch", "-w", help="持续刷新"),
    interval: float = typer.Option(3.0, "--interval", "-i", help="刷新间隔（秒）"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="配置文件路径"),
):
    """查看所有服务器的 GPU 状态."""
    config = load_config(config_path)
    if not config.servers:
        console.print("[yellow]未配置服务器。运行 [bold]gpu-scheduler config init[/bold] 生成示例配置[/yellow]")
        return

    if watch:
        import time
        from rich.live import Live

        try:
            with Live(refresh_per_second=1 / interval) as live:
                while True:
                    gpus = query_all_gpus_sync(config)
                    table = render_gpu_table(gpus)
                    live.update(table)
                    time.sleep(interval)
        except KeyboardInterrupt:
            pass
    else:
        gpus = query_all_gpus_sync(config)
        print_gpu_table(gpus)


# ── run ───────────────────────────────────────────────


@app.command("run")
def run_command(
    command: list[str] = typer.Argument(..., help="要执行的命令"),
    gpu_count: int = typer.Option(1, "--gpus", "-g", help="需要的 GPU 数量"),
    priority: int = typer.Option(0, "--priority", "-p", help="优先级（越小越高）"),
    gpu_memory: int = typer.Option(0, "--gpu-memory", "-m", help="最低显存要求 (MB)"),
    wait: bool = typer.Option(False, "--wait", "-w", help="等待任务完成"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="配置文件路径"),
):
    """提交任务到队列并等待执行（等同于 submit + wait）."""
    config = load_config(config_path)
    init_db(config)

    task = Task(
        command=" ".join(command),
        priority=priority,
        gpu_count=gpu_count,
        gpu_memory_min=gpu_memory,
    )
    task_id = add_task(config, task)
    console.print(f"[green][OK] 任务已提交[/green] ID={task_id}")

    if wait:
        console.print(f"[dim]等待任务 #{task_id} 完成...[/dim]")
        while True:
            import time
            time.sleep(2)
            t = get_task(config, task_id)
            if t is None:
                console.print("[red]任务丢失[/red]")
                return
            if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                if t.status == TaskStatus.COMPLETED:
                    console.print(f"[green][OK] 任务 #{task_id} 完成 (exit={t.exit_code})[/green]")
                else:
                    console.print(f"[red][FAIL] 任务 #{task_id} {t.status.value} (exit={t.exit_code})[/red]")
                if t.output:
                    console.print(f"[dim]--- 输出 ---[/dim]\n{t.output}")
                return


# ── submit ────────────────────────────────────────────


@app.command("submit")
def submit(
    command: list[str] = typer.Argument(..., help="要执行的命令"),
    gpu_count: int = typer.Option(1, "--gpus", "-g", help="需要的 GPU 数量"),
    priority: int = typer.Option(0, "--priority", "-p", help="优先级（越小越高）"),
    gpu_memory: int = typer.Option(0, "--gpu-memory", "-m", help="最低显存要求 (MB)"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="配置文件路径"),
):
    """提交任务到队列（非阻塞，需要 daemon 来消费）."""
    config = load_config(config_path)
    init_db(config)

    task = Task(
        command=" ".join(command),
        priority=priority,
        gpu_count=gpu_count,
        gpu_memory_min=gpu_memory,
    )
    task_id = add_task(config, task)
    console.print(f"[green][OK] 任务 #{task_id} 已加入队列[/green]")


# ── queue ─────────────────────────────────────────────


@app.command("queue")
def queue(
    status: Optional[str] = typer.Option(None, "--status", "-s", help="筛选状态 (pending/running/completed/failed/cancelled)"),
    limit: int = typer.Option(20, "--limit", "-n", help="显示条数"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="配置文件路径"),
):
    """查看任务队列."""
    config = load_config(config_path)
    init_db(config)

    status_filter = TaskStatus(status) if status else None
    tasks = list_tasks(config, status=status_filter, limit=limit)

    if not tasks:
        console.print("[dim]队列为空[/dim]")
    else:
        table = render_queue_table(tasks)
        console.print(table)


# ── cancel ────────────────────────────────────────────


@app.command("cancel")
def cancel(
    task_id: int = typer.Argument(..., help="要取消的任务 ID"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="配置文件路径"),
):
    """取消一个等待中的任务."""
    config = load_config(config_path)
    if cancel_task(config, task_id):
        console.print(f"[green][OK] 任务 #{task_id} 已取消[/green]")
    else:
        console.print(f"[yellow]任务 #{task_id} 不存在或已在执行中[/yellow]")


# ── daemon ────────────────────────────────────────────


@app.command("daemon")
def daemon(
    once: bool = typer.Option(False, "--once", help="只执行一轮调度"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="配置文件路径"),
):
    """启动调度 daemon（前台运行，Ctrl+C 停止）."""
    config = load_config(config_path)
    if not config.servers:
        console.print("[red]未配置服务器，请先编辑配置文件[/red]")
        console.print(f"  配置文件路径: {config_path or '~/.config/gpu-scheduler/config.toml'}")
        return

    try:
        asyncio.run(scheduler_loop(config, once=once))
    except KeyboardInterrupt:
        console.print("\n[dim]daemon 已停止[/dim]")


# ── config ────────────────────────────────────────────

config_app = typer.Typer(help="配置管理")
app.add_typer(config_app, name="config")


@config_app.command("init")
def config_init():
    """生成示例配置文件."""
    config_dir = Path.home() / ".config" / "gpu-scheduler"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.toml"

    if config_file.exists():
        console.print(f"[yellow]配置文件已存在: {config_file}[/yellow]")
        console.print("如需覆盖请先删除")
        return

    config_file.write_text(generate_example_config(), encoding="utf-8")
    console.print(f"[green][OK] 示例配置已生成: {config_file}[/green]")
    console.print("[dim]请编辑此文件，填入你的服务器信息[/dim]")


@config_app.command("show")
def config_show(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="配置文件路径"),
):
    """显示当前配置."""
    config = load_config(config_path)
    console.print(f"[bold]服务器列表 ({len(config.servers)} 台):[/bold]")
    for s in config.servers:
        console.print(f"  - {s.user}@{s.host}:{s.port}")
        if s.labels:
            console.print(f"    标签: {s.labels}")
    console.print(f"\n[bold]调度器配置:[/bold]")
    console.print(f"  轮询间隔: {config.scheduler.poll_interval}s")


@config_app.command("test")
def config_test(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="配置文件路径"),
):
    """测试所有服务器的 SSH 连接."""
    from gpu_scheduler.executor import check_ssh

    config = load_config(config_path)
    if not config.servers:
        console.print("[red]未配置服务器[/red]")
        return

    async def _test_all():
        results = []
        for s in config.servers:
            console.print(f"[dim]测试 {s.host}...[/dim]")
            ok, msg = await check_ssh(s)
            results.append((s, ok, msg))
        return results

    results = asyncio.run(_test_all())
    for server, ok, msg in results:
        if ok:
            console.print(f"[green][OK] {server.host}[/green] - {msg.split(chr(10))[0][:80]}")
        else:
            console.print(f"[red][FAIL] {server.host}[/red] - {msg[:120]}")


# ── main ──────────────────────────────────────────────

def main():
    app()


if __name__ == "__main__":
    main()
