"""Rich 终端渲染工具."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.text import Text

from gpu_scheduler.gpu import GPUInfo

console = Console()


def render_gpu_table(gpus: list[GPUInfo]) -> Table:
    """渲染 GPU 状态表格."""
    table = Table(title="GPU 集群状态", title_style="bold cyan")

    table.add_column("Server", style="bold")
    table.add_column("GPU", justify="center")
    table.add_column("Name", style="dim")
    table.add_column("显存", justify="right")
    table.add_column("利用率", justify="right")
    table.add_column("温度", justify="right")
    table.add_column("进程", justify="right")

    for g in gpus:
        if g.index < 0:
            # 错误状态
            table.add_row(
                g.server,
                "—",
                Text(g.name, style="red"),
                "—",
                "—",
                "—",
                "—",
            )
            continue

        # 显存格式
        mem_str = f"{g.memory_used}/{g.memory_total} MB"
        mem_style = "green" if g.is_free else "yellow" if g.memory_usage_pct < 80 else "red"

        # 利用率
        util_str = f"{g.gpu_util}%"
        util_style = "green" if g.gpu_util < 20 else "yellow" if g.gpu_util < 80 else "red"

        # 温度
        temp_str = f"{g.temperature}°C"
        temp_style = "green" if g.temperature < 70 else "yellow" if g.temperature < 85 else "red"

        # 进程数
        proc_count = len(g.processes)
        proc_str = str(proc_count) if proc_count > 0 else "—"
        proc_style = "red" if proc_count > 0 else "green"

        table.add_row(
            g.server,
            str(g.index),
            g.name,
            Text(mem_str, style=mem_style),
            Text(util_str, style=util_style),
            Text(temp_str, style=temp_style),
            Text(proc_str, style=proc_style),
        )

    return table


def print_gpu_table(gpus: list[GPUInfo]) -> None:
    """打印 GPU 状态表格."""
    if not gpus:
        console.print("[yellow]未配置任何服务器，请先运行 [bold]gpu-scheduler config init[/bold][/yellow]")
        return
    table = render_gpu_table(gpus)
    console.print(table)


def render_queue_table(tasks: list) -> Table:
    """渲染任务队列表格."""
    from gpu_scheduler.scheduler import TaskStatus

    table = Table(title="任务队列", title_style="bold cyan")

    table.add_column("ID", justify="right", style="dim")
    table.add_column("状态")
    table.add_column("优先级", justify="center")
    table.add_column("GPU", justify="center")
    table.add_column("服务器")
    table.add_column("命令", style="dim")
    table.add_column("创建时间")

    status_colors = {
        TaskStatus.PENDING: "yellow",
        TaskStatus.RUNNING: "cyan",
        TaskStatus.COMPLETED: "green",
        TaskStatus.FAILED: "red",
        TaskStatus.CANCELLED: "dim",
    }

    for t in tasks:
        status = t.status if hasattr(t, 'status') else TaskStatus.PENDING
        style = status_colors.get(status, "white")
        table.add_row(
            str(t.id),
            Text(status.value, style=style),
            str(t.priority),
            str(t.gpu_count),
            t.server or "—",
            t.command[:60] + ("..." if len(t.command) > 60 else ""),
            t.created_at[:19] if t.created_at else "—",
        )

    return table
