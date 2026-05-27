"""GPU Scheduler 单元测试 — 不需要真实 GPU 服务器."""

import sys
import tempfile
from pathlib import Path

# Ensure the workspace is on the path
sys.path.insert(0, str(Path(__file__).parent))

# ── 1. CSV 解析测试 ────────────────────────────────────


def test_parse_gpu_csv():
    """模拟 nvidia-smi CSV 输出，验证解析逻辑."""
    from gpu_scheduler.gpu.query import parse_gpu_csv

    csv_text = (
        "0, GPU-abc123, Tesla V100-SXM2-32GB, 65, 87, 45, 24576, 32768, 8192, 250.00, 60\n"
        "1, GPU-def456, Tesla V100-SXM2-32GB, 52, 0, 0, 0, 32768, 32768, 30.00, 40\n"
        "2, GPU-ghi789, Tesla V100-SXM2-32GB, 70, 95, 80, 30000, 32768, 2768, 280.00, 80\n"
    )
    gpus = parse_gpu_csv(csv_text, "server-1")

    assert len(gpus) == 3

    # GPU 0: 高负载
    assert gpus[0].index == 0
    assert gpus[0].name == "Tesla V100-SXM2-32GB"
    assert gpus[0].server == "server-1"
    assert gpus[0].temperature == 65
    assert gpus[0].gpu_util == 87
    assert gpus[0].memory_used == 24576
    assert gpus[0].memory_total == 32768
    assert gpus[0].memory_free == 8192
    assert gpus[0].power_draw == 250.0

    # GPU 1: 空闲
    assert gpus[1].index == 1
    assert gpus[1].gpu_util == 0
    assert gpus[1].memory_used == 0

    # GPU 2: 接近满载
    assert gpus[2].memory_used == 30000

    print("[OK] test_parse_gpu_csv passed")


def test_parse_compute_apps_csv():
    """模拟进程查询 CSV 输出."""
    from gpu_scheduler.gpu.query import parse_compute_apps_csv

    csv_text = "12345, python, 21504, GPU-abc123\n67890, python, 3072, GPU-abc123\n"
    procs = parse_compute_apps_csv(csv_text)

    assert 0 in procs  # 归到 GPU 0（当前实现）
    assert len(procs[0]) == 2
    assert procs[0][0].pid == 12345
    assert procs[0][0].name == "python"
    assert procs[0][0].gpu_memory_mb == 21504
    assert procs[0][1].pid == 67890
    assert procs[0][1].gpu_memory_mb == 3072

    print("[OK] test_parse_compute_apps_csv passed")


def test_parse_compute_apps_with_uuid_map():
    """测试通过 UUID 映射将进程分配到正确的 GPU."""
    from gpu_scheduler.gpu.query import parse_compute_apps_csv

    csv_text = (
        "100, proc_a, 5000, GPU-aaa\n"
        "200, proc_b, 3000, GPU-bbb\n"
    )
    uuid_map = {"GPU-aaa": 0, "GPU-bbb": 2}
    procs = parse_compute_apps_csv(csv_text, uuid_to_index=uuid_map)

    assert 0 in procs
    assert 2 in procs
    assert len(procs[0]) == 1
    assert procs[0][0].pid == 100
    assert len(procs[2]) == 1
    assert procs[2][0].pid == 200

    print("[OK] test_parse_compute_apps_csv passed")


def test_parse_compute_apps_with_uuid_map():
    """测试通过 UUID 映射将进程分配到正确的 GPU."""
    from gpu_scheduler.gpu.query import parse_compute_apps_csv

    csv_text = (
        "100, proc_a, 5000, GPU-aaa\n"
        "200, proc_b, 3000, GPU-bbb\n"
    )
    uuid_map = {"GPU-aaa": 0, "GPU-bbb": 2}
    procs = parse_compute_apps_csv(csv_text, uuid_to_index=uuid_map)

    assert 0 in procs
    assert 2 in procs
    assert len(procs[0]) == 1
    assert procs[0][0].pid == 100
    assert len(procs[2]) == 1
    assert procs[2][0].pid == 200

    print("[OK] test_parse_compute_apps_with_uuid_map passed")


def test_parse_gpu_csv_with_n_a():
    """测试 nvidia-smi 中 [Not Supported] 和 N/A 值的处理."""
    from gpu_scheduler.gpu.query import parse_gpu_csv

    csv_text = "0, GPU-xyz, GeForce RTX 3090, N/A, [Not Supported], N/A, 8192, 24576, 16384, N/A, [Not Supported]\n"
    gpus = parse_gpu_csv(csv_text, "desktop")

    assert len(gpus) == 1
    assert gpus[0].temperature == 0
    assert gpus[0].gpu_util == 0
    assert gpus[0].power_draw == 0.0
    assert gpus[0].fan_speed == 0
    assert gpus[0].memory_used == 8192
    assert gpus[0].memory_total == 24576

    print("[OK] test_parse_gpu_csv_with_n_a passed")


# ── 2. GPUInfo 模型测试 ─────────────────────────────────


def test_gpu_is_free():
    """测试空闲判断逻辑."""
    from gpu_scheduler.gpu import GPUInfo

    # 显存 < 500MB 且无进程 → 空闲
    gpu = GPUInfo(index=0, name="Test", server="s1", memory_used=100, memory_total=32768)
    assert gpu.is_free

    # 显存 >= 500MB → 不空闲
    gpu.memory_used = 600
    assert not gpu.is_free

    # 有进程 → 不空闲
    from gpu_scheduler.gpu import ProcessInfo
    gpu.memory_used = 100
    gpu.processes = [ProcessInfo(pid=1, name="test", gpu_memory_mb=50)]
    assert not gpu.is_free

    print("[OK] test_gpu_is_free passed")


def test_memory_usage_pct():
    """测试显存百分比计算."""
    from gpu_scheduler.gpu import GPUInfo

    gpu = GPUInfo(index=0, name="Test", server="s1", memory_used=16384, memory_total=32768)
    assert gpu.memory_usage_pct == 50.0

    gpu.memory_total = 0
    assert gpu.memory_usage_pct == 0.0  # 除零保护

    print("[OK] test_memory_usage_pct passed")


# ── 3. 任务模型测试 ────────────────────────────────────


def test_task_to_row_from_row():
    """测试 Task 的序列化往返."""
    from gpu_scheduler.scheduler import Task, TaskStatus

    task = Task(
        id=1,
        command="python train.py",
        status=TaskStatus.RUNNING,
        priority=5,
        gpu_count=2,
        gpu_ids="0,1",
        server="gpu-1",
        created_at="2025-01-01T00:00:00",
        exit_code=0,
        gpu_memory_min=16000,
    )
    row = task.to_row()
    restored = Task.from_row({"id": 1, **row})

    assert restored.command == "python train.py"
    assert restored.status == TaskStatus.RUNNING
    assert restored.priority == 5
    assert restored.gpu_count == 2
    assert restored.gpu_ids == "0,1"
    assert restored.server == "gpu-1"
    assert restored.gpu_memory_min == 16000
    assert restored.exit_code == 0

    print("[OK] test_task_to_row_from_row passed")


# ── 4. GPU 分配逻辑测试 ────────────────────────────────


def test_find_free_gpus_single():
    """测试在单台服务器上找空闲 GPU."""
    from gpu_scheduler.gpu import GPUInfo
    from gpu_scheduler.scheduler.queue import _find_free_gpus

    gpus = [
        GPUInfo(index=0, name="A100", server="node1", memory_used=30000, memory_total=40960, memory_free=10960),
        GPUInfo(index=1, name="A100", server="node1", memory_used=0, memory_total=40960, memory_free=40960),
        GPUInfo(index=2, name="A100", server="node1", memory_used=0, memory_total=40960, memory_free=40960),
        GPUInfo(index=3, name="A100", server="node1", memory_used=200, memory_total=40960, memory_free=40760),
    ]

    # 需要 2 张空闲 GPU
    host, ids = _find_free_gpus(gpus, gpu_count=2, gpu_memory_min=0)
    assert host == "node1"
    assert len(ids) == 2

    # GPU 0 被占用，GPU 1-3 空闲 → 选显存最大的 2 张 (1 和 2，都是 40960 free)
    assert 0 not in ids
    assert 1 in ids
    assert 2 in ids

    print("[OK] test_find_free_gpus_single passed")


def test_find_free_gpus_not_enough():
    """测试没有足够空闲 GPU 的情况."""
    from gpu_scheduler.gpu import GPUInfo
    from gpu_scheduler.scheduler.queue import _find_free_gpus

    gpus = [
        GPUInfo(index=0, name="A100", server="node1", memory_used=30000, memory_total=40960, memory_free=10960),
        GPUInfo(index=1, name="A100", server="node1", memory_used=25000, memory_total=40960, memory_free=15960),
    ]

    # 需要 1 张空闲 GPU → 都不空闲（显存都 > 500MB）
    host, ids = _find_free_gpus(gpus, gpu_count=1, gpu_memory_min=0)
    assert host is None
    assert ids == []

    print("[OK] test_find_free_gpus_not_enough passed")


def test_find_free_gpus_memory_requirement():
    """测试显存要求过滤."""
    from gpu_scheduler.gpu import GPUInfo
    from gpu_scheduler.scheduler.queue import _find_free_gpus

    gpus = [
        GPUInfo(index=0, name="A100", server="node1", memory_used=0, memory_total=40960, memory_free=40960),
        GPUInfo(index=1, name="V100", server="node1", memory_used=0, memory_total=16384, memory_free=16384),
    ]

    # 需要 1 张 GPU，最低显存 24000MB → GPU 1 不够 (16384 < 24000)
    host, ids = _find_free_gpus(gpus, gpu_count=1, gpu_memory_min=24000)
    assert host == "node1"
    assert ids == [0]  # 只有 GPU 0 满足

    # 最低 48000MB → 都不够
    host, ids = _find_free_gpus(gpus, gpu_count=1, gpu_memory_min=48000)
    assert host is None

    print("[OK] test_find_free_gpus_memory_requirement passed")


def test_find_free_gpus_multi_server():
    """测试跨服务器选择：只从同一台服务器分配."""
    from gpu_scheduler.gpu import GPUInfo
    from gpu_scheduler.scheduler.queue import _find_free_gpus

    gpus = [
        GPUInfo(index=0, name="A100", server="node1", memory_used=0, memory_total=40960, memory_free=40960),
        GPUInfo(index=0, name="A100", server="node2", memory_used=0, memory_total=40960, memory_free=40960),
    ]

    # 需要 2 张 GPU → 不在同一服务器，应该失败
    host, ids = _find_free_gpus(gpus, gpu_count=2, gpu_memory_min=0)
    assert host is None

    print("[OK] test_find_free_gpus_multi_server passed")


# ── 5. SQLite 队列测试 ──────────────────────────────────


def test_sqlite_queue():
    """测试 SQLite 任务队列的增删改查."""
    import tempfile
    from gpu_scheduler.config import Config, SchedulerConfig
    from gpu_scheduler.scheduler import Task, TaskStatus
    from gpu_scheduler.scheduler.queue import (
        init_db, add_task, get_task, list_tasks,
        get_next_pending, update_task, cancel_task,
    )

    # 用临时目录避免污染真实数据库
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test_queue.db")
        config = Config(
            servers=[],
            scheduler=SchedulerConfig(db_path=db_path),
        )
        init_db(config)

        # 添加 3 个任务
        t1 = Task(command="echo hello", priority=10)
        t2 = Task(command="echo world", priority=0)   # 最高优先级
        t3 = Task(command="echo foo", priority=5)

        id1 = add_task(config, t1)
        id2 = add_task(config, t2)
        id3 = add_task(config, t3)

        assert id1 > 0 and id2 > 0 and id3 > 0

        # 获取单个任务
        got = get_task(config, id1)
        assert got is not None
        assert got.command == "echo hello"
        assert got.status == TaskStatus.PENDING

        # 获取最高优先级 pending 任务 → t2 (priority=0)
        next_task = get_next_pending(config)
        assert next_task is not None
        assert next_task.id == id2
        assert next_task.priority == 0

        # 列出所有任务
        all_tasks = list_tasks(config, limit=10)
        assert len(all_tasks) == 3

        # 按状态筛选
        pending = list_tasks(config, status=TaskStatus.PENDING)
        assert len(pending) == 3

        # 更新任务状态
        t2.id = id2  # add_task 不会自动设置 task.id
        t2.status = TaskStatus.RUNNING
        t2.gpu_ids = "0"
        t2.server = "node1"
        update_task(config, t2)

        updated = get_task(config, id2)
        assert updated.status == TaskStatus.RUNNING
        assert updated.gpu_ids == "0"

        # 下一个 pending 应该是 t3 (priority=5)
        next_task = get_next_pending(config)
        assert next_task.id == id3

        # 取消任务
        ok = cancel_task(config, id3)
        assert ok
        cancelled = get_task(config, id3)
        assert cancelled.status == TaskStatus.CANCELLED

        # 只剩 1 个 pending (t1)
        next_task = get_next_pending(config)
        assert next_task.id == id1

    print("[OK] test_sqlite_queue passed")


# ── 6. 配置加载测试 ─────────────────────────────────────


def test_load_empty_config():
    """测试无配置文件时返回空配置."""
    from gpu_scheduler.config import load_config

    config = load_config("/nonexistent/path/config.toml")
    assert config.servers == []
    assert config.scheduler.poll_interval == 5.0

    print("[OK] test_load_empty_config passed")


# ── main ────────────────────────────────────────────────

if __name__ == "__main__":
    test_parse_gpu_csv()
    test_parse_compute_apps_csv()
    test_parse_compute_apps_with_uuid_map()
    test_parse_gpu_csv_with_n_a()
    test_gpu_is_free()
    test_memory_usage_pct()
    test_task_to_row_from_row()
    test_find_free_gpus_single()
    test_find_free_gpus_not_enough()
    test_find_free_gpus_memory_requirement()
    test_find_free_gpus_multi_server()
    test_sqlite_queue()
    test_load_empty_config()
    print("\n=== 13/13 tests passed ===")