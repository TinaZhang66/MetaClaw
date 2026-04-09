#!/usr/bin/env python3
"""
paper_static_skills_run.py — paper-aligned static-skills benchmark runner on the full 30-day dataset.

Compared with static_skills_run.py:
  1. Uses paper-static-skills.yaml
  2. Defaults to Qwen3-8B-style settings
  3. Uses the full metaclaw-bench dataset so results are comparable to paper_rl_run.py
"""

import json
import os
import pty
import re
import select
import shutil
import signal
import socket
import sys
import subprocess
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_ROOT = SCRIPT_DIR.parent
REPO_ROOT = BENCH_ROOT.parent


class cfg:
    LOG_FILE = str(BENCH_ROOT / "logs" / "paper_static_skills_full_run" / "bench_run.log")
    BENCH_BIN = shutil.which("metaclaw-bench")
    BENCH_INPUT = str(BENCH_ROOT / "data" / "metaclaw-bench" / "all_tests_metaclaw.json")
    BENCH_OUTPUT = str(BENCH_ROOT / "results" / "paper_static_skills_full")
    BENCH_COUNT = 3
    API_KEY_SCRIPT = None
    PROXY_SCRIPT = str(SCRIPT_DIR / "proxy_run.py")
    PROXY_CONFIG = str(SCRIPT_DIR / "config" / "paper-static-skills.yaml")
    ORIGINAL_SKILL_DIR = str(REPO_ROOT / "memory_data" / "skills")


def bench_cmd() -> list[str]:
    if cfg.BENCH_BIN:
        return [cfg.BENCH_BIN]
    return [sys.executable, "-m", "src.cli"]


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def resolve_log_path(log_file: str) -> Path:
    p = Path(log_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        return p
    stem, suffix = p.stem, p.suffix
    n = 1
    while True:
        candidate = p.parent / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def load_env_from_shell(script_path: str) -> dict:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        tmp_path = f.name
    try:
        result = subprocess.run(
            ["bash", "-c",
             f"source {script_path} && "
             "python3 -c 'import os,json; json.dump(dict(os.environ),open(os.environ[\"__TMP_ENV\"],\"w\"))'"],
            env={**os.environ, "__TMP_ENV": tmp_path},
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"加载 env 脚本失败：{script_path}")
        with open(tmp_path) as f:
            return json.load(f)
    finally:
        os.unlink(tmp_path)


def write_proxy_config(env: dict, temp_skill_dir: str, port: int) -> str:
    import yaml as _yaml

    with open(cfg.PROXY_CONFIG, encoding="utf-8") as f:
        content = f.read()

    def replace(m):
        var = m.group(1)
        val = env.get(var, "")
        if not val:
            print(f"[config] 警告：环境变量 {var} 未设置，替换为空字符串")
        return val

    content = re.sub(r"\$\{(\w+)\}", replace, content)

    data = _yaml.safe_load(content) or {}
    data.setdefault("skills", {})["dir"] = temp_skill_dir
    data.setdefault("proxy", {})["port"] = port

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="metaclaw_cfg_",
        delete=False, encoding="utf-8",
    )
    _yaml.dump(data, tmp, default_flow_style=False, allow_unicode=True)
    tmp.close()
    print(f"[config] 临时配置已写入 {tmp.name} (port={port})")
    return tmp.name


def start_proxy(env: dict, config_path: str, port: int) -> subprocess.Popen:
    print(f"[proxy] 正在启动 proxy (paper static skills, port={port})...")
    proxy_env = dict(env) if env else dict(os.environ)
    proxy_env["METACLAW_CONFIG_FILE"] = config_path
    # Benchmark runs start their own per-work-copy OpenClaw gateways, so
    # touching the user's global OpenClaw profile only creates cross-run
    # interference when multiple paper_* runners execute in parallel.
    proxy_env["METACLAW_SKIP_OPENCLAW_AUTOCONFIG"] = "1"
    proc = subprocess.Popen(
        [sys.executable, cfg.PROXY_SCRIPT],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=proxy_env,
        start_new_session=True,
    )

    url = f"http://localhost:{port}/healthz"
    print("[proxy] 等待就绪...")
    deadline = time.time() + 120
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"[proxy] 进程意外退出（退出码 {proc.returncode}）")
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    print("[proxy] 就绪，继续执行 benchmark...")
                    return proc
        except Exception:
            pass
        time.sleep(2)

    raise RuntimeError("[proxy] 等待超时（120s），未收到健康检查响应")


def stop_proxy(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    print("[proxy] 正在停止 proxy...")
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.wait()
    print("[proxy] proxy 已停止")


def run_command(cmd: list, log_path: Path, env: dict = None, cwd: str | None = None) -> int:
    master_fd, slave_fd = pty.openpty()
    proc = subprocess.Popen(
        cmd,
        stdout=slave_fd,
        stderr=slave_fd,
        env=env,
        cwd=cwd,
        close_fds=True,
    )
    os.close(slave_fd)

    with open(log_path, "a", encoding="utf-8") as log_f:
        while True:
            try:
                r, _, _ = select.select([master_fd], [], [], 1.0)
                if r:
                    chunk = os.read(master_fd, 4096)
                    if not chunk:
                        break
                    text = chunk.decode("utf-8", errors="replace")
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    log_f.write(text)
                    log_f.flush()
                elif proc.poll() is not None:
                    break
            except OSError:
                break

    os.close(master_fd)
    proc.wait()
    return proc.returncode


def append_timing(log_path: Path, start: datetime, end: datetime):
    elapsed = (end - start).total_seconds()
    lines = [
        "",
        "----------------------------------------",
        "命令执行完成！",
        f"开始时间：{start.strftime('%Y-%m-%d %H:%M:%S')}",
        f"结束时间：{end.strftime('%Y-%m-%d %H:%M:%S')}",
        f"总耗时：{elapsed:.3f} 秒",
        "========================================",
        "",
    ]
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("----------------------------------------")
    print(f"命令执行完成！总耗时：{elapsed:.3f} 秒")
    print(f"全部信息已写入日志文件：{log_path}")


def main():
    os.system("clear")

    log_path = resolve_log_path(cfg.LOG_FILE)

    env = None
    if cfg.API_KEY_SCRIPT:
        env = load_env_from_shell(cfg.API_KEY_SCRIPT)

    base_env = dict(env) if env else dict(os.environ)
    if not base_env.get("TINKER_KEY"):
        base_env["TINKER_KEY"] = (
            base_env.get("TINKER_API_KEY")
            or base_env.get("SKILLS_ONLY_TINKER_API_KEY")
            or ""
        )
    base_env.setdefault("METACLAW_ROOT", str(REPO_ROOT))
    base_env.setdefault("BENCHMARK_MODEL", "Qwen3-8B")

    port = find_free_port()
    print(f"[port] 自动分配端口: {port}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_skill_dir = str(Path(tempfile.gettempdir()) / f"metaclaw_paper_static_skills_full_{ts}")
    shutil.copytree(cfg.ORIGINAL_SKILL_DIR, temp_skill_dir)
    print(f"[skills] 已复制初始 skill 目录到: {temp_skill_dir}")

    tmp_config_path = write_proxy_config(base_env, temp_skill_dir, port)
    proxy_proc = start_proxy(base_env, tmp_config_path, port)

    bench_env = dict(base_env)
    bench_env["METACLAW_PROXY_PORT"] = str(port)

    start = datetime.now()
    try:
        run_cmd = [
            *bench_cmd(), "run",
            "-i", cfg.BENCH_INPUT,
            "-o", cfg.BENCH_OUTPUT,
            "-w", "1",
            "-n", str(cfg.BENCH_COUNT),
        ]
        run_command(run_cmd, log_path, env=bench_env, cwd=str(BENCH_ROOT))
    finally:
        stop_proxy(proxy_proc)
        if os.path.isdir(temp_skill_dir):
            shutil.rmtree(temp_skill_dir)
            print(f"[skills] 已清理临时 skill 目录: {temp_skill_dir}")
        if os.path.isfile(tmp_config_path):
            os.unlink(tmp_config_path)
            print(f"[config] 已清理临时配置文件: {tmp_config_path}")

    end = datetime.now()
    append_timing(log_path, start, end)


if __name__ == "__main__":
    main()
