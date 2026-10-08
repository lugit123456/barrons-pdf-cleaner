#!/usr/bin/env python3
"""Poll a configured folder and process Barron's PDF issues."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VALID_SELECTION_MODES = {"latest", "all_unprocessed"}
VALID_CONTENT_MODES = {"original", "bilingual"}


def load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config 必须是 JSON object")
    required = ("input_dir", "output_dir", "state_db")
    missing = [key for key in required if not str(data.get(key) or "").strip()]
    if missing:
        raise ValueError(f"config 缺少必填字段：{', '.join(missing)}")
    mode = str(data.get("selection_mode") or "latest").strip().lower()
    if mode not in VALID_SELECTION_MODES:
        raise ValueError("selection_mode 必须是 latest 或 all_unprocessed")
    data["selection_mode"] = mode
    content_mode = str(data.get("content_mode") or "original").strip().lower()
    if content_mode not in VALID_CONTENT_MODES:
        raise ValueError("content_mode 必须是 original 或 bilingual")
    data["content_mode"] = content_mode
    return data


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def publication_sort_key(path: Path) -> tuple[str, float, str]:
    from metadata_parser import parse_pdf_metadata

    metadata = parse_pdf_metadata(str(path))
    try:
        modified_at = path.stat().st_mtime
    except OSError:
        modified_at = 0.0
    return metadata.publication_date, modified_at, path.name.lower()


def discover_candidates(
    config: dict[str, Any],
    *,
    config_path: Path,
    file_selectors: list[str] | None = None,
) -> list[Path]:
    base_dir = config_path.parent
    input_dir = resolve_path(str(config["input_dir"]), base_dir)
    if not input_dir.exists():
        print(f"⚠️ 输入目录不存在：{input_dir}")
        return []

    pattern = str(config.get("pattern") or "*Barron*.pdf")
    iterator = (
        input_dir.rglob(pattern)
        if bool(config.get("recursive", False))
        else input_dir.glob(pattern)
    )
    candidates = [path for path in iterator if path.is_file()]
    if file_selectors:
        candidates = [
            path
            for path in candidates
            if any(
                fnmatch.fnmatch(path.name.lower(), selector)
                or fnmatch.fnmatch(path.stem.lower(), selector)
                for selector in file_selectors
            )
        ]

    stable_seconds = max(float(config.get("stable_seconds", 20)), 0)
    min_pdf_bytes = max(int(config.get("min_pdf_bytes", 1024)), 1)
    candidates = [
        path
        for path in candidates
        if is_ready_pdf(
            path,
            stable_seconds=stable_seconds,
            min_bytes=min_pdf_bytes,
        )
    ]
    candidates.sort(key=publication_sort_key, reverse=True)
    if config["selection_mode"] == "latest":
        return candidates[:1]
    return candidates


def is_ready_pdf(
    path: Path,
    *,
    stable_seconds: float,
    min_bytes: int,
) -> bool:
    try:
        stat = path.stat()
    except OSError:
        print(f"⏭️ PDF 暂时无法读取，本轮跳过：{path}")
        return False
    if stat.st_size < min_bytes:
        print(
            f"⏭️ PDF 文件过小，本轮跳过：{path.name} "
            f"({stat.st_size} bytes < {min_bytes} bytes)"
        )
        return False
    age_seconds = time.time() - stat.st_mtime
    if stable_seconds > 0 and age_seconds < stable_seconds:
        print(
            f"⏭️ PDF 最近仍有更新，本轮跳过：{path.name} "
            f"（已静置 {age_seconds:.0f}/{stable_seconds:.0f} 秒）"
        )
        return False
    if not has_pdf_eof_marker(path):
        print(f"⏭️ PDF 尚未写入完整 EOF，本轮跳过：{path.name}")
        return False
    return True


def has_pdf_eof_marker(path: Path, tail_bytes: int = 16384) -> bool:
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(size - tail_bytes, 0))
            return b"%%EOF" in stream.read()
    except OSError:
        return False


def process_once(
    config: dict[str, Any],
    *,
    config_path: Path,
    file_selectors: list[str] | None = None,
    dry_run: bool = False,
) -> int:
    candidates = discover_candidates(
        config,
        config_path=config_path,
        file_selectors=file_selectors,
    )
    if not candidates:
        print("ℹ️ 本轮没有可处理的 Barron's PDF。")
        return 0

    base_dir = config_path.parent
    output_dir = resolve_path(str(config["output_dir"]), base_dir)
    state_db = resolve_path(str(config["state_db"]), base_dir)
    publication = str(config.get("publication_type") or "BARRONS").strip()
    dotenv_value = str(config.get("dotenv_path") or "").strip()
    dotenv_path = resolve_path(dotenv_value, base_dir) if dotenv_value else None
    completed = 0

    for pdf in candidates:
        command = [
            sys.executable,
            str(ROOT / "main.py"),
            "--input-dir",
            str(pdf.parent),
            "--file",
            pdf.name,
            "--output-dir",
            str(output_dir),
            "--state-db",
            str(state_db),
            "--publication",
            publication,
        ]
        print(f"🔎 检查：{pdf}")
        if dry_run:
            print("   🧪 dry-run：不执行解析")
            completed += 1
            continue

        environment = os.environ.copy()
        if dotenv_path:
            environment["DOTENV_PATH"] = str(dotenv_path)
        if config["content_mode"] == "original":
            environment.update({
                "LLM_COMPILE_ARTICLES": "false",
                "LLM_ANALYZE_ARTICLE_IMAGES": "false",
                "LLM_GLOSSARY_ENABLED": "false",
                "BARRONS_LOCAL_SPLIT_FIRST": "true",
            })
        result = subprocess.run(command, cwd=ROOT, env=environment)
        if result.returncode != 0:
            print(f"❌ 清洗命令失败，退出码：{result.returncode}")
            continue
        completed += 1
    return completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="定时获取并清洗最新 Barron's PDF")
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--once", action="store_true", help="运行一轮后退出")
    parser.add_argument("--interval", type=int, help="覆盖配置中的轮询间隔秒数")
    parser.add_argument("--file", action="append", help="仅处理匹配的文件名或 glob")
    parser.add_argument("--dry-run", action="store_true", help="只显示选择结果")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    selectors = [
        value.strip().lower()
        for value in args.file or []
        if value.strip()
    ] or None

    try:
        while True:
            config = load_config(config_path)
            process_once(
                config,
                config_path=config_path,
                file_selectors=selectors,
                dry_run=args.dry_run,
            )
            if args.once:
                return
            interval = max(
                int(args.interval or config.get("poll_interval_seconds", 10800)),
                5,
            )
            print(f"💤 下次检查将在 {interval} 秒后进行。")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n🛑 已停止 Barron's PDF 目录轮询。")


if __name__ == "__main__":
    main()
