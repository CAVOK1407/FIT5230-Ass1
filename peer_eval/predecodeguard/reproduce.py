"""Visible, user-started local replay of the unchanged seven-plus-three pilot."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

from .contract import (
    HERE, check_protocol, package_manifest, read_json, require_agreement,
    sha256, write_json,
)


REPLAY = HERE / "artifacts/local_reproduction_2026-10-18"
REFERENCE = HERE / "artifacts/pilot_blind_7plus3_2026-10-18"
PACKAGE = HERE.parents[3] / "predecodeguard/PreDecodeGuard_ModelBreakers_Pilot_v1"
TIMEZONE = ZoneInfo("Asia/Kuala_Lumpur")


def preflight():
    path = REPLAY / "protocol.local_reproduction.json"
    protocol = read_json(path)
    source_path = REFERENCE / "protocol.confirmed.json"
    reference_csv = REFERENCE / "verification/per_sample.csv"
    replay = protocol["local_reproduction"]
    if sha256(source_path) != replay["source_protocol_sha256"]:
        raise ValueError("The source protocol changed; stop and review the replay settings")
    if sha256(reference_csv) != replay["reference_csv_sha256"]:
        raise ValueError("The saved comparison record changed")
    source = read_json(source_path)
    for key in ("budget", "threshold", "methods", "sample_filenames",
                "requested_methods", "returned_batches", "seed_rule", "review",
                "control_selection"):
        if protocol[key] != source[key]:
            raise ValueError(f"Replay differs from the agreed configuration: {key}")
    manifest = package_manifest(PACKAGE)
    check_protocol(protocol, manifest, sha256(PACKAGE / "manifest.json"))
    require_agreement(protocol, "lstop", execute_approved=True, protocol_path=path)
    with reference_csv.open(newline="", encoding="utf-8") as handle:
        reference_rows = list(csv.DictReader(handle))
    names = {row["filename"] for row in manifest["samples"]}
    if len(reference_rows) != 10 or {row["filename"] for row in reference_rows} != names:
        raise ValueError("The saved comparison record must contain exactly ten inputs")
    for row in reference_rows:
        if sha256(REFERENCE / "verification/returned" / row["filename"]) != row["sha256"]:
            raise ValueError(f"The saved comparison file changed: {row['filename']}")
    return path, protocol, reference_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true",
                        help="Validate configuration and files without optimizing or creating a run")
    args = parser.parse_args(argv)
    protocol_path, protocol, reference_rows = preflight()
    cfg = protocol["methods"]["lstop"]
    controls = len(cfg["control_filenames"])
    print(f"配置检查通过：{10 - controls} 个早停 PGD 样本 + {controls} 个原样对照。", flush=True)
    print(f"预算 L∞ ≤ {protocol['budget']['epsilon']}；最多 {cfg['steps']} 步；"
          f"步长 {cfg['alpha']}；早停 margin {cfg['early_stop_margin']}。", flush=True)
    if args.check_only:
        print("仅检查完成：没有运行攻击，也没有创建新的实验结果。", flush=True)
        return 0

    # Import the experiment runner only for the user-started real replay.
    from .run import main as run_experiment

    now = datetime.now(TIMEZONE)
    runs = REPLAY / "runs"
    runs.mkdir(exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix=f"replay_{now:%Y%m%d_%H%M%S}_", dir=runs))
    out = session / "run"
    write_json(session / "replay_session.json", {
        "started_at": now.isoformat(), "purpose": "user_requested_local_reproduction",
        "protocol_sha256": sha256(protocol_path),
        "reference_csv_sha256": protocol["local_reproduction"]["reference_csv_sha256"],
        "status": "starting", "external_submission": False,
    })
    print(f"\n开始实际复跑：{now:%Y-%m-%d %H:%M:%S %Z}", flush=True)
    print(f"新结果目录：{out}", flush=True)
    print("正在重新计算 PGD、保存 latent，并运行官方评分复验……", flush=True)
    try:
        code = run_experiment([
            "experiment", "--package", str(PACKAGE), "--protocol", str(protocol_path),
            "--method", "lstop", "--execute-approved", "--out", str(out),
        ])
    except Exception as error:
        session_record = read_json(session / "replay_session.json")
        session_record.update(status="failed", error=str(error),
                              finished_at=datetime.now(TIMEZONE).isoformat())
        write_json(session / "replay_session.json", session_record)
        raise
    if code:
        session_record = read_json(session / "replay_session.json")
        session_record.update(status="failed", exit_code=code,
                              finished_at=datetime.now(TIMEZONE).isoformat())
        write_json(session / "replay_session.json", session_record)
        print(f"运行未通过完整验证，请查看：{out / 'run_config.json'}", flush=True)
        return code

    summary = read_json(out / "summary.json")
    with (out / "per_sample.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    reference = {row["filename"]: row for row in reference_rows}
    comparisons = [{"filename": row["filename"], "new_sha256": row["sha256"],
                    "reference_sha256": reference[row["filename"]]["sha256"],
                    "identical_bytes": row["sha256"] == reference[row["filename"]]["sha256"]}
                   for row in rows]
    identical = sum(row["identical_bytes"] for row in comparisons)
    write_json(session / "reproducibility.json", {
        "finished_at": datetime.now(TIMEZONE).isoformat(),
        "new_optimization_run": True, "files_compared": len(comparisons),
        "identical_files": identical, "comparisons": comparisons,
        "external_submission": False,
    })
    print("\n实际运行结果（包含对照身份，仅供本地查看）：", flush=True)
    print(f"{'文件':<14} {'类型':<8} {'步数':>4} {'原分数':>11} {'新分数':>11} {'Linf':>10}")
    for row in rows:
        kind = "攻击" if row["attempted"] == "True" else "对照"
        print(f"{row['filename']:<14} {kind:<8} {row['steps']:>4} "
              f"{float(row['baseline_score']):11.7f} {float(row['score']):11.7f} "
              f"{float(row['linf']):10.7f}")
    print(f"\n运行验证通过；检测器改判：{summary['detector_flips']}/{summary['eligible_attempts']}。")
    print(f"与已验证的 7+3 结果逐文件哈希一致：{identical}/{len(comparisons)}。")
    print(f"完成时间：{datetime.now(TIMEZONE):%Y-%m-%d %H:%M:%S %Z}")
    print(f"结果报告：{out / 'summary.md'}")
    print("内容保持是否成功仍待人工复核。当前结果含原名和分数，不是对外盲审包。", flush=True)
    write_json(session / "replay_session.json", {
        "started_at": now.isoformat(), "finished_at": datetime.now(TIMEZONE).isoformat(),
        "purpose": "user_requested_local_reproduction", "protocol_sha256": sha256(protocol_path),
        "status": "reproduced" if identical == 10 else "verified_but_hashes_differ",
        "external_submission": False,
    })
    return 0 if identical == 10 else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, PermissionError, OSError) as error:
        print(f"停止：{error}", flush=True)
        raise SystemExit(2)
