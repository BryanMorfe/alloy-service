import argparse
import sys
import time
from typing import Iterable

from .gpu_manager import SnapshotScope
from .runtime import get_runtime


def _print_table(rows: Iterable[str]) -> None:
    for row in rows:
        print(row)


def cmd_list() -> int:
    runtime = get_runtime()
    models = runtime.registry.list_models()
    if not models:
        print("No models registered")
        return 0
    _print_table(models)
    return 0


def cmd_ps() -> int:
    runtime = get_runtime()
    manager = runtime.gpu_manager

    try:
        snapshot = manager.status_snapshot(scope=SnapshotScope.SYSTEM)
        scope_label = "system"
    except Exception as exc:
        snapshot = manager.status_snapshot(scope=SnapshotScope.MANAGER)
        scope_label = "manager"
        print(f"Warning: system snapshot unavailable ({exc})", file=sys.stderr)

    print(f"GPU usage ({scope_label})")
    for state in snapshot.gpus:
        print(
            f"- {state.gpu_id}: total={state.total_vram_mb}MB "
            f"used={state.used_vram_mb}MB free={state.free_vram_mb}MB"
        )

    if snapshot.allocations:
        print("Allocated models")
        now = time.time()
        for record in snapshot.allocations:
            idle_s = int(max(now - record.last_used_at, 0))
            gpu_ids = sorted(record.vram_by_gpu_mb.keys())
            gpu_list = ",".join(gpu_ids)
            vram_detail = ",".join(
                f"{gpu_id}:{record.vram_by_gpu_mb[gpu_id]}MB" for gpu_id in gpu_ids
            )
            parts = (
                " parts=" + ",".join(f"{part}->{gpu}" for part, gpu in record.gpu_assignment.items())
                if record.gpu_assignment
                else ""
            )
            print(
                f"- {record.model.model_id}: gpus={gpu_list} "
                f"vram={vram_detail} active={record.active_requests} "
                f"idle_s={idle_s}{parts}"
            )
    else:
        print("Allocated models: none")

    if snapshot.queue:
        print("Queued models")
        for request in snapshot.queue:
            print(
                f"- {request.model.model_id}: priority={request.priority} "
                f"requested_at={int(request.requested_at)}"
            )

    return 0


def cmd_serve(host: str, port: int) -> int:
    try:
        import uvicorn
    except Exception as exc:
        print(f"uvicorn is required to run the server ({exc})", file=sys.stderr)
        return 1

    uvicorn.run("alloyai.server:app", host=host, port=port, reload=False)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="alloy", description="Alloy CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List available models")
    subparsers.add_parser("ps", help="Show GPU status and allocations")

    serve_parser = subparsers.add_parser("serve", help="Run the HTTP server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command == "list":
        return cmd_list()
    if args.command == "ps":
        return cmd_ps()
    if args.command == "serve":
        return cmd_serve(args.host, args.port)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
