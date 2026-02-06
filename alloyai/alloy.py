import argparse
import logging
import time
from typing import Iterable

from .gpu_manager import SnapshotScope
from .logging_utils import configure_logging
from .runtime import get_runtime


def _print_table(rows: Iterable[str]) -> None:
    logger = logging.getLogger("alloyai.table")
    for row in rows:
        logger.info(row)


def cmd_list() -> int:
    runtime = get_runtime()
    models = runtime.registry.list_models()
    if not models:
        logging.getLogger("alloyai").info("No models registered")
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
        logging.getLogger("alloyai").warning(
            "System snapshot unavailable (%s)", exc, exc_info=False
        )

    logging.getLogger("alloyai").info("GPU usage (%s)", scope_label)
    for state in snapshot.gpus:
        logging.getLogger("alloyai").info(
            "- %s: total=%sMB used=%sMB free=%sMB",
            state.gpu_id,
            state.total_vram_mb,
            state.used_vram_mb,
            state.free_vram_mb,
        )

    if snapshot.allocations:
        logging.getLogger("alloyai").info("Allocated models")
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
            logging.getLogger("alloyai").info(
                "- %s: gpus=%s vram=%s active=%s idle_s=%s%s",
                record.model.model_id,
                gpu_list,
                vram_detail,
                record.active_requests,
                idle_s,
                parts,
            )
    else:
        logging.getLogger("alloyai").info("Allocated models: none")

    if snapshot.queue:
        logging.getLogger("alloyai").info("Queued models")
        for request in snapshot.queue:
            logging.getLogger("alloyai").info(
                "- %s: priority=%s requested_at=%s",
                request.model.model_id,
                request.priority,
                int(request.requested_at),
            )

    return 0


def cmd_serve(host: str, port: int, log_level: str | None) -> int:
    try:
        import uvicorn
    except Exception as exc:
        logging.getLogger("alloyai").error(
            "uvicorn is required to run the server (%s)", exc, exc_info=False
        )
        return 1

    uvicorn.run(
        "alloyai.server:app",
        host=host,
        port=port,
        reload=False,
        log_level=(log_level or "info"),
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="alloy", description="Alloy CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List available models")
    subparsers.add_parser("ps", help="Show GPU status and allocations")

    serve_parser = subparsers.add_parser("serve", help="Run the HTTP server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--log-level", default=None)
    serve_parser.add_argument("--log-prefix", default=None)
    serve_parser.add_argument("--log-format", default=None)

    args = parser.parse_args()
    configure_logging(args.log_level, args.log_prefix, args.log_format)

    if args.command == "list":
        return cmd_list()
    if args.command == "ps":
        return cmd_ps()
    if args.command == "serve":
        return cmd_serve(args.host, args.port, args.log_level)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
