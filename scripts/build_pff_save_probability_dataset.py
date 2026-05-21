"""Build a save-probability training table from PFF FC World Cup 2022 files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pff_ingestion import build_pff_save_probability_rows, hydrate_rows_with_pff_tracking, load_pff_events  # noqa: E402


def _resolve_events_path(pff_root: Path, events_path: Path | None) -> Path:
    candidates = []
    if events_path:
        candidates.append(events_path)
    candidates.extend(
        [
            pff_root / "events.json",
            pff_root / "event_data" / "events.json",
            pff_root / "events" / "events.json",
        ]
    )
    if pff_root.exists() and any(pff_root.glob("*.json")):
        candidates.append(pff_root)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find PFF events.json. Checked:\n{checked}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert PFF FC World Cup 2022 shot events into the tabular "
            "keeper_state + shot_state + outcome format used by the demo model."
        )
    )
    parser.add_argument("--pff-root", type=Path, default=Path("data/pff_worldcup_2022"))
    parser.add_argument("--events", type=Path, default=None, help="Optional explicit path to events.json.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/pff_worldcup_2022_save_probability.csv"),
    )
    parser.add_argument("--include-unlabeled", action="store_true", help="Keep misses/blocks in the output for analysis.")
    parser.add_argument("--keeper-default-u", type=float, default=0.5)
    parser.add_argument("--keeper-default-v", type=float, default=-0.45)
    parser.add_argument(
        "--tracking-dir",
        type=Path,
        default=None,
        help="Optional directory with <game_id>.jsonl or <game_id>.jsonl.bz2 tracking files.",
    )
    parser.add_argument("--strict-tracking", action="store_true", help="Fail if a required tracking file is missing.")
    parser.add_argument("--preview", type=int, default=5, help="Rows to print after writing.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        events_path = _resolve_events_path(args.pff_root, args.events)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        print("Place the PFF download under data/pff_worldcup_2022/ or pass --events.")
        return 1

    events = load_pff_events(events_path)
    rows = build_pff_save_probability_rows(
        events,
        include_unlabeled=args.include_unlabeled,
        keeper_default_u=args.keeper_default_u,
        keeper_default_v=args.keeper_default_v,
    )
    tracking_dir = args.tracking_dir or (args.pff_root / "tracking")
    if tracking_dir.exists() or args.strict_tracking:
        rows = hydrate_rows_with_pff_tracking(rows, tracking_dir, strict=args.strict_tracking)
    if rows.empty:
        print("[ERROR] No usable shot rows found. Check the PFF events schema/export.")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out, index=False)

    labeled = rows["save_label"].notna() if "save_label" in rows.columns else []
    print(f"PFF events: {len(events)}")
    print(f"Shot rows written: {len(rows)}")
    print(f"Labeled save/goal rows: {int(labeled.sum()) if len(rows) else 0}")
    print(f"Output: {args.out}")
    fallback_count = int((rows.get("keeper_source") == "fallback_center").sum()) if "keeper_source" in rows.columns else 0
    if fallback_count:
        print(f"WARNING: {fallback_count} rows used fallback keeper position; add tracking/freeze-frame fields for stronger ML.")
    if args.preview > 0:
        print(rows.head(args.preview).to_string(index=False))
    print("Next:")
    print(f"  python scripts/train_save_probability_model.py --data {args.out} --outcome-col outcome")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
