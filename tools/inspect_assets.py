"""Print compact diagnostics about the generated drawing instructions."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "public" / "assets"


def main() -> None:
    instructions = json.loads((ASSETS / "drawing-instructions.json").read_text(encoding="utf-8"))
    report = json.loads((ASSETS / "motion-report.json").read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for track in instructions["tracks"]:
        for state in track["states"]:
            counts[state["primitive"]] = counts.get(state["primitive"], 0) + 1
    print("instruction bytes", (ASSETS / "drawing-instructions.json").stat().st_size)
    print("primitive state counts", counts)
    print(
        "track summaries",
        [
            (t["id"], t["length"], t["totalDisplacement"], t["meanDeformation"], t["primitive"])
            for t in instructions["tracks"]
        ],
    )
    print("first samples", report["samples"][:3])
    print("last samples", report["samples"][-3:])
    for asset in sorted(ASSETS.iterdir()):
        print(asset.name, asset.stat().st_size)


if __name__ == "__main__":
    main()
