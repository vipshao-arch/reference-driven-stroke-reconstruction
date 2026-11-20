"""Remove duplicate head metadata introduced during source authoring."""

from pathlib import Path


path = Path(__file__).resolve().parents[1] / "index.html"
lines = path.read_text(encoding="utf-8").splitlines()
seen_description = 0
seen_title = 0
cleaned = []
for line in lines:
    if '<meta name="description"' in line:
        seen_description += 1
        if seen_description == 1:
            continue
    if '<title>' in line:
        seen_title += 1
        if seen_title == 1:
            continue
    cleaned.append(line)
path.write_text("\n".join(cleaned) + "\n", encoding="utf-8")
print(f"removed duplicate metadata: description={seen_description - 1}, title={seen_title - 1}")
