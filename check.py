import json

with open("cap.rc2.test1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

empty = []

for item in data:
    caption = item.get("caption")

    if caption is None or not str(caption).strip():
        empty.append(item)

print("Empty captions:", len(empty))

for item in empty:
    print(
        "pairid =", item.get("pairid"),
        "| reference =", item.get("reference"),
        "| caption =", repr(item.get("caption")),
    )