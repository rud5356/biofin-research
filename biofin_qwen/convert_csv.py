import csv, json
from pathlib import Path

src = Path(r"C:\Users\lenovo\Downloads\sentiment_train_20.csv")
dst = Path(__file__).with_name("sentiment.jsonl")
with src.open(encoding="utf-8-sig", newline="") as f, dst.open("w", encoding="utf-8") as out:
    for row in csv.DictReader(f):
        json.dump({"text": row["text"], "label": row["label"]}, out, ensure_ascii=False)
        out.write("\n")
print(f"변환 완료: {dst}")
