# usage: python make_sentences.py <dataset.csv|xlsx> <text_column_name>
import re
import sys

import pandas as pd

path, col = sys.argv[1], sys.argv[2]
df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
s = df[col].dropna().astype(str).map(lambda t: re.sub(r"\s+", " ", t).strip())
s = [t for t in s if t]
with open("sentences.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(s))
print(len(s), "sentences written to sentences.txt")
