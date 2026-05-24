import glob
import os
import pandas as pd

src_dir = "xiaoe_articles"
dst_dir = "my_strategy/strategies"
keep_cols = ["日期", "编号", "持有股票", "持有股票代码"]

os.makedirs(dst_dir, exist_ok=True)

for f in glob.glob(os.path.join(src_dir, "*.csv")):
    df = pd.read_csv(f)
    df = df[keep_cols]
    out = os.path.join(dst_dir, os.path.basename(f))
    df.to_csv(out, index=False)
    print(f"{os.path.basename(f)} -> {out}  ({len(df)} rows)")

print("Done.")
