"""
summarize_datasets.py — TỔNG HỢP OUTPUT cho từng dataset sau khi chạy pipeline.

Đọc mọi aggregate.json (PPO/DQN, mean±std qua seed) + comparison.csv (4 scheduler)
rồi xuất:
  - outputs/summary_all_datasets.csv   (bảng gộp, dán vào báo cáo/slide)
  - in bảng gọn ra màn hình

Chạy trong thư mục gốc project:  python summarize_datasets.py
Tự nhận diện mọi dataset đã chạy (predictive_maintenance, german_credit, pima, heart,
parkinsons, ...) — không cần sửa gì.
"""
import json, csv, glob, os, re
from pathlib import Path

ROOT = Path("outputs/rl_results/simulate")
COMP = Path("outputs/comparison")
rows = []

def agg(path, key):
    try:
        j = json.load(open(path))
        if key in j: return round(j[key]["mean"],3), round(j[key]["std"],3)
    except Exception: pass
    return None, None

# 1) Multiseed aggregates (PPO/DQN baseline) theo từng dataset
for d in sorted(ROOT.glob("*_local_baseline")):
    m = re.match(r"(ppo|dqn)_(.+)_local_baseline", d.name)
    if not m: continue
    agent, ds = m.group(1).upper(), m.group(2)
    a = d/"aggregate.json"
    if not a.exists(): continue
    acc, accs = agg(a,"acc_end"); f1,f1s = agg(a,"f1_end")
    dep,deps = agg(a,"depth_end"); rb,rbs = agg(a,"n_rollbacks")
    rows.append({"dataset":ds,"agent":agent,"source":"multiseed(3)",
                 "acc":f"{acc}±{accs}" if acc is not None else "-",
                 "f1":f"{f1}±{f1s}" if f1 is not None else "-",
                 "depth":f"{dep}±{deps}" if dep is not None else "-",
                 "rollback":f"{rb}" if rb is not None else "-"})

# 2) Comparison 4-scheduler (single seed) theo dataset
for c in sorted(COMP.glob("*/comparison*.csv")):
    ds = c.parent.name.replace("_q8_local","").replace("_local","")
    try:
        for r in csv.DictReader(open(c)):
            rows.append({"dataset":ds,"agent":r["scheduler"].upper(),"source":"compare(1seed)",
                         "acc":round(float(r["acc_end"]),3),"f1":round(float(r["f1_end"]),3),
                         "depth":f'{r["depth_start"]}->{r["depth_end"]}',
                         "rollback":r.get("n_rollbacks","-")})
    except Exception as e:
        print("skip",c,e)

# xuất CSV
os.makedirs("outputs", exist_ok=True)
out = Path("outputs/summary_all_datasets.csv")
if rows:
    with open(out,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset","agent","source","acc","f1","depth","rollback"])
        w.writeheader(); w.writerows(rows)
    print(f"\nLUU -> {out}  ({len(rows)} dong)\n")
    # in gọn
    print(f'{"dataset":<22}{"agent":<8}{"source":<15}{"acc":<14}{"f1":<14}{"depth":<12}{"rb"}')
    print("-"*95)
    for r in rows:
        print(f'{r["dataset"]:<22}{r["agent"]:<8}{r["source"]:<15}{str(r["acc"]):<14}{str(r["f1"]):<14}{str(r["depth"]):<12}{r["rollback"]}')
else:
    print("Chua co ket qua nao. Chay run_new_datasets.ps1 truoc.")
