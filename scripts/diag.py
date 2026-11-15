import sys, pandas as pd, numpy as np
ex=sys.argv[1]
d=pd.read_csv(f"{ex}/epoch_capabilities_index/processed_data_for_eci.csv")
M=d.pivot_table(index="model",columns="benchmark",values="performance")
M=M.loc[:,M.notna().sum()>=25]
O=M.notna().astype(int).T@M.notna().astype(int)
old=["TriviaQA","HellaSwag","OpenBookQA","GSM8K","ARC AI2","PIQA","BBH","Winogrande","MMLU"]
print("overlap counts, old rows (min/median across other benchmarks):")
for b in old: 
    o=O.loc[b].drop(b); print(f"{b:12s} min {o.min():3d} med {o.median():4.0f} #<12: {(o<12).sum()}  mean score {M[b].mean():.2f}")
print("old-vs-old corr:\n",M[old].corr(min_periods=12).round(2).to_string())
e=pd.read_csv(f"{ex}/epoch_capabilities_index/eci_scores.csv"); print(e.head(3).to_string()); print(set(M.index)&set(e.Model), len(set(M.index)&set(e["Display name"])), len(set(M.index)&set(e["Model"])))
