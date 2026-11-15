import sys, numpy as np, pandas as pd
ex, out = sys.argv[1], sys.argv[2]
a=pd.read_csv(f"{out}/loadings_pairwise.csv",index_col=0); b=pd.read_csv(f"{out}/loadings_em.csv",index_col=0)
cos=lambda x,y: float(x@y/np.linalg.norm(x)/np.linalg.norm(y))
print("cosine pw vs EM:", {c: round(abs(cos(a[c].values,b[c].values)),3) for c in ["PC1","PC2"]})
edi=pd.read_csv(f"{ex}/epoch_capabilities_index/edi_scores.csv").set_index("benchmark_name")
meta=pd.read_csv(f"{ex}/benchmark_metadata.csv").set_index("benchmark")
j=a.join(edi[["edi","estimated_slope_scaled"]]).join(meta[["release_date","random_baseline"]])
j["rel"]=pd.to_datetime(j.release_date).astype("int64")/86400e9/365.25
print("n matched EDI:", j.edi.notna().sum())
for c in ["PC1","PC2"]:
    print(c,"vs EDI (difficulty) r=%.2f"%j[c].corr(j.edi),"| vs release date r=%.2f"%j[c].corr(j.rel),"| vs slope r=%.2f"%j[c].corr(j.estimated_slope_scaled))
print(j[["PC1","PC2","edi","release_date"]].sort_values("PC2").round(2).to_string())
s=pd.read_csv(f"{out}/model_scores_em.csv",index_col=0)
e=pd.read_csv(f"{ex}/epoch_capabilities_index/eci_scores.csv").set_index("Model")
m=s.join(e[["eci","date"]]); m["t"]=pd.to_datetime(m.date).astype("int64")/86400e9/365.25
print("model PC1 vs ECI r=%.2f, PC2 vs ECI r=%.2f, PC2 vs model date r=%.2f, PC1 vs date r=%.2f"%(m.PC1.corr(m.eci),m.PC2.corr(m.eci),m.PC2.corr(m.t),m.PC1.corr(m.t)))
# PC2 after controlling for ECI
r=m.PC2-np.polyval(np.polyfit(m.eci,m.PC2,1),m.eci); print("PC2 resid vs date r=%.2f"%r.corr(m.t))
