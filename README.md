# ECI PCA

PCA of the benchmarks behind the [Epoch Capabilities Index](https://epoch.ai/eci). See `RESULTS.md` for findings and `PLAN.md` for method.

## Reproduce
```
pip install pandas scikit-learn matplotlib scipy
python -I scripts/download_data.py data
python -I scripts/run_pca.py data/extracted outputs
python -I scripts/interpret.py data/extracted outputs
python -I scripts/residual_pca.py data/extracted outputs   # superseded first pass
python -I scripts/spec_pca.py data/extracted outputs/spec_logit logit   # main analysis, see RESULTS.md
```

Data: Epoch AI, "Capabilities & benchmarking", https://epoch.ai/benchmarks (CC-BY 4.0). Not redistributed here; downloaded by the script above.
