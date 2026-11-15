# ECI PCA

PCA of the benchmarks behind the [Epoch Capabilities Index](https://epoch.ai/eci). See `RESULTS.md` for findings, `METHODOLOGY.md` for the equations and caveats, and `PLAN.md` for the original plan.

## Reproduce
```
pip install pandas scikit-learn matplotlib scipy
python -I scripts/download_data.py data
python -I scripts/run_pca.py data/extracted outputs
python -I scripts/interpret.py data/extracted outputs
python -I scripts/residual_pca.py data/extracted outputs   # superseded first pass
python -I scripts/spec_pca.py data/extracted outputs/spec_logit logit   # earlier analysis (partly superseded)
python -I scripts/variance_and_eci_resid.py data/extracted outputs/variance   # main analysis, see RESULTS.md
python -I scripts/eci_vs_components_v2.py data/extracted outputs/variance
python -I scripts/loss_comparison.py data/extracted outputs/loss_comparison
python -I scripts/make_figures.py data/extracted outputs outputs/figures
```

Data: Epoch AI, "Capabilities & benchmarking", https://epoch.ai/benchmarks (CC-BY 4.0). Not redistributed here; downloaded by the script above.
