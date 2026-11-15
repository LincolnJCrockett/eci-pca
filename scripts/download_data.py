"""Download and extract Epoch AI benchmark data. Usage: python -I scripts/download_data.py data"""
import sys, io, zipfile, urllib.request, pathlib
dest = pathlib.Path(sys.argv[1]); (dest / "raw").mkdir(parents=True, exist_ok=True)
blob = urllib.request.urlopen("https://epoch.ai/data/benchmark_data.zip").read()
(dest / "raw" / "benchmark_data.zip").write_bytes(blob)
zipfile.ZipFile(io.BytesIO(blob)).extractall(dest / "extracted")
print("ok", len(blob), "bytes")
