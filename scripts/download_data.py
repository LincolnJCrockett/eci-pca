"""Download and extract Epoch AI benchmark data. Usage: python -I scripts/download_data.py data

Epoch updates this file regularly, so a new download will usually differ from the snapshot the saved results were computed on.
The script records the download date and SHA-256 of what it fetched in <dest>/SNAPSHOT.txt and warns if it differs from
ANALYSED_SHA256 (the snapshot used for the results in this repo; see README "Data snapshot")."""
import sys, io, zipfile, hashlib, datetime, urllib.request, pathlib

ANALYSED_SHA256 = "d2c8ab82bfce75c8f244232ee0027ddc648a5b30a13e73f847618875e8d31d8f"   # downloaded 2026-11-15

dest = pathlib.Path(sys.argv[1]); (dest / "raw").mkdir(parents=True, exist_ok=True)
blob = urllib.request.urlopen("https://epoch.ai/data/benchmark_data.zip").read()
(dest / "raw" / "benchmark_data.zip").write_bytes(blob)
zipfile.ZipFile(io.BytesIO(blob)).extractall(dest / "extracted")
sha = hashlib.sha256(blob).hexdigest()
(dest / "SNAPSHOT.txt").write_text(f"downloaded {datetime.date.today().isoformat()}\nsha256 {sha}\nbytes {len(blob)}\n")
print("ok", len(blob), "bytes, sha256", sha)
if sha != ANALYSED_SHA256:
    print("WARNING: this is not the data snapshot the saved results were computed on; numbers will differ.")
