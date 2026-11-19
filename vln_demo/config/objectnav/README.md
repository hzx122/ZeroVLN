# Optional ObjectNav inputs

ObjectNav is a legacy optional workflow, separate from the paper's SAGE/R2R/RxR
experiments. Dataset catalogs, scene assets, and derived episode subsets are not
included in the anonymous supplement.

For a configured legacy DLC deployment, provide an external normalized catalog
with `--objectnav-manifest /path/to/catalog.jsonl` (or `.jsonl.gz`). Alternatively,
set `manifest` in `../objectnav.json` to that external path. Its default is `null`.
The interactive service accepts the `OBJECTNAV_MANIFEST` environment variable.
Batch evaluation additionally requires an adjacent `lock.json` containing the
catalog SHA256 and episode count, as checked by `batch/objectnav_batch.py`.
