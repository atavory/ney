# Authoritative Section 4 release

This is the single entry point for the submission-facing Section 4 evidence. `source_map.csv` enumerates the component bundles and their verification hashes. `paper_values.csv` gives every manuscript-facing value, its exact source selector, and the source-file SHA-256. The generated TeX files are copied verbatim into the paper repository and must not be edited by hand.

Rebuild into an empty directory with:

```sh
python3 scripts/assemble_section4_release.py --data-root . --out-dir REBUILT
```
