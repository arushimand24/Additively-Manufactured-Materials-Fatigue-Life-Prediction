# Dataset Provenance Record

This record states where the data came from, when and how it was obtained, and how its integrity was checked. The machine-readable version is `results/audit/file_provenance.json`; the full API response is `results/audit/figshare_api_record.json`.

## Source

| Item | Value |
|---|---|
| Dataset | Fatigue Database of Additively Manufactured Alloys (FatigueData-AM2022) |
| Creators | Zian Zhang, Zhiping Xu (Applied Mechanics Laboratory, Department of Engineering Mechanics, Tsinghua University) |
| Landing page | https://figshare.com/articles/dataset/Fatigue_Database_of_Additively_Manufactured_Alloys/22337629 |
| DOI | 10.6084/m9.figshare.22337629 (version-specific: 10.6084/m9.figshare.22337629.v2) |
| Figshare version | 2, published 2023-06-01T09:10:55Z |
| Licence | CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/) |
| Descriptor paper | Zhang, Z. & Xu, Z. Fatigue database of additively manufactured alloys. *Scientific Data* 10, 249 (2023). https://doi.org/10.1038/s41597-023-02150-x |

## How the files were obtained

- Download date (UTC): **2026-11-19T21:23:36+00:00** (20 November 2026, about 02:53 Indian Standard Time).
- Method: the public Figshare API record `https://api.figshare.com/v2/articles/22337629` was queried for the file list, and each file was fetched from the `download_url` that the API returned. No other source was used.
- The download can be repeated without manual steps: `src/data_utils.ensure_raw_data()` performs the same query and MD5 check, and notebook 01 calls it when the workbook is missing.

## Files

| | Workbook (used by the pipeline) | JSON (downloaded, not used by the pipeline) |
|---|---|---|
| Original file name | `FatigueData-AM2022.xlsx` | `FatigueData-AM2022.json` |
| Stored at | `data/raw/FatigueData-AM2022.xlsx` | `data/raw/FatigueData-AM2022.json` |
| Download URL | https://ndownloader.figshare.com/files/41023484 | https://ndownloader.figshare.com/files/41023481 |
| Figshare file id | 41023484 | 41023481 |
| Size listed by Figshare | 7,044,408 bytes | 69,558,939 bytes |
| Size on disk | 7,044,408 bytes | 69,558,939 bytes |
| MD5 published by Figshare | `0a7db5fe40892cc420b9ac3713e596ad` | `f0d565902c000b277480bb6fa2e5f711` |
| MD5 on disk | `0a7db5fe40892cc420b9ac3713e596ad` (match: True) | `f0d565902c000b277480bb6fa2e5f711` (match: True) |
| SHA-256 on disk | `95c0c41266f32a275bcef834bfd7674079861a2d00d3974e5e6fd0ad7100528b` | `36a5ed57538739c37056bc956880f7c89bed009090999ea214d42612cf2c40de` |

The record also lists `FatigueData-AM2022.mat` (MATLAB format). It holds the same data and was not downloaded.

The JSON file was fetched as a fallback in case the workbook lacked article metadata. The audit showed that the workbook carries title, authors and DOI for every dataset, so the JSON is not read by any notebook. It was only inspected once to confirm that it holds the same 1,610 S-N, 236 e-N and 614 da/dN datasets.

## Integrity checks performed

1. File size equals the size listed by Figshare, for both files.
2. MD5 equals the MD5 published by Figshare, for both files.
3. The workbook opens with `openpyxl` and contains the four expected sheets: `S-N`, `e-N`, `dadn`, `parameter`.
4. Notebook 01 recomputes the SHA-256 at every run and stops if it differs from this record.
5. The headline counts reproduce the figures reported by the dataset authors exactly: 15,146 S-N points, 1,610 S-N datasets, 459 S-N source articles (table `results/audit/counted_vs_reported.csv`).

## Handling rules

- Files in `data/raw/` are never modified, renamed or overwritten. `ensure_raw_data()` refuses to overwrite an existing file.
- Every derived table is written to `data/processed/` or `results/`.
- `data/raw/` is excluded from version control by `.gitignore`.

## How to cite

Zhang, Z. & Xu, Z. Fatigue database of additively manufactured alloys. *Scientific Data* 10, 249 (2023). https://doi.org/10.1038/s41597-023-02150-x

Zhang, Z. & Xu, Z. Fatigue Database of Additively Manufactured Alloys (version 2). Figshare (2023). https://doi.org/10.6084/m9.figshare.22337629

The data are used under the CC BY 4.0 licence, which requires attribution.
