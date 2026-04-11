# Data Directory

Raw data is not committed to the public repository.

Place the required files here locally when running the project:

- `released_df.csv.gz` or `released_df.csv`
- `split_stay_id.json`

Expected behavior:

- the code reads raw data from the local data directory
- large generated files are written to ignored output folders

Do not commit patient-level raw data or regenerated large CSV outputs.
