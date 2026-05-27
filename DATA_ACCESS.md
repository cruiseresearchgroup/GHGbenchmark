# Data Access and Redistribution Notes

## Public building component

The building-level benchmark is intended for public release, including:

- canonical building table
- climate table
- coordinate provenance / geocoding side table
- Sentinel-2 metadata
- Sentinel-2 embeddings
- sample package

These are hosted separately from this code repository and linked from the
project documentation.

## Restricted component

The company-level benchmark relies on upstream restricted data sources. This
repository includes scripts to:

- download / query upstream sources where permitted
- enrich with financial and text metadata
- build derived benchmark tables
- reproduce modeling and evaluation pipelines

This repository does **not** itself redistribute the restricted raw company
data.

## Expected public release structure

- GitHub repository -> code bundle
- dataset hosting -> building release and sample package
- dataset landing page -> explains building release and company reconstruction-only status
