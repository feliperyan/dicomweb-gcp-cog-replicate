# PoC 

This is a proof of concept for running a DICOM classification model on Replicate using a GCP DICOM store as the data source.

## Setup

1. Install dependencies: `uv sync`
2. On GCP:
    1. create a Healthcare dataset
    2. Create a Dicom Store
    3. Upload the .dcm files to a GCS bucket
    4. Import .dcm from GCS to Dicom Store
3. Acquire a Service Account key: `gcloud auth application-default login` and name it `gcp_key.json`
4. Sanity check: Look at `fetch.py` make the necessary changes and run `uv run python fetch.py`
5. Run with `cog predict`. You can look at .cog/openapi_schema.json for the API schema.

