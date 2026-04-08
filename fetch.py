"""Fetch DICOM data from a Google Cloud Healthcare DICOM store.

Uses DICOMweb (QIDO-RS for search, WADO-RS for retrieval) via the
dicomweb_client library, authenticated with a GCP service account key file.

Usage:
    uv run python fetch.py

Downloaded instances are saved to ./downloads/ as <SOPInstanceUID>.dcm and
<SOPInstanceUID>.png alongside each other.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pydicom
import google.auth.transport.requests
import google.oauth2.service_account
from dicomweb_client import DICOMwebClient


# --- Configuration -----------------------------------------------------------

PROJECT = "cloudflare-fryan-01"
LOCATION = "us-central1"
DATASET = "fryan-healthdataset-01"
DICOM_STORE = "fryan-nih-dicom"

BASE_URL = (
    f"https://healthcare.googleapis.com/v1"
    f"/projects/{PROJECT}"
    f"/locations/{LOCATION}"
    f"/datasets/{DATASET}"
    f"/dicomStores/{DICOM_STORE}"
    f"/dicomWeb"
)

DOWNLOADS_DIR = Path(__file__).parent / "downloads"
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
SERVICE_ACCOUNT_KEY = Path(__file__).parent / "gcp_key.json"


# --- Auth --------------------------------------------------------------------

def make_client() -> DICOMwebClient:
    """Create a DICOMwebClient authenticated via a service account key file."""
    credentials = google.oauth2.service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_KEY, scopes=SCOPES
    )
    session = google.auth.transport.requests.AuthorizedSession(credentials)
    return DICOMwebClient(url=BASE_URL, session=session)


# --- Helpers -----------------------------------------------------------------

def tag(dataset: dict, keyword: str, default: str = "") -> str:
    """Extract the first string value of a DICOM JSON tag by keyword."""
    from pydicom.datadict import tag_for_keyword
    t = tag_for_keyword(keyword)
    if t is None:
        return default
    key = f"{t:08X}"
    # DICOM JSON uses uppercase hex with no separator
    key = f"{(t >> 16) & 0xFFFF:04X}{t & 0xFFFF:04X}"
    entry = dataset.get(key, {})
    values = entry.get("Value", [])
    return str(values[0]) if values else default


# --- Main --------------------------------------------------------------------

def main() -> None:
    print(f"Connecting to DICOM store: {DICOM_STORE}")
    client = make_client()

    # --- Search for studies (QIDO-RS) ----------------------------------------
    print("\nSearching for studies...")
    studies = client.search_for_studies()
    if not studies:
        print("No studies found in the DICOM store.")
        sys.exit(0)

    print(f"Found {len(studies)} study/studies:\n")

    DOWNLOADS_DIR.mkdir(exist_ok=True)
    total_saved = 0

    for study in studies:
        study_uid = tag(study, "StudyInstanceUID")
        patient_id = tag(study, "PatientID", default="unknown")
        print(f"  Study: {study_uid}  (PatientID: {patient_id})")

        # --- Search for series within the study (QIDO-RS) --------------------
        series_list = client.search_for_series(study_instance_uid=study_uid)
        print(f"    {len(series_list)} series")

        for series in series_list:
            series_uid = tag(series, "SeriesInstanceUID")
            modality = tag(series, "Modality", default="??")
            print(f"    Series: {series_uid}  modality={modality}")

            # --- Search for instances within the series (QIDO-RS) ------------
            instances = client.search_for_instances(
                study_instance_uid=study_uid,
                series_instance_uid=series_uid,
            )
            print(f"      {len(instances)} instance(s)")

            for instance in instances:
                sop_uid = tag(instance, "SOPInstanceUID")

                # --- Retrieve pixel data (WADO-RS) ---------------------------
                raw = client.retrieve_instance(
                    study_instance_uid=study_uid,
                    series_instance_uid=series_uid,
                    sop_instance_uid=sop_uid,
                )

                # raw is a pydicom Dataset when returned by dicomweb_client
                dcm_path = DOWNLOADS_DIR / f"{sop_uid}.dcm"
                if isinstance(raw, pydicom.Dataset):
                    ds = raw
                    pydicom.dcmwrite(str(dcm_path), ds)
                else:
                    # Fallback: raw bytes — re-read to get a Dataset for PNG export
                    dcm_path.write_bytes(bytes(raw))
                    ds = pydicom.dcmread(str(dcm_path))

                # Save alongside as PNG
                png_path = DOWNLOADS_DIR / f"{sop_uid}.png"
                pixel_array = ds.pixel_array
                # Normalise to 8-bit for display if needed
                if pixel_array.dtype != np.uint8:
                    pmin, pmax = pixel_array.min(), pixel_array.max()
                    if pmax > pmin:
                        pixel_array = ((pixel_array - pmin) / (pmax - pmin) * 255).astype(np.uint8)
                    else:
                        pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
                Image.fromarray(pixel_array, mode="L").save(png_path)

                print(f"      Saved: {dcm_path.name}  +  {png_path.name}")
                total_saved += 1

    print(f"\nDone. {total_saved} instance(s) saved to {DOWNLOADS_DIR}/")


if __name__ == "__main__":
    main()
