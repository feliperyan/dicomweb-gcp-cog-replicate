"""Convert PNG images in ./images to DICOM (.dcm) files.

Images sharing the same filename prefix (e.g. 00000001_000.png and
00000001_001.png both have prefix 00000001) are assigned the same
StudyInstanceUID and SeriesInstanceUID, with a unique SOPInstanceUID
per image.

Output .dcm files are written alongside the source PNGs in ./images/.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import (
    ExplicitVRLittleEndian,
    SecondaryCaptureImageStorage,
    generate_uid,
)


IMAGES_DIR = Path(__file__).parent / "images"


def build_file_meta(sop_instance_uid: str) -> FileMetaDataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = sop_instance_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    pydicom.dataset.validate_file_meta(meta)
    return meta


def png_to_dcm(
    png_path: Path,
    study_uid: str,
    series_uid: str,
    instance_number: int,
) -> Path:
    img = Image.open(png_path)

    # Ensure grayscale
    if img.mode != "L":
        img = img.convert("L")

    pixel_array = np.array(img, dtype=np.uint8)
    rows, cols = pixel_array.shape

    sop_instance_uid = generate_uid()

    ds = Dataset()
    ds.file_meta = build_file_meta(sop_instance_uid)
    ds.is_implicit_VR = False
    ds.is_little_endian = True

    # --- Patient ---
    ds.PatientName = png_path.stem.split("_")[0]
    ds.PatientID = png_path.stem.split("_")[0]
    ds.PatientBirthDate = ""
    ds.PatientSex = ""

    # --- Study ---
    ds.StudyInstanceUID = study_uid
    ds.StudyDate = ""
    ds.StudyTime = ""
    ds.ReferringPhysicianName = ""
    ds.StudyID = ""
    ds.AccessionNumber = ""

    # --- Series ---
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = 1
    ds.Modality = "OT"

    # --- Instance ---
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = sop_instance_uid
    ds.InstanceNumber = instance_number
    ds.ConversionType = "WSD"  # Workstation, required for Secondary Capture

    # --- Image pixel ---
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0  # unsigned
    ds.PixelData = pixel_array.tobytes()

    out_path = png_path.with_suffix(".dcm")
    pydicom.dcmwrite(out_path, ds, write_like_original=False)
    return out_path


def main() -> None:
    png_files = sorted(IMAGES_DIR.glob("*.png"))
    if not png_files:
        print(f"No PNG files found in {IMAGES_DIR}")
        sys.exit(1)

    # Group images by filename prefix (part before the first underscore)
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in png_files:
        prefix = p.stem.split("_")[0]
        groups[prefix].append(p)

    total = 0
    for prefix, paths in sorted(groups.items()):
        study_uid = generate_uid()
        series_uid = generate_uid()
        print(f"\nGroup '{prefix}'  study={study_uid}")
        for instance_number, png_path in enumerate(sorted(paths), start=1):
            out = png_to_dcm(png_path, study_uid, series_uid, instance_number)
            print(f"  [{instance_number}/{len(paths)}] {png_path.name} -> {out.name}")
            total += 1

    print(f"\nDone. Converted {total} image(s).")


if __name__ == "__main__":
    main()
