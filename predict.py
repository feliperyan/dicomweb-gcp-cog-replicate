import os
os.environ["TORCH_HOME"] = "."

import numpy as np
import torch
from cog import BasePredictor, Input
from pathlib import Path
from PIL import Image
from torchvision import models
import google.auth.transport.requests
import google.oauth2.service_account
from dicomweb_client import DICOMwebClient


WEIGHTS = models.ResNet50_Weights.IMAGENET1K_V1

DICOM_STORE_URL = (
    "https://healthcare.googleapis.com/v1"
    "/projects/cloudflare-fryan-01"
    "/locations/us-central1"
    "/datasets/fryan-healthdataset-01"
    "/dicomStores/fryan-nih-dicom"
    "/dicomWeb"
)
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
SERVICE_ACCOUNT_KEY = Path(__file__).parent / "gcp_key.json"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model and initialise the DICOM store client."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(weights=WEIGHTS).to(self.device)
        self.model.eval()

        credentials = google.oauth2.service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_KEY, scopes=SCOPES
        )
        session = google.auth.transport.requests.AuthorizedSession(credentials)
        self.dicom_client = DICOMwebClient(url=DICOM_STORE_URL, session=session)

    def predict(
        self,
        study_uid: str = Input(description="DICOM StudyInstanceUID"),
        series_uid: str = Input(description="DICOM SeriesInstanceUID"),
        sop_instance_uid: str = Input(description="DICOM SOPInstanceUID"),
    ) -> dict:
        """Fetch a single DICOM instance from the store and run classification."""
        # 1. Retrieve the DICOM instance via WADO-RS
        ds = self.dicom_client.retrieve_instance(
            study_instance_uid=study_uid,
            series_instance_uid=series_uid,
            sop_instance_uid=sop_instance_uid,
        )

        # 2. Extract pixel data and normalise to 8-bit
        pixel_array = ds.pixel_array
        if pixel_array.dtype != np.uint8:
            pmin, pmax = pixel_array.min(), pixel_array.max()
            if pmax > pmin:
                pixel_array = ((pixel_array - pmin) / (pmax - pmin) * 255).astype(np.uint8)
            else:
                pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)

        # 3. Convert to PIL RGB (ResNet expects 3-channel input)
        img = Image.fromarray(pixel_array).convert("RGB")

        # 4. Run classification
        preds = self.model(WEIGHTS.transforms()(img).unsqueeze(0).to(self.device))
        top3 = preds[0].softmax(0).topk(3)
        categories = WEIGHTS.meta["categories"]
        return {categories[i]: p.detach().item() for p, i in zip(*top3)}
