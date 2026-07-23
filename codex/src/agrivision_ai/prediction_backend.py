"""Backend-only prediction utilities for AgriVision AI.

This module prepares the future workflow:

Upload Leaf Image -> Identify Plant Species -> Determine Healthy or Diseased
-> Identify Disease -> Return Confidence -> Display Information/Treatment.

No GUI is created here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

try:
    from .disease_database import DiseaseDatabase
except ImportError:
    from disease_database import DiseaseDatabase

print("USING PREDICTION_BACKEND FROM:", __file__)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionResult:
    """Structured result returned by the prediction backend."""

    plant_species: str
    health_status: str
    disease_name: str
    confidence_percentage: float
    disease_information: dict[str, Any] | None


class AgriVisionPredictor:
    """Load trained assets and predict plant health from one image."""

    def __init__(
        self,
        model_path: Path = Path("models/best_model.keras"),
        dataset_config_path: Path = Path("config/dataset_config.json"),
        disease_database_path: Path = Path("config/disease_database.json"),
    ) -> None:
        """Create a predictor without loading TensorFlow until needed."""

        self.model_path = model_path
        self.dataset_config_path = dataset_config_path
        self.disease_database = DiseaseDatabase(disease_database_path)
        print("DATABASE PATH:", disease_database_path.resolve())
        self.image_size, self.class_names = self._load_dataset_settings(dataset_config_path)
        self._model = None

    def predict(self, image_path: Path) -> PredictionResult:

        model = self._load_model()

        image_batch = self._prepare_image(image_path)

        prediction = model.predict(image_batch, verbose=0)

        predicted_index = int(np.argmax(prediction))

        class_name = self.class_names[predicted_index]

        confidence = float(prediction[0][predicted_index]) * 100

        parts = class_name.split("/")

        plant_species = parts[0].strip().lower()

        if len(parts) >= 2:
            disease_name = "/".join(parts[1:])
        else:
            disease_name = class_name

        disease_name = (
            disease_name.lower()
            .replace("_", " ")
            .replace("/", " ")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "")
        )

        disease_name = " ".join(disease_name.split())

        if disease_name == "healthy":

            health_status = "healthy"

            disease_name = "none"

            disease_info = None

        else:

            health_status = "diseased"

            disease_info = self.disease_database.find(
                plant_species,
                disease_name
            )

            if disease_info is None:
                disease_info = self.disease_database.find_by_disease(
                    disease_name
                )

            if disease_info is None:

                print("DATABASE LOOKUP FAILED")
                print("Plant :", plant_species)
                print("Disease:", disease_name)

                disease_info = {
                    "plant_name": plant_species.title(),
                    "disease_name": disease_name.title(),
                    "short_description": "Information not available.",
                    "cause": [],
                    "symptoms": [],
                    "treatment": [],
                    "prevention": [],
                }

           

            if disease_info is None:
                disease_info = self.disease_database.find_by_disease(
                disease_name
                )

            if disease_info is None:

                print("DATABASE LOOKUP FAILED")
                print("Plant :", plant_species)
                print("Disease:", disease_name)

                disease_info = {
                    "plant_name": plant_species.title(),
                    "disease_name": disease_name.title(),
                    "short_description": "Information not available.",
                    "cause": [],
                    "symptoms": [],
                    "treatment": [],
                    "prevention": [],
                }

        return PredictionResult(
            plant_species=plant_species,
            health_status=health_status,
            disease_name=disease_name,
            confidence_percentage=round(confidence, 2),
            disease_information=(
                asdict(disease_info)
                if disease_info is not None and hasattr(disease_info, "__dataclass_fields__")
                else disease_info
            ),
        )

    def _load_model(self):
        """Load the Keras model lazily so importing this module is lightweight."""

        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Trained model not found: {self.model_path}")

            import tensorflow as tf

            self._model = tf.keras.models.load_model(self.model_path)
            LOGGER.info("Loaded model from %s", self.model_path)
        return self._model

    def _prepare_image(self, image_path: Path) -> np.ndarray:
        """Validate and convert one image into a model-ready batch."""

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB").resize(self.image_size)
                image_array = np.asarray(rgb_image, dtype=np.float32)
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise ValueError(f"Invalid image file: {image_path}") from error

        image_batch = np.expand_dims(image_array.astype(np.float32), axis=0)

# TEMP TEST
# DO NOT preprocess
# image_batch = tf.keras.applications.mobilenet_v2.preprocess_input(image_batch)
     
        np.save("debug_streamlit.npy", image_batch)

        print("Saved debug image.")

        return image_batch

    @staticmethod
    def _load_dataset_settings(dataset_config_path: Path) -> tuple[tuple[int, int], list[str]]:
        """Read image size and class names from dataset configuration."""

        if not dataset_config_path.exists():
            raise FileNotFoundError(f"Dataset config not found: {dataset_config_path}")

        with dataset_config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)

        image_size = tuple(config["image_size"])
        return (int(image_size[0]), int(image_size[1])), list(config["classes"])