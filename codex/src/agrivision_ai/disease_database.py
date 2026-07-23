"""Disease knowledge base utilities for AgriVision AI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DiseaseInfo:
    plant_name: str
    disease_name: str
    short_description: str
    cause: list[str]
    symptoms: list[str]
    treatment: list[str]
    prevention: list[str]


class DiseaseDatabase:

    def __init__(self, database_path: Path):

        self.database_path = database_path
        self._entries = self._load_entries(database_path)

        print(f"Loaded {len(self._entries)} disease entries.")

    def find(self, plant_name: str, disease_name: str) -> DiseaseInfo | None:

        plant_key = normalize_key(plant_name)
        disease_key = normalize_key(disease_name)

        for entry in self._entries:

            if (
                normalize_key(entry.plant_name) == plant_key
                and
                normalize_key(entry.disease_name) == disease_key
            ):
                return entry

        return None

    def find_by_disease(self, disease_name: str) -> DiseaseInfo | None:

        disease_key = normalize_key(disease_name)

        for entry in self._entries:

            if normalize_key(entry.disease_name) == disease_key:
                return entry

        return None

    def all_entries(self):

        return list(self._entries)

    @staticmethod
    def _load_entries(database_path: Path):

        if not database_path.exists():
            raise FileNotFoundError(
                f"Disease database not found: {database_path}"
            )

        with open(database_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        entries = []

        for raw in payload["diseases"]:

            cause = raw.get("cause", [])

            if isinstance(cause, str):
                cause = [cause]

            entries.append(
                DiseaseInfo(
                    plant_name=raw["plant_name"],
                    disease_name=raw["disease_name"],
                    short_description=raw["short_description"],
                    cause=cause,
                    symptoms=raw.get("symptoms", []),
                    treatment=raw.get("treatment", []),
                    prevention=raw.get("prevention", []),
                )
            )

        return entries


def normalize_key(text: str) -> str:

    text = text.lower()

    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.replace(",", "")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace("/", " ")

    text = " ".join(text.split())

    return text