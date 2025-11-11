# app/models/schemas.py
from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional, Tuple, List, Literal, Dict, Any

BBox = Tuple[int, float, float, float, float]  # (page, x, y, w, h)

class ProjectCreate(BaseModel):
    project_id: str = Field(..., description="Unique identifier for the project (e.g., webapp_voting_v1)")
    config_yaml: str = Field(..., description="YAML string containing project configuration")

class ProjectOut(BaseModel):
    id: int
    project_id: str
    config_yaml: str
    parsed_config: Optional[dict] = None
    created_at: datetime

class TextField(BaseModel):
    text: str = ""
    bbox: Optional[BBox] = None

class Option(BaseModel):
    text: str
    bbox: Optional[BBox] = None
    selected: bool = False
    selection_confidence: float = 0.0

    @field_validator("selection_confidence")
    @classmethod
    def _conf_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("selection_confidence must be in [0.0, 1.0]")
        return v

class Question(BaseModel):
    id: Optional[str] = None
    text: str = ""
    bbox: Optional[BBox] = None
    options: List[Option] = []
    per_side: Optional[bool] = None

class Rating(BaseModel):
    value: Optional[float] = None
    label: Optional[str] = None
    bbox: Optional[BBox] = None

class ExtractedPayload(BaseModel):
    status: Literal["pending_extraction", "extracted_minimal", "extracted", "failed"] = "pending_extraction"
    prompt: Optional[TextField] = None
    questions: List[Question] = []
    rating: Optional[Rating] = None
    explanation: Optional[TextField] = None
    page_instructions: Optional[TextField] = None
    notes: Dict[str, Any] = Field(default_factory=dict)  # misc: warnings, metrics, etc.

class BundleOut(BaseModel):
    bundle_id: int
    project_id: str
    artifact_path: str
    file_sha256: str
    pages: int
    vector_text: bool
    dpi: int
    ocr_conf: float
    created_at: datetime
    duplicate: bool = False
    extracted: ExtractedPayload
