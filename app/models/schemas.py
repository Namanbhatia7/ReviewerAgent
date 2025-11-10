# app/models/schemas.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class ProjectCreate(BaseModel):
    project_id: str = Field(..., description="Unique identifier for the project (e.g., webapp_voting_v1)")
    config_yaml: str = Field(..., description="YAML string containing project configuration")

class ProjectOut(BaseModel):
    id: int
    project_id: str
    config_yaml: str
    parsed_config: Optional[dict] = None
    created_at: datetime
