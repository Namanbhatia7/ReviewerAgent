# app/services/ingest_pdf/anchors.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import yaml

@dataclass
class Heuristics:
    block_scan_window_px: int = 1200
    option_markers: List[str] = None
    option_mark_corridor_px: int = 28
    baseline_tolerance_px: int = 6
    raster_dpi: int = 260

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Heuristics":
        return cls(
            block_scan_window_px=int(d.get("block_scan_window_px", 1200)),
            option_markers=d.get("option_markers", ["•", "-", "[ ]", "[x]", "( )", "(x)"]),
            option_mark_corridor_px=int(d.get("option_mark_corridor_px", 28)),
            baseline_tolerance_px=int(d.get("baseline_tolerance_px", 6)),
            raster_dpi=int(d.get("raster_dpi", 260)),
        )

@dataclass
class ProjectConfig:
    project_id: str
    version: str
    anchors: Dict[str, List[str]]
    heuristics: Heuristics
    questions: List[Dict[str, Any]]

def load_project_config(yaml_text: str) -> ProjectConfig:
    cfg = yaml.safe_load(yaml_text) or {}
    project_id = str(cfg.get("project_id", "unknown"))
    version = str(cfg.get("version", "0"))
    extractors = cfg.get("extractors", {})
    anchors = extractors.get("anchors", {}) or {}
    heur = Heuristics.from_dict(extractors.get("heuristics", {}) or {})
    questions = cfg.get("questions", []) or []
    # normalize each alias to lowercase (matching is case-insensitive)
    norm_anchors = {k: [str(a).strip() for a in v] for k, v in anchors.items()}
    return ProjectConfig(project_id=project_id, version=version, anchors=norm_anchors, heuristics=heur, questions=questions)

def all_aliases_for(anchor_key: str, cfg: ProjectConfig) -> List[str]:
    return [a.lower() for a in cfg.anchors.get(anchor_key, [])]
