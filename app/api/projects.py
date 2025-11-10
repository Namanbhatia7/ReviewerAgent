# app/api/projects.py
from fastapi import APIRouter, UploadFile, Form, HTTPException, Depends
from sqlalchemy.orm import Session
from datetime import datetime
import yaml

from app.core.db import get_db
from app.models.orm import Project
from app.models.schemas import ProjectOut

router = APIRouter(prefix="/projects", tags=["projects"])

@router.post("", response_model=ProjectOut)
async def create_project(
    project_id: str = Form(...),
    file: UploadFile = Form(...),
    db: Session = Depends(get_db),
):
    # basic file checks
    fname = (file.filename or "").lower()
    if not (fname.endswith(".yaml") or fname.endswith(".yml")):
        raise HTTPException(status_code=400, detail="Upload a .yaml or .yml file")

    # ensure project_id is unique
    if db.query(Project).filter(Project.project_id == project_id).first():
        raise HTTPException(status_code=400, detail=f"Project '{project_id}' already exists")

    # read & decode yaml
    raw_bytes = await file.read()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="YAML file must be UTF-8 encoded")

    # parse to validate YAML (and optionally sanity-check keys)
    try:
        parsed = yaml.safe_load(text)
        if not isinstance(parsed, dict):
            raise ValueError("YAML root must be a mapping/object")
        # optional: if YAML contains a project_id, ensure it matches the form field
        yaml_pid = parsed.get("project_id")
        if yaml_pid and str(yaml_pid) != project_id:
            raise HTTPException(
                status_code=400,
                detail=f"project_id mismatch: form='{project_id}' vs yaml='{yaml_pid}'",
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")

    # persist
    project = Project(
        project_id=project_id,
        config_yaml=text,
        created_at=datetime.utcnow(),
    )
    db.add(project)
    db.commit()
    db.refresh(project)

    return ProjectOut(
        id=project.id,
        project_id=project.project_id,
        config_yaml=project.config_yaml,
        parsed_config=parsed,
        created_at=project.created_at,
    )

@router.get("/{project_id}", response_model=ProjectOut)
def get_project(project_id: str, db: Session = Depends(get_db)):
    row = db.query(Project).filter(Project.project_id == project_id).first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
    try:
        parsed = yaml.safe_load(row.config_yaml)
    except Exception:
        parsed = {"error": "Invalid YAML - could not parse"}
    return ProjectOut(
        id=row.id,
        project_id=row.project_id,
        config_yaml=row.config_yaml,
        parsed_config=parsed,
        created_at=row.created_at,
    )

@router.put("/{project_id}", response_model=ProjectOut)
async def update_project(
    project_id: str,
    file: UploadFile = Form(...),
    db: Session = Depends(get_db),
):
    row = db.query(Project).filter(Project.project_id == project_id).first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    fname = (file.filename or "").lower()
    if not (fname.endswith(".yaml") or fname.endswith(".yml")):
        raise HTTPException(status_code=400, detail="Upload a .yaml or .yml file")

    raw_bytes = await file.read()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="YAML file must be UTF-8 encoded")

    try:
        parsed = yaml.safe_load(text)
        if not isinstance(parsed, dict):
            raise ValueError("YAML root must be a mapping/object")
        yaml_pid = parsed.get("project_id")
        if yaml_pid and str(yaml_pid) != project_id:
            raise HTTPException(
                status_code=400,
                detail=f"project_id mismatch: path='{project_id}' vs yaml='{yaml_pid}'",
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")

    row.config_yaml = text
    db.commit()
    db.refresh(row)

    return ProjectOut(
        id=row.id,
        project_id=row.project_id,
        config_yaml=row.config_yaml,
        parsed_config=parsed,
        created_at=row.created_at,
    )

@router.get("", response_model=list[ProjectOut])
def list_projects(db: Session = Depends(get_db)):
    rows = db.query(Project).all()
    out = []
    for r in rows:
        try:
            parsed = yaml.safe_load(r.config_yaml)
        except Exception:
            parsed = None
        out.append(
            ProjectOut(
                id=r.id,
                project_id=r.project_id,
                config_yaml=r.config_yaml,
                parsed_config=parsed,
                created_at=r.created_at,
            )
        )
    return out
