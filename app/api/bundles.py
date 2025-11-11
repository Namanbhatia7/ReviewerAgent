# app/api/bundles.py
from __future__ import annotations
from fastapi import APIRouter, UploadFile, Form, HTTPException, Depends
from sqlalchemy.orm import Session
from datetime import datetime
from pathlib import Path

from app.core.db import get_db
from app.models.orm import Artifact, Bundle, ArtifactKind, Project
from app.models.schemas import BundleOut, ExtractedPayload
from app.services.storage.local_fs import save_bytes
from app.services.ingest_pdf.extractor import preflight_pdf, extract_pdf

router = APIRouter(prefix="/bundles", tags=["bundles"])

_PDF_SIG = b"%PDF"
_ALLOWED_EXT = (".pdf",)

def _ensure_project(db: Session, project_id: str) -> Project:
    proj = db.query(Project).filter(Project.project_id == project_id).first()
    if not proj:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
    return proj

def _read_pdf_bytes(file: UploadFile) -> bytes:
    fname = (file.filename or "").lower()
    if not fname.endswith(_ALLOWED_EXT):
        raise HTTPException(status_code=400, detail="Upload a .pdf file")
    raw = file.file.read()
    if not raw or not raw.startswith(_PDF_SIG):
        raise HTTPException(status_code=400, detail="File is not a valid PDF (missing %PDF header)")
    return raw

@router.post("/pdf", response_model=BundleOut)
async def create_bundle_pdf(
    project_id: str = Form(...),
    file: UploadFile = Form(...),
    db: Session = Depends(get_db),
):
    _ensure_project(db, project_id)
    pdf_bytes = _read_pdf_bytes(file)

    # Preflight quickly (page count + vector text) before persisting
    try:
        pf = preflight_pdf(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {e}")

    # Store to local FS (subdir by project)
    stored = save_bytes(
        pdf_bytes,
        subdir=project_id,
        filename=file.filename or "uploaded.pdf",
        mime_hint=file.content_type or "application/pdf",
    )

    # Idempotency: dedupe by sha within same project
    dup_bundle = (
        db.query(Bundle)
        .filter(Bundle.project_id == project_id, Bundle.file_sha256 == stored.sha256)
        .order_by(Bundle.created_at.desc())
        .first()
    )
    if dup_bundle:
        return BundleOut(
            bundle_id=dup_bundle.id,
            project_id=dup_bundle.project_id,
            artifact_path=stored.path,
            file_sha256=dup_bundle.file_sha256,
            pages=dup_bundle.pages,
            vector_text=dup_bundle.vector_text,
            dpi=dup_bundle.dpi,
            ocr_conf=dup_bundle.ocr_conf,
            created_at=dup_bundle.created_at,
            duplicate=True,
            extracted=ExtractedPayload.model_validate(dup_bundle.extracted_json),
        )

    # Create artifact row
    art = Artifact(
        kind=ArtifactKind.pdf,
        path=stored.path,
        mime=stored.mime,
        bytes=stored.bytes,
        sha256=stored.sha256,
        created_at=datetime.utcnow(),
    )
    db.add(art)
    db.flush()  # get art.id

    # Create bundle (pending)
    pending = ExtractedPayload(status="pending_extraction").model_dump()
    bundle = Bundle(
        project_id=project_id,
        artifact_id=art.id,
        pages=pf.pages,
        vector_text=pf.vector_text,
        dpi=0,
        ocr_conf=0.0,
        file_sha256=stored.sha256,
        extracted_json=pending,
        created_at=datetime.utcnow(),
    )
    db.add(bundle)
    db.commit()
    db.refresh(bundle)

    return BundleOut(
        bundle_id=bundle.id,
        project_id=bundle.project_id,
        artifact_path=stored.path,
        file_sha256=bundle.file_sha256,
        pages=bundle.pages,
        vector_text=bundle.vector_text,
        dpi=bundle.dpi,
        ocr_conf=bundle.ocr_conf,
        created_at=bundle.created_at,
        duplicate=False,
        extracted=ExtractedPayload.model_validate(bundle.extracted_json),
    )

@router.get("/{bundle_id}", response_model=BundleOut)
def get_bundle(bundle_id: int, db: Session = Depends(get_db)):
    row = db.query(Bundle).filter(Bundle.id == bundle_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Bundle not found")

    return BundleOut(
        bundle_id=row.id,
        project_id=row.project_id,
        artifact_path=row.artifact.path,
        file_sha256=row.file_sha256,
        pages=row.pages,
        vector_text=row.vector_text,
        dpi=row.dpi,
        ocr_conf=row.ocr_conf,
        created_at=row.created_at,
        duplicate=False,
        extracted=ExtractedPayload.model_validate(row.extracted_json),
    )

@router.post("/{bundle_id}/extract", response_model=BundleOut)
def run_extraction(bundle_id: int, db: Session = Depends(get_db)):
    bundle = db.query(Bundle).filter(Bundle.id == bundle_id).first()
    if not bundle:
        raise HTTPException(status_code=404, detail="Bundle not found")

    project_id = bundle.project_id
    pdf_path = bundle.artifact.path
    pdf_bytes = Path(pdf_path).read_bytes()

    status, mean_conf, payload = extract_pdf(db, project_id, pdf_bytes)

    bundle.extracted_json = payload
    bundle.ocr_conf = mean_conf
    db.commit()
    db.refresh(bundle)

    return BundleOut(
        bundle_id=bundle.id,
        project_id=bundle.project_id,
        artifact_path=bundle.artifact.path,
        file_sha256=bundle.file_sha256,
        pages=bundle.pages,
        vector_text=False,
        dpi=0,
        ocr_conf=mean_conf,
        created_at=bundle.created_at,
        duplicate=False,
        extracted=payload,
    )
