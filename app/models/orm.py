from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, Float, JSON, ForeignKey, Boolean, DateTime, Text, Enum, Index, func
import enum

class Base(DeclarativeBase):
    pass

class ArtifactKind(str, enum.Enum):
    pdf = "pdf"
    image = "image"


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    # anchors/rules/weights (YAML or JSON as text)
    config_yaml: Mapped[Text]
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())

class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[ArtifactKind] = mapped_column(Enum(ArtifactKind), index=True)
    path: Mapped[str] = mapped_column(String(512))         # local filesystem path (POC)
    mime: Mapped[str] = mapped_column(String(128))
    bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(80), index=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())

class Bundle(Base):
    """
    One uploaded PDF (printed web page) → one Bundle.
    extracted_json will hold prompt/questions/options/selected/rating/explanation + bboxes.
    """
    __tablename__ = "bundles"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[str] = mapped_column(String(64), index=True)
    artifact_id: Mapped[int] = mapped_column(ForeignKey("artifacts.id"))
    pages: Mapped[int] = mapped_column(Integer)
    vector_text: Mapped[bool] = mapped_column(Boolean)     # True if PDF has selectable text
    dpi: Mapped[int] = mapped_column(Integer)              # for raster path
    ocr_conf: Mapped[float] = mapped_column(Float)         # mean OCR conf for raster PDFs
    file_sha256: Mapped[str] = mapped_column(String(80), index=True)
    extracted_json: Mapped[dict] = mapped_column(JSON)     # ReviewBundle.extracted
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())

    artifact: Mapped["Artifact"] = relationship("Artifact")

    __table_args__ = (Index("ix_bundle_project_created", "project_id", "created_at"),)

class Rubric(Base):
    __tablename__ = "rubrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[str] = mapped_column(String(32))
    rubric_json: Mapped[dict] = mapped_column(JSON)        # compiled rubric (dimensions, weights, rules)
    compiled_from_docs: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())

class Review(Base):
    __tablename__ = "reviews"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[str] = mapped_column(String(64), index=True)
    bundle_id: Mapped[int] = mapped_column(ForeignKey("bundles.id"))
    rubric_id: Mapped[int] = mapped_column(ForeignKey("rubrics.id"))
    review_json: Mapped[dict] = mapped_column(JSON)        # scores/findings/citations/fixes/uncertainty
    weighted_total: Mapped[float] = mapped_column(Float)
    passed: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())

    bundle: Mapped["Bundle"] = relationship("Bundle")
    rubric: Mapped["Rubric"] = relationship("Rubric")
