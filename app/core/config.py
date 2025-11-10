from networkx import fiedler_vector
from pydantic import BaseSettings, Field, SecretStr, field_validator
from pathlib import Path

class Settings(BaseSettings):
    # App info
    APP_ENV: str = Field(default="dev")
    APP_NAME: str = Field(default="reviewer-poc")
    PORT: int = Field(default=8080)

    # Database components
    DB_HOST: str = Field(default="localhost")
    DB_PORT: int = Field(default=5432)
    DB_NAME: str = Field(default="reviewer")
    DB_USER: str = Field(default="postgres")
    DB_PASSWORD: SecretStr = Field(default=SecretStr("postgres"))

    # SQLAlchemy behavior
    SQL_ECHO: bool = Field(default=False)
    POOL_SIZE: int = Field(default=5)
    POOL_MAX_OVERFLOW: int = Field(default=10)

    # File storage (POC)
    ARTIFACT_ROOT: str = Field(default="./data/artifacts")

    @property
    def sqlalchemy_url(self) -> str:
        """Compose SQLAlchemy URL dynamically."""
        pwd = self.DB_PASSWORD.get_secret_value()
        return (
            f"postgresql+psycopg://{self.DB_USER}:{pwd}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @field_validator("ARTIFACT_ROOT", pre=True)
    def resolve_artifact_root(cls, v: str) -> str:
        return str(Path(v).resolve())

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global singleton
settings = Settings()
