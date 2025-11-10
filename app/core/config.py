from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr, computed_field
from pathlib import Path

class Settings(BaseSettings):
    """Application configuration based on environment variables."""

    # --- App ---
    APP_ENV: str = Field(default="dev")
    APP_NAME: str = Field(default="reviewer-poc")
    PORT: int = Field(default=8080)

    # --- Database ---
    DB_HOST: str = Field(default="localhost")
    DB_PORT: int = Field(default=5432)
    DB_NAME: str = Field(default="reviewer")
    DB_USER: str = Field(default="postgres")
    DB_PASSWORD: SecretStr = Field(default=SecretStr("postgres"))

    # --- SQLAlchemy behavior ---
    SQL_ECHO: bool = Field(default=False)
    POOL_SIZE: int = Field(default=5)
    POOL_MAX_OVERFLOW: int = Field(default=10)

    # --- Local storage ---
    ARTIFACT_ROOT: str = Field(default="./data/artifacts")

    @computed_field(return_type=str)
    @property
    def sqlalchemy_url(self) -> str:
        """Compose SQLAlchemy connection URL dynamically."""
        password = self.DB_PASSWORD.get_secret_value()
        return (
            f"postgresql+psycopg://{self.DB_USER}:{password}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @computed_field(return_type=str)
    @property
    def artifact_root_path(self) -> str:
        """Resolve artifact storage path to absolute path."""
        return str(Path(self.ARTIFACT_ROOT).resolve())

    class Config:
        env_file = ".env"
        extra = "ignore"


# Global settings instance
settings = Settings()
