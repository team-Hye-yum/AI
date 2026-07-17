from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "hyeyum-ai"
    app_env: str = "local"
    root_path: str = "/ai"
    upload_dir: str = "uploads"
    max_upload_file_size_mb: int = 100

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
