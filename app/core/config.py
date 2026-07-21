from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "hyeyum-ai"
    app_env: str = "local"
    root_path: str = "/ai"
    upload_dir: str = "uploads"
    max_upload_file_size_mb: int = 100
    seed_training_data_dir: str = "uploads"
    weaviate_url: str = "http://localhost:8081"
    weaviate_collection: str = "TrainingDocument"
    weaviate_ingest_enabled: bool = True
    weaviate_request_timeout_seconds: int = 120
    seed_training_data_on_startup: bool = True
    seed_training_data_state_file: str = ".training-data-seed-state.json"
    seed_training_data_max_attempts: int = 60
    seed_training_data_retry_delay_seconds: int = 5
    ai_review_cache_path: str = "temp/ai-review-cache.sqlite3"
    ai_review_cache_ttl_seconds: int = 604800
    openai_api_key: str = ""
    openai_model: str = "gpt-5.6"
    openai_timeout_seconds: float = 20.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
