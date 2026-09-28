from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_id: str = "anutej9/bertweet-guard"
    environment: str = "development"

    # In Docker/Cloud Run this points to the model snapshot baked into the image.
    # During local development it can remain None and the model is downloaded
    # from Hugging Face.
    model_dir: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",

        # Important:
        # Ignore old environment variables such as DECISION_THRESHOLD.
        extra="ignore",
    )


settings = Settings()