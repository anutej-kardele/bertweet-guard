from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_id: str = "anutej9/bertweet-guard"
    environment: str = "development"

    # Current deployed 5-epoch v6 artifact.
    decision_threshold: float = 0.7990

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
