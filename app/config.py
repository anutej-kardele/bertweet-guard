from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_id: str = "anutej9/bertweet-guard"
    environment: str = "development"
    decision_threshold: float = 0.6760

    # Explicitly configure Pydantic to load variables from the .env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

# Instantiate a single global settings object to import across the app
settings = Settings()