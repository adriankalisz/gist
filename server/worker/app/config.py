from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # You can change this to any other model you want to use
    model_name : str = "sshleifer/distilbart-cnn-12-6"
    num_beams : int = 2  # Default number of beams for summarization
    device : str = "auto"  # "auto" picks cuda when available, otherwise cpu; or set explicitly, e.g. "cpu", "cuda"

settings = Settings()