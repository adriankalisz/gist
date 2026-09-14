from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # You can change this to any other model you want to use
    model_name : str = "sshleifer/distilbart-cnn-12-6"
    num_beams : int = 2  # Default number of beams for summarization
    summarization_batch_size : int = 8  # Default batch size for summarization
    batch_wait_ms : int = 50  # Max time to keep collecting a batch after the first item arrives, in ms

settings = Settings()