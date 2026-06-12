import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    chat_model: str
    embedding_model: str
    temperature: float
    retrieval_k: int
    chunk_size: int
    chunk_overlap: int

    @property
    def has_api_key(self) -> bool:
        return bool(self.openai_api_key)


def _parse_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _parse_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        temperature=_parse_float("OPENAI_TEMPERATURE", 0.0),
        retrieval_k=_parse_int("RETRIEVAL_K", 4),
        chunk_size=_parse_int("CHUNK_SIZE", 1000),
        chunk_overlap=_parse_int("CHUNK_OVERLAP", 200),
    )


def validate_extension(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in SUPPORTED_EXTENSIONS
