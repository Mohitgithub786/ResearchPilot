import pytest
import os
from unittest import mock
from app.core.config import validate_config
from app.core.errors import ConfigurationError

def test_validate_config_missing_openai():
    with mock.patch.dict(os.environ, {"EMBEDDING_PROVIDER": "openai", "OPENAI_API_KEY": ""}, clear=True):
        with mock.patch("pathlib.Path.exists", return_value=True):
            with pytest.raises(ConfigurationError, match="OPENAI_API_KEY is required"):
                validate_config()

def test_validate_config_missing_groq():
    with mock.patch.dict(os.environ, {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "", "EMBEDDING_PROVIDER": "other", "OPENAI_API_KEY": ""}, clear=True):
        with mock.patch("pathlib.Path.exists", return_value=True):
            with pytest.raises(ConfigurationError, match="GROQ_API_KEY is required"):
                validate_config()
