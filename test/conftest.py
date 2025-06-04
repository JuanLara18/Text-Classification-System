# tests/conftest.py
# Pytest configuration and fixtures

import pytest
import os
import sys
import tempfile
import shutil
from unittest.mock import Mock

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="ai_classification_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def mock_config():
    """Create a mock configuration object."""
    config = Mock()
    config.get_preprocessing_options.return_value = {
        'lowercase': True,
        'remove_punctuation': True,
        'remove_stopwords': True,
        'min_word_length': 2
    }
    config.get_feature_extraction_config.return_value = {
        'method': 'tfidf',
        'tfidf': {'max_features': 100, 'ngram_range': [1, 2]}
    }
    config.get_logging_config.return_value = {
        'level': 'INFO',
        'console_output': True
    }
    config.get_spark_config.return_value = {
        'executor_memory': '1g',
        'driver_memory': '1g'
    }
    return config

@pytest.fixture
def mock_logger():
    """Create a mock logger object."""
    logger = Mock()
    return logger

@pytest.fixture
def sample_dataframe():
    """Create a sample pandas DataFrame for testing."""
    import pandas as pd
    return pd.DataFrame({
        'position_title': ['Software Engineer', 'Data Scientist', 'Product Manager'],
        'job_description': ['Python programming', 'Machine learning', 'Product strategy'],
        'category': ['Tech', 'Tech', 'Business']
    })

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment."""
    # Create necessary directories
    os.makedirs('tests/data', exist_ok=True)
    os.makedirs('tests/output', exist_ok=True)
    os.makedirs('tests/results', exist_ok=True)
    os.makedirs('tests/cache', exist_ok=True)
    os.makedirs('tests/logs', exist_ok=True)

@pytest.fixture
def openai_mock():
    """Mock OpenAI API responses."""
    import openai
    from unittest.mock import patch, Mock
    
    with patch('openai.chat.completions.create') as mock_create:
        mock_response = Mock()
        mock_response.choices[0].message.content = "Tech"
        mock_response.usage.completion_tokens = 5
        mock_create.return_value = mock_response
        yield mock_create