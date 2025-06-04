# AI Text Classification System - Test Suite

## Overview

This comprehensive test suite validates all components of the AI Text Classification System, from individual modules to full pipeline integration.

## Quick Start

```bash
# Setup test environment
cd tests/
python setup_test_env.py

# Generate test data
python test_data_generator.py

# Run all tests
python run_all_tests.py

# Quick test (skip slow tests)
python run_all_tests.py --quick
```

## Test Structure

```
tests/
├── run_all_tests.py           # Main test runner
├── test_data_generator.py     # Generate realistic test data
├── setup_test_env.py          # Environment setup
├── requirements_test.txt      # Test dependencies
├── pytest.ini                 # Pytest configuration
├── conftest.py                # Pytest fixtures
├── 
├── test_config.py             # Configuration system tests
├── test_data_processor.py     # Data processing tests
├── test_ai_classifier.py      # AI classification tests
├── test_clustering.py         # Traditional clustering tests
├── test_evaluation.py         # Evaluation & visualization tests
├── test_pipeline.py           # Integration tests
├── test_utilities.py          # Utilities tests
├── 
├── configs/                   # Test configuration files
│   ├── test_config_ai.yaml
│   ├── test_config_clustering.yaml
│   └── test_config_hybrid.yaml
├── 
├── data/                      # Test data files (auto-generated)
├── output/                    # Test outputs
├── results/                   # Test results and reports
├── cache/                     # Test cache files
└── logs/                      # Test log files
```

## Test Categories

### 1. Unit Tests
- **Configuration Tests**: YAML loading, validation, CLI integration
- **Data Processing Tests**: Text preprocessing, feature extraction, caching  
- **AI Classification Tests**: OpenAI integration, unique value processing, cost tracking
- **Clustering Tests**: K-means, HDBSCAN, Agglomerative clustering algorithms
- **Evaluation Tests**: Metrics calculation, visualization generation, reporting
- **Utilities Tests**: Logging, Spark management, file operations, performance monitoring

### 2. Integration Tests
- **Pipeline Tests**: Full end-to-end workflows
- **Multi-perspective Tests**: Combined AI + clustering workflows
- **Error Handling Tests**: Invalid configs, missing files, API failures

## Running Tests

### All Tests
```bash
python run_all_tests.py
```

### Quick Tests (Skip Slow Tests)
```bash
python run_all_tests.py --quick
```

### Verbose Output
```bash
python run_all_tests.py --verbose
```

### Specific Test Categories
```bash
# Using pytest directly
pytest test_config.py -v
pytest test_ai_classifier.py -v
pytest -m "not slow" -v  # Skip slow tests
```

### With Coverage
```bash
pytest --cov=modules --cov-report=html
```

## Test Data

The test suite uses realistic job posting data generated automatically:

- **Size**: 1,000 job postings
- **Categories**: 8 job categories (Software Engineering, Data Science, etc.)
- **Attributes**: Position titles, descriptions, skills, locations, etc.
- **Format**: Stata (.dta) files for realistic testing

## Dependencies

### Required
- pytest>=7.0.0
- pandas>=1.5.0
- numpy>=1.21.0
- scikit-learn>=1.1.0
- PyYAML>=6.0

### Optional
- matplotlib (for visualization tests)
- hdbscan (for HDBSCAN clustering tests)
- umap (for dimensionality reduction tests)
- openai (for AI classification tests - mocked)

## Environment Variables

For AI classification tests:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Note: Most AI tests use mocked responses, so an actual API key is not required.

## Test Performance

Typical execution times (on modern hardware):

- **Unit Tests**: ~30-60 seconds
- **Integration Tests**: ~2-5 minutes  
- **All Tests**: ~3-8 minutes
- **Quick Mode**: ~1-2 minutes

## Continuous Integration

The test suite is designed for CI/CD integration:

```yaml
# Example GitHub Actions
- name: Run Tests
  run: |
    cd tests/
    python setup_test_env.py
    python test_data_generator.py
    python run_all_tests.py --save-report
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure you're in the tests directory
   cd tests/
   # Check Python path
   python -c "import sys; print(sys.path)"
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements_test.txt
   ```

3. **Spark Issues**
   ```bash
   # Tests mock Spark by default, but ensure Java is available
   java -version
   ```

4. **Test Data Missing**
   ```bash
   python test_data_generator.py
   ```

### Debug Mode

For debugging specific tests:
```bash
python -m pytest test_config.py::TestConfigManager::test_yaml_config_loading -v -s
```

## Test Coverage

The test suite aims for >80% code coverage across all modules:

- Configuration system: >90%
- Data processing: >85%
- AI classification: >80%
- Traditional clustering: >85%
- Evaluation: >80%
- Utilities: >90%

## Contributing

When adding new features:

1. Write corresponding tests
2. Ensure >80% coverage for new code
3. Update test documentation
4. Run full test suite before committing

## Performance Benchmarks

The test suite also serves as a performance benchmark:

- Data processing speed
- Memory usage patterns
- API efficiency metrics
- Clustering algorithm performance

Results are reported in test output and can be tracked over time.