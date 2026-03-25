# Changelog

All notable changes to classifai are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- `config.example.yaml` with a customer support ticket classification scenario
- `LICENSE` (MIT), `CONTRIBUTING.md`, `CHANGELOG.md`, `pyproject.toml`
- GitHub issue and PR templates
- Real-world use case mention (HBS/BMW manufacturing error analysis)

### Changed
- Renamed project to **classifai**
- CSV promoted as primary input format (Stata still supported)
- README rewritten to reflect new name and scope

### Removed
- Deprecated project-specific config files (`confi.yaml`, `confi2.yaml`, `confi3.yaml`)
- Standalone utility scripts (`extract_names.py`, `nltk_download.py`)
- Duplicate Streamlit app (`tools/app-local.py`)

---

## [0.1.0] — 2024 (initial release as Text-Classification-System)

### Added
- Dual-path pipeline: OpenAI GPT classification + traditional clustering (K-Means, HDBSCAN, Agglomerative)
- Unique value deduplication to minimize API calls
- YAML-based configuration with multi-perspective support
- HTML evaluation reports with Plotly visualizations
- Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics
- UMAP dimensionality reduction for embedding visualization
- Checkpoint system for resumable processing
- Streamlit web UI for configuration generation
- PySpark integration for large-scale distributed processing
- Cost management and rate limiting for OpenAI API calls
