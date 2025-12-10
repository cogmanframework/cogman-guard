# Project Structure

```
cogman_tools/
├── src/
│   └── cogman_tools/          # Main package
│       ├── __init__.py         # Package initialization
│       ├── embedding_inspector.py      # Embedding Quality Inspector
│       ├── behavioral_analyzer.py      # Baseline Behavioral Analyzer
│       ├── eimas_analyzer.py           # EIMAS Analyzer
│       ├── embedding_analyzer.py       # Circuit-based analyzer (legacy)
│       └── text_signal_physics.py       # Signal physics utilities
│
├── tests/                      # Test suite
│   ├── quick_test.py           # Quick test for Embedding Quality Inspector
│   ├── quick_test_behavioral.py # Quick test for Behavioral Analyzer
│   ├── test_behavioral_analyzer.py  # Full test suite for Behavioral Analyzer
│   ├── test_eimas_analyzer.py       # Full test suite for EIMAS Analyzer
│   └── run_all_tests.py       # Run all tests in chronological order
│
├── docs/                       # Documentation
│   ├── BASELINE_BEHAVIORAL_ANALYSIS_SPEC.md
│   └── EIMAS_MAPPING.md
│
├── examples/                   # Example scripts (optional)
│
├── outputs/                    # Generated outputs
│   ├── reports/                # Analysis reports
│   └── visualizations/         # Visualization files (HTML, images)
│
├── README.md                   # Main documentation
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── .gitignore                  # Git ignore rules
└── PROJECT_STRUCTURE.md        # This file
```

## Directory Descriptions

### `src/cogman_tools/`
Main package containing all core analyzers:
- **embedding_inspector.py**: Embedding Quality Inspector (formerly Embedding Physics Inspector)
- **behavioral_analyzer.py**: Baseline Behavioral Analyzer
- **eimas_analyzer.py**: EIMAS Analyzer (comprehensive system)
- **embedding_analyzer.py**: Circuit-based analyzer (legacy/reference implementation)
- **text_signal_physics.py**: Signal processing utilities

### `tests/`
Test suite organized by component:
- Quick tests for fast validation
- Full test suites for comprehensive testing
- `run_all_tests.py` runs all tests in chronological order

### `docs/`
Documentation and specifications:
- Baseline Behavioral Analysis Specification
- EIMAS Mapping documentation

### `outputs/`
Generated files (gitignored):
- **reports/**: Analysis reports (.txt)
- **visualizations/**: Interactive visualizations (.html)

## Installation

```bash
# Install in development mode
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

## Usage

```python
# Import from package
from cogman_tools import EmbeddingQualityInspector, BehavioralAnalyzer, EIMASAnalyzer

# Or import directly
from cogman_tools.embedding_inspector import EmbeddingQualityInspector
```

## Running Tests

```bash
# Run all tests
python tests/run_all_tests.py

# Run specific test suite
python tests/test_behavioral_analyzer.py
python tests/test_eimas_analyzer.py
```

