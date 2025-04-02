# DSPy ELO Rating System

Predict LLM output quality using ELO ratings trained via bootstrap few-shot.

## ✅ Fully Implemented
- Core ELO rating system with K-factor configuration
- Training pipeline with rating scaling (1-9 → 100-900)
- CLI interface for training
- Demo script with example comparisons
- Integration tests with real data
- DeepSeek model integration

## ⚠️ Partially Implemented
- LLM comparison reliability
  - Basic integration exists but needs more validation
  - Cache disabling implemented but not fully tested
- Bootstrap few-shot training
  - Core functionality works but needs more diverse test cases

## ❌ Not Implemented
- Model persistence (currently only saves training info)
- Comprehensive error handling for API failures
- Advanced rating visualization
- Custom K-factor configuration in CLI

## Test Coverage
```shell
# Run all tests
pytest tests/

# Run specific test suites
pytest tests/test_elo_rating.py -v        # Core rating system
pytest tests/test_llm_comparison.py -v    # DeepSeek integration
pytest tests/test_integration.py -v       # Full training workflow
```

## Test Coverage Status

```text
tests/
├── test_demo.py            ✅ Basic demo output checks
├── test_e2e.py             ✅ End-to-end workflow tests  
├── test_e2e_spec.py        ✅ Spec verification tests
├── test_elo_rating.py      ✅ Core ELO logic tests
├── test_integration.py     ✅ Data integration tests
├── test_llm_comparison.py  ⚠️ Needs less mocking
├── test_package_installation.py ✅ Packaging checks
└── test_training.py        ⚠️ Needs true bootstrap tests

## Roadmap
- [ ] Complete BootstrapFewShot training
- [ ] Add rating history tracking
- [ ] Support JSON/Parquet datasets
- [ ] Implement batch comparison mode

## Installation

```bash
pip install -e .
pip install dspy-ai
```

## Configuration

Set your OpenAI API key (for DeepSeek) in environment variables:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Running the Demo

```bash
python -m dspy_elo.demo
```

This will show:
1. Initial ratings for two modules
2. Rating changes after LLM output comparisons
3. Final ratings

## Running Tests

```bash
pytest tests/
```

The tests verify:
- New modules get default ELO ratings
- Ratings update correctly after comparisons
- LLM output comparison works via DSPy
- Demo script produces expected output

## Training ELO Predictor

### Input Data Format
Create a CSV file with:
- `text`: LLM output text (string)
- `rating`: Quality rating (1-9 scale, integer)

Example `data/ratings.csv`:
```csv
text,rating
"Detailed response with examples",8
"Concise answer",5
"Vague response",2
```

### CLI Usage
```bash
python -m dspy_elo.train data/ratings.csv --output-dir elo_model
```

### Output Files
Trained model directory contains:
- `training_info.txt`: Metadata about training run
- (Future: model weights when implemented)

### Python API
```python
from dspy_elo import train_elo_predictor
import pandas as pd

# Load your data
data = pd.read_csv("data/ratings.csv")

# Train and save model
predictor = train_elo_predictor(data, output_dir="elo_model")

# Compare new texts
prediction = predictor.predict("Full analysis", "Brief summary")
print(f"Preferred output: Text {prediction[0]}")
```
