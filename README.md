# DSPy ELO Rating System

A simple implementation of the ELO rating system for comparing LLM outputs using DSPy and DeepSeek.

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

To train on your own dataset:
```python
from dspy_elo.training import train_elo_predictor
import pandas as pd

data = pd.DataFrame({
    "text": ["sample1", "sample2"],
    "rating": [5, 8]  # Ratings 1-9
})
predictor = train_elo_predictor(data)
