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
