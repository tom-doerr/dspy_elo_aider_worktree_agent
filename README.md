# DSPy ELO Rating System

A simple implementation of the ELO rating system for comparing modules.

## Installation

```bash
pip install -e .
```

## Running the Demo

```bash
python -m dspy_elo.demo
```

This will show:
1. Initial ratings for two modules
2. Rating changes after comparisons
3. Final ratings

## Running Tests

```bash
pytest tests/
```

The tests verify:
- New modules get default ELO ratings
- Ratings update correctly after comparisons
- Demo script produces expected output
