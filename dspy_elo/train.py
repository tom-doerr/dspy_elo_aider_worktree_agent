"""Command-line interface for training ELO predictor"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from dspy_elo import train_elo_predictor

def main():
    parser = argparse.ArgumentParser(description='Train ELO predictor model.')
    parser.add_argument('input_csv', help='Path to CSV file with training data')
    parser.add_argument('--output-dir', default='elo_model', 
                      help='Output directory for trained model (default: elo_model)')
    args = parser.parse_args()
    
    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    
    try:
        # Load and validate data
        df = pd.read_csv(input_path)
        
        # Train model (assignment not needed as we just need side effects)
        train_elo_predictor(df, output_dir)
        
        print(f"Successfully trained on {len(df)} examples")
        print(f"Model files saved to: {output_dir.resolve()}")
        
    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
