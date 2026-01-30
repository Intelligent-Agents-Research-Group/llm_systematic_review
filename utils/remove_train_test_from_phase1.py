#!/usr/bin/env python3
"""
Script to remove train and test examples from the phase I screening file.
Matches examples by extracting the Title from each prompt.
"""

import pandas as pd
import re
import os


def extract_title(text):
    """Extract the title from a prompt text."""
    # Handle both \n (actual newline) and /n (literal string) patterns
    match = re.search(r'Title in Investigation:\s*(.+?)(?:\n|/n|Abstract)', str(text))
    if match:
        return match.group(1).strip()
    return None


def main():
    # Get the data directory path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    
    train_file = os.path.join(data_dir, 'train_data_LFM.csv')
    test_file = os.path.join(data_dir, 'test_data_LFM.csv')
    phase1_file = os.path.join(data_dir, 'phase I screening_ALL studies_cleaned_prompts - phase I screening_ALL studies_cleaned_prompts.csv')
    
    # Load all files
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    phase1_df = pd.read_csv(phase1_file)
    
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Phase I rows before: {len(phase1_df)}")
    print()
    
    # Get titles from train and test
    train_titles = set(train_df.iloc[:, 0].apply(extract_title).dropna())
    test_titles = set(test_df.iloc[:, 0].apply(extract_title).dropna())
    all_titles = train_titles | test_titles
    
    print(f"Unique titles from train: {len(train_titles)}")
    print(f"Unique titles from test: {len(test_titles)}")
    print(f"Combined unique titles: {len(all_titles)}")
    print()
    
    # Extract titles from phase I
    phase1_df['_title'] = phase1_df['prompt'].apply(extract_title)
    print(f"Phase I titles found: {phase1_df['_title'].notna().sum()}")
    
    # Find matching rows
    matches = phase1_df['_title'].isin(all_titles)
    print(f"Matching rows found: {matches.sum()}")
    print()
    
    # Remove matching rows
    phase1_filtered = phase1_df[~matches].copy()
    phase1_filtered = phase1_filtered.drop(columns=['_title'])
    
    print(f"Phase I rows after removal: {len(phase1_filtered)}")
    print(f"Rows removed: {len(phase1_df) - len(phase1_filtered)}")
    
    # Save the filtered file
    phase1_filtered.to_csv(phase1_file, index=False)
    print(f"\nFile saved: {phase1_file}")


if __name__ == "__main__":
    main()
