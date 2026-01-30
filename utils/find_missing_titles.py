#!/usr/bin/env python3
"""Find entries without titles and try to match by abstract."""

import pandas as pd
import re
import os


def extract_title(text):
    match = re.search(r'Title in Investigation:\s*(.+?)(?:\n|/n|Abstract)', str(text))
    if match:
        return match.group(1).strip()
    return None


def extract_abstract_start(text):
    match = re.search(r'Abstract in Investigation:\s*(.+)', str(text), re.DOTALL)
    if match:
        abstract = match.group(1).strip()
        # Get first 100 chars
        return abstract[:100]
    return None


def main():
    data_dir = '/Users/kayems/Documents/GitHub/llm_systematic_review/data'
    train_df = pd.read_csv(os.path.join(data_dir, 'train_data_LFM.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_data_LFM.csv'))
    phase1_df = pd.read_csv(os.path.join(data_dir, 'phase I screening_ALL studies_cleaned_prompts - phase I screening_ALL studies_cleaned_prompts.csv'))

    # Find rows without titles in train
    print('=== TRAIN: Rows without titles ===')
    train_missing = []
    for i, row in train_df.iterrows():
        title = extract_title(row.iloc[0])
        if not title:
            abstract_start = extract_abstract_start(row.iloc[0])
            print(f'Row {i}: No title')
            print(f'  Abstract start: {abstract_start}')
            train_missing.append((i, abstract_start))
            print()

    print('=== TEST: Rows without titles ===')
    test_missing = []
    for i, row in test_df.iterrows():
        title = extract_title(row.iloc[0])
        if not title:
            abstract_start = extract_abstract_start(row.iloc[0])
            print(f'Row {i}: No title')
            print(f'  Abstract start: {abstract_start}')
            test_missing.append((i, abstract_start))
            print()

    # If there are missing titles, try to find matches in phase1 by abstract
    all_missing = train_missing + test_missing
    if all_missing:
        print(f'\n=== Searching Phase I for {len(all_missing)} entries by abstract ===')
        phase1_abstracts = phase1_df['prompt'].apply(extract_abstract_start)
        
        for idx, abstract_start in all_missing:
            if abstract_start:
                # Search for matching abstract in phase1
                matches = phase1_df[phase1_abstracts.str.contains(abstract_start[:50], na=False, regex=False)]
                print(f'Train/Test row {idx}: Found {len(matches)} potential matches in Phase I')
    else:
        print('\nNo missing titles found!')


if __name__ == "__main__":
    main()
