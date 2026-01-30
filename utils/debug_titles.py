#!/usr/bin/env python3
"""Debug script to compare titles between train/test and phase I."""

import pandas as pd
import re
import os

def extract_title(text):
    match = re.search(r'Title in Investigation:\s*(.+?)(?:\n|Abstract)', str(text))
    if match:
        return match.group(1).strip()
    return None

# Load files
data_dir = '/Users/kayems/Documents/GitHub/llm_systematic_review/data'
train_df = pd.read_csv(os.path.join(data_dir, 'train_data_LFM.csv'))
phase1_df = pd.read_csv(os.path.join(data_dir, 'phase I screening_ALL studies_cleaned_prompts - phase I screening_ALL studies_cleaned_prompts.csv'))

# Get sample titles
train_title = extract_title(train_df.iloc[0, 0])

print("Sample train title:")
print(repr(train_title))
print()

# Look at raw prompt from phase I
print("Raw phase I prompt (first 500 chars):")
print(repr(phase1_df.iloc[0, 0][:500]))
print()

# Check for the title pattern
print("Looking for 'Title in Investigation' in phase I...")
sample = phase1_df.iloc[0, 0]
if 'Title in Investigation' in sample:
    print("Found!")
    idx = sample.find('Title in Investigation')
    print(repr(sample[idx:idx+200]))
else:
    print("Not found - let's see what patterns exist:")
    print(repr(sample[:300]))
