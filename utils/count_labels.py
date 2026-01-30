#!/usr/bin/env python3
"""
Utility script to count excluded (0) and included (1) reviews
in the train and test data files.
"""

import re
import os
import pandas as pd


def count_labels(file_path: str) -> dict:
    """
    Read a CSV file with pandas and count the number of excluded (0) and included (1) labels.
    
    Labels are found in the text column in the format:
    <|im_start|>assistant
    0<|im_end|>
    or
    <|im_start|>assistant
    1<|im_end|>
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Dictionary with counts for '0' (excluded) and '1' (included)
    """
    counts = {'0': 0, '1': 0}
    
    # Read CSV with pandas
    df = pd.read_csv(file_path)
    
    # Pattern to match the label format
    pattern = r'<\|im_start\|>assistant\s*\n\s*([01])\s*<\|im_end\|>'
    
    # Get the text column (first column)
    text_col = df.columns[0]
    
    for text in df[text_col]:
        match = re.search(pattern, str(text))
        if match:
            label = match.group(1)
            counts[label] += 1
    
    return counts, len(df)


def main():
    # Get the data directory path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    
    train_file = os.path.join(data_dir, 'train_data_LFM.csv')
    test_file = os.path.join(data_dir, 'test_data_LFM.csv')
    
    print("=" * 50)
    print("Label Counts for Systematic Review Data")
    print("=" * 50)
    
    # Count labels in train file
    if os.path.exists(train_file):
        train_counts, train_rows = count_labels(train_file)
        total_train = train_counts['0'] + train_counts['1']
        print(f"\nTrain Data ({os.path.basename(train_file)}):")
        print(f"  Total Rows:   {train_rows}")
        print(f"  Excluded (0): {train_counts['0']}")
        print(f"  Included (1): {train_counts['1']}")
        print(f"  Labels Found: {total_train}")
        if total_train > 0:
            print(f"  Exclusion Rate: {train_counts['0'] / total_train * 100:.2f}%")
            print(f"  Inclusion Rate: {train_counts['1'] / total_train * 100:.2f}%")
    else:
        print(f"\nTrain file not found: {train_file}")
    
    # Count labels in test file
    if os.path.exists(test_file):
        test_counts, test_rows = count_labels(test_file)
        total_test = test_counts['0'] + test_counts['1']
        print(f"\nTest Data ({os.path.basename(test_file)}):")
        print(f"  Total Rows:   {test_rows}")
        print(f"  Excluded (0): {test_counts['0']}")
        print(f"  Included (1): {test_counts['1']}")
        print(f"  Labels Found: {total_test}")
        if total_test > 0:
            print(f"  Exclusion Rate: {test_counts['0'] / total_test * 100:.2f}%")
            print(f"  Inclusion Rate: {test_counts['1'] / total_test * 100:.2f}%")
    else:
        print(f"\nTest file not found: {test_file}")
    
    # Combined totals
    if os.path.exists(train_file) and os.path.exists(test_file):
        combined_rows = train_rows + test_rows
        combined_excluded = train_counts['0'] + test_counts['0']
        combined_included = train_counts['1'] + test_counts['1']
        combined_total = combined_excluded + combined_included
        print(f"\nCombined Totals:")
        print(f"  Total Rows:   {combined_rows}")
        print(f"  Excluded (0): {combined_excluded}")
        print(f"  Included (1): {combined_included}")
        print(f"  Labels Found: {combined_total}")
        if combined_total > 0:
            print(f"  Exclusion Rate: {combined_excluded / combined_total * 100:.2f}%")
            print(f"  Inclusion Rate: {combined_included / combined_total * 100:.2f}%")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
