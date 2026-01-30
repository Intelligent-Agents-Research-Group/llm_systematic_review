#!/usr/bin/env python3
"""
Script to swap 20% of includes from 200_0_172_1 LFM.csv with Phase I samples.

This ensures Phase I has some positive examples (1s) for meaningful evaluation.
"""

import pandas as pd
import re
import random

# Set seed for reproducibility
random.seed(42)

# File paths
LFM_FILE = 'data/200_0_172_1 LFM.csv'
PHASE1_FILE = 'data/phase I screening_ALL studies_cleaned_prompts - phase I screening_ALL studies_cleaned_prompts.csv'


def extract_label(text):
    """Extract the 0 or 1 label from the formatted LFM text."""
    # The file uses literal \n (backslash-n) not actual newlines
    pattern = r'assistant\\n([01])<\|im_end\|>'
    match = re.search(pattern, str(text))
    if match:
        return int(match.group(1))
    return None


def extract_prompt_from_lfm(text):
    """
    Extract the prompt content from LFM formatted text.
    Returns just the prompt without the tags, suitable for Phase I format.
    """
    # Pattern to extract content between <|im_start|>user and <|im_end|>
    # The file uses literal \n
    pattern = r'<\|startoftext\|><\|im_start\|>user\\n(.*?)<\|im_end\|>'
    match = re.search(pattern, str(text), re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def format_for_lfm(prompt, decision):
    """
    Format a Phase I prompt for LFM training format.
    
    Args:
        prompt: The prompt text from Phase I
        decision: The decision (0 or 1)
        
    Returns:
        Formatted string with proper tags (using literal \\n)
    """
    # Use literal backslash-n to match the existing format
    formatted = f'<|startoftext|><|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n{decision}<|im_end|>\\n'
    return formatted


def extract_title(text):
    """Extract title from prompt for display purposes."""
    match = re.search(r'Title in Investigation:\s*(.+?)(?:/n|\\n|\n|Abstract)', str(text))
    if match:
        return match.group(1).strip()[:60] + "..."
    return "Unknown title"


def main():
    # Load files
    print("Loading files...")
    lfm_df = pd.read_csv(LFM_FILE)
    phase1_df = pd.read_csv(PHASE1_FILE)
    
    text_col = lfm_df.columns[0]
    
    print(f"\nLFM file: {len(lfm_df)} rows")
    print(f"Phase I file: {len(phase1_df)} rows")
    
    # Find all includes (1s) in LFM file
    includes_indices = []
    for idx, row in lfm_df.iterrows():
        label = extract_label(row[text_col])
        if label == 1:
            includes_indices.append(idx)
    
    print(f"\nIncludes (1) in LFM: {len(includes_indices)}")
    
    # Calculate 20% to swap
    num_to_swap = max(1, int(len(includes_indices) * 0.2))
    print(f"20% to swap: {num_to_swap}")
    
    # Randomly select indices to swap
    swap_indices = random.sample(includes_indices, num_to_swap)
    print(f"Selected indices: {swap_indices}")
    
    # Randomly select Phase I indices to use as replacements
    phase1_indices = random.sample(range(len(phase1_df)), num_to_swap)
    print(f"Phase I replacement indices: {phase1_indices}")
    
    print("\n" + "="*70)
    print("SWAPPING SAMPLES")
    print("="*70)
    
    # Perform swaps
    new_phase1_rows = []
    
    for i, (lfm_idx, phase1_idx) in enumerate(zip(swap_indices, phase1_indices)):
        print(f"\n--- Swap {i+1}/{num_to_swap} ---")
        
        # Get LFM sample (include) to move to Phase I
        lfm_text = lfm_df.iloc[lfm_idx][text_col]
        lfm_prompt = extract_prompt_from_lfm(lfm_text)
        lfm_label = extract_label(lfm_text)
        lfm_title = extract_title(lfm_text)
        
        print(f"\nMOVING FROM LFM (idx={lfm_idx}) -> Phase I:")
        print(f"  Title: {lfm_title}")
        print(f"  Label: {lfm_label}")
        print(f"  Prompt length: {len(lfm_prompt) if lfm_prompt else 0} chars")
        
        # Get Phase I sample to move to LFM
        phase1_prompt = phase1_df.iloc[phase1_idx]['prompt']
        phase1_decision = phase1_df.iloc[phase1_idx]['Decision']
        phase1_title = extract_title(phase1_prompt)
        
        print(f"\nMOVING FROM Phase I (idx={phase1_idx}) -> LFM:")
        print(f"  Title: {phase1_title}")
        print(f"  Decision: {phase1_decision}")
        print(f"  Prompt length: {len(phase1_prompt)} chars")
        
        # Format Phase I sample for LFM
        new_lfm_text = format_for_lfm(phase1_prompt, phase1_decision)
        
        # Verify formatting
        new_label = extract_label(new_lfm_text)
        print(f"\n  Verification - New LFM text label extracted: {new_label}")
        
        # Store new Phase I row
        new_phase1_rows.append({
            'prompt': lfm_prompt,
            'Decision': lfm_label  # This will be 1
        })
        
        # Update LFM dataframe
        lfm_df.at[lfm_idx, text_col] = new_lfm_text
    
    # Remove the Phase I rows that were moved to LFM
    phase1_df = phase1_df.drop(phase1_indices).reset_index(drop=True)
    
    # Add the new Phase I rows (former LFM includes)
    new_phase1_df = pd.DataFrame(new_phase1_rows)
    phase1_df = pd.concat([phase1_df, new_phase1_df], ignore_index=True)
    
    print("\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)
    
    # Recount labels in LFM
    lfm_zeros = 0
    lfm_ones = 0
    for idx, row in lfm_df.iterrows():
        label = extract_label(row[text_col])
        if label == 0:
            lfm_zeros += 1
        elif label == 1:
            lfm_ones += 1
    
    print(f"\nLFM file after swap:")
    print(f"  Total rows: {len(lfm_df)}")
    print(f"  Excludes (0): {lfm_zeros}")
    print(f"  Includes (1): {lfm_ones}")
    
    print(f"\nPhase I file after swap:")
    print(f"  Total rows: {len(phase1_df)}")
    print(f"  Decision value counts: {phase1_df['Decision'].value_counts().to_dict()}")
    
    # Show a sample of the new entries in Phase I
    print("\n" + "="*70)
    print("SAMPLE NEW ENTRIES IN PHASE I (should have Decision=1)")
    print("="*70)
    for i, row in new_phase1_df.iterrows():
        title = extract_title(row['prompt'])
        print(f"  {i+1}. Decision={row['Decision']}, Title: {title}")
    
    # Save updated files
    lfm_df.to_csv(LFM_FILE, index=False)
    phase1_df.to_csv(PHASE1_FILE, index=False)
    
    print(f"\n✓ Saved updated LFM file: {LFM_FILE}")
    print(f"✓ Saved updated Phase I file: {PHASE1_FILE}")
    
    # Validate Phase I has 1s now
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)
    reload_phase1 = pd.read_csv(PHASE1_FILE)
    print(f"Phase I Decision counts after save: {reload_phase1['Decision'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
