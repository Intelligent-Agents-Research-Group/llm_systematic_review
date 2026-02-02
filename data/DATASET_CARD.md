# Dataset Card: Systematic Review Screening Prompts

## Dataset Summary
This dataset supports a binary classification task for systematic review screening. Each record is a prompt that includes review objectives, inclusion criteria, and a candidate paper’s title and abstract. Some files include an explicit decision label (0/1), while others embed the label inside a formatted chat transcript for instruction-tuned training.

**Primary task:** Determine whether a study should be **included (1)** or **excluded (0)** based on the inclusion criteria.

## Languages
- English

## Data Files
- **data/phase I screening_ALL studies_cleaned_prompts.csv**
  - **Rows:** 8,277
  - **Columns:** `prompt`, `Decision`
  - **Description:** Cleaned prompts with explicit binary decision labels.
  - **Label distribution:**
    - `0` (exclude): 8,243
    - `1` (include): 34

- **data/train_data_LFM.csv**
  - **Rows:** 315
  - **Columns:** `train`
  - **Description:** Instruction-tuned training examples in a single text column. The label is embedded inside the chat transcript as the assistant response (0/1).

- **data/test_data_LFM.csv**
  - **Rows:** 56
  - **Columns:** `test`
  - **Description:** Instruction-tuned test examples in a single text column. The label is embedded inside the chat transcript as the assistant response (0/1).

- **data/200_0_172_1 LFM.csv**
  - **Rows:** 371
  - **Columns:** `Ready for LLM Training`
  - **Description:** Instruction-tuned prompts in a single text column for model training; label is embedded in the chat transcript.

## Data Fields
### phase I screening_ALL studies_cleaned_prompts.csv
- **prompt**: A full prompt including objectives, inclusion criteria, and the candidate study’s title and abstract.
- **Decision**: Binary label where `1` indicates inclusion and `0` indicates exclusion.

### train_data_LFM.csv / test_data_LFM.csv / 200_0_172_1 LFM.csv
- Single-column text fields containing a full chat transcript with:
  - Systematic review objectives and inclusion criteria
  - Title and abstract
  - Instruction block
  - Assistant response (label: 0/1)

## Intended Use
- Training and evaluation of instruction-tuned models for systematic review screening.
- Human-in-the-loop triage of candidate studies for inclusion.

## Out-of-Scope Use
- Clinical or policy decisions without expert review.
- Non-English abstracts or domains not aligned with the inclusion criteria.

## Known Limitations
- Severe class imbalance (very few includes).
- Prompt templates and criteria are specialized to a particular review topic.

## Licensing
- Not specified in this repository. Please consult the project maintainers or original data sources.

## How to Load (Example)
```python
import csv

with open("data/phase I screening_ALL studies_cleaned_prompts.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        prompt = row["prompt"]
        decision = int(row["Decision"])
        break
```

## Citation
If you use this dataset, please cite the repository:

```bibtex
@misc{llm_systematic_review_dataset_2026,
  title        = {Systematic Review Screening Dataset},
  author       = {Intelligent Agents Research Group},
  year         = {2026},
  howpublished = {\url{https://github.com/Intelligent-Agents-Research-Group/llm_systematic_review}}
}
```
