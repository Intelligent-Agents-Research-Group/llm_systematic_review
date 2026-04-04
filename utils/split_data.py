import pandas as pd
from sklearn.model_selection import train_test_split


def normalize_text_columns(df):
    """Replace escaped newline literals ('\\n', '/n') with actual newlines in all string columns."""
    for col in df.select_dtypes(include="object").columns:
        df[col] = (
            df[col]
            .str.replace("\\n", "\n", regex=False)
            .str.replace("/n", "\n", regex=False)
        )
    return df

def split_data(input_csv, train_csv, test_csv, test_size=0.15, random_state=140126):
    # random_state=140126 is the fixed seed used for reproducibility of the published train/test split.
    data = pd.read_csv(input_csv)
    data = normalize_text_columns(data)

    # Split the dataset into training and testing sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)

    # Save the splits to CSV files
    train_data.to_csv(train_csv, index=False, header=["train"])
    test_data.to_csv(test_csv, index=False, header=["test"])
    print(f"Data split into {train_csv} and {test_csv}, with train size {len(train_data)} and test size {len(test_data)}.")

if __name__ == "__main__":
    split_data("data/200_0_172_1 LFM.csv", "data/train_data_LFM.csv", "data/test_data_LFM.csv")
