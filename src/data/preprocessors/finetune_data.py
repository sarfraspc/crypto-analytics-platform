from datasets import load_dataset
import pandas as pd

def prepare_sentiment_data():
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")

    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()

    label_map = {"positive": 2, "neutral": 1, "negative": 0}
    train_df["label"] = train_df["label"].map(label_map)
    val_df["label"] = val_df["label"].map(label_map)

    train_df.to_parquet("src/data/datasets/finetune_data/train.parquet", index=False)
    val_df.to_parquet("src/data/datasets/finetune_data/val.parquet", index=False)

    print("Sentiment dataset prepared and saved.")
    print("Label distribution (train):\n", train_df["label"].value_counts())

if __name__ == "__main__":
    prepare_sentiment_data()