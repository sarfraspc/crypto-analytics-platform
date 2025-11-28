"""Data preprocessing utilities for ML model fine-tuning datasets."""

import pandas as pd
from datasets import load_dataset

FOLDER = "src/data/datasets/finetune_data"
train_path = f"{FOLDER}/train.parquet"
val_path = f"{FOLDER}/val.parquet"


def prepare_sentiment_data():
    """Download and prepare Twitter financial sentiment dataset for fine-tuning."""
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
    
    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    
    train_df = train_df.dropna(subset=['label'])
    val_df = val_df.dropna(subset=['label'])
    train_df['label'] = train_df['label'].astype('int64')
    val_df['label'] = val_df['label'].astype('int64')
    
    train_df[['text', 'label']].to_parquet(train_path, index=False)
    val_df[['text', 'label']].to_parquet(val_path, index=False)
    
    print("Sentiment dataset prepared and saved.")
    print(f"Train saved to: {train_path} | Shape: {train_df.shape}")
    print(f"Val saved to: {val_path} | Shape: {val_df.shape}")
    print("\nTrain label distribution (0=Bearish, 1=Bullish, 2=Neutral):\n", train_df["label"].value_counts().sort_index())
    print("\nVal label distribution:\n", val_df["label"].value_counts().sort_index())

if __name__ == "__main__":
    prepare_sentiment_data()