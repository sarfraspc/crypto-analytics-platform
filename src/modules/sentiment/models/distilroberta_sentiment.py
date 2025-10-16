import numpy as np
import random
import logging
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import mlflow
import torch
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import transformers

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

class SentimentTrainer:    
    def __init__(self, model_name: str = "distilroberta-base"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.class_weights = None
        
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    
    def log_dataset_stats(self, dataset, split_name):
        labels = dataset["label"]
        counts = pd.Series(labels).value_counts().sort_index()
        logger.info(f"{split_name} label distribution: {dict(counts)}")
        return counts

    class WeightedTrainer(Trainer):
        def __init__(self, class_weights=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights
        
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            
            if self.class_weights is not None:
                weights = self.class_weights.to(model.device)
                loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    def fine_tune(
        self,
        train_parquet_path: str,
        val_parquet_path: str,
        save_dir: str,
        max_length: int = 128,
        batch_size: int = 16,
        num_epochs: int = 4,
        learning_rate: float = 2e-5,
        run_name: str = "sentiment_finetune",
        mlflow_uri: str = None,
    ):
        try:
            if mlflow_uri:
                mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment("crypto-sentiment-finetuning")
            
            dataset = load_dataset("parquet", data_files={
                "train": train_parquet_path, 
                "validation": val_parquet_path
            })
            
            dataset = dataset.filter(lambda example: example["label"] is not None)
            logger.info(f"Dataset loaded: Train={len(dataset['train'])}, Val={len(dataset['validation'])}")
            
            train_counts = self.log_dataset_stats(dataset["train"], "train")
            val_counts = self.log_dataset_stats(dataset["validation"], "val")
            
            labels = dataset["train"]["label"]
            self.class_weights = torch.tensor(
                compute_class_weight('balanced', classes=np.unique(labels), y=labels),
                dtype=torch.float32
            )
            logger.info(f"Class weights: {self.class_weights}")
            
            if mlflow.active_run():
                mlflow.end_run()
            
            with mlflow.start_run(run_name=run_name):
                params = {
                    "train_size": len(dataset['train']),
                    "val_size": len(dataset['validation']),
                    "max_length": max_length,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "learning_rate": learning_rate,
                    "class_weights": self.class_weights.tolist(),
                }
                for key, value in params.items():
                    mlflow.log_param(key, value)
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                
                def tokenize_function(examples):
                    tokenized = self.tokenizer(
                        examples["text"], 
                        truncation=True, 
                        max_length=max_length, 
                        padding=False
                    )
                    tokenized["labels"] = examples["label"]
                    return tokenized
                
                tokenized_datasets = dataset.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=[col for col in dataset["train"].column_names if col != "label"]
                ).with_format("torch")
                
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=3,
                    problem_type="single_label_classification",
                    id2label={0: "BEARISH", 1: "BULLISH", 2: "NEUTRAL"},
                    label2id={"BEARISH": 0, "BULLISH": 1, "NEUTRAL": 2},
                )
                
                training_args = TrainingArguments(
                    output_dir=save_dir,
                    num_train_epochs=num_epochs,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size * 2,
                    learning_rate=learning_rate,
                    weight_decay=0.01,
                    warmup_ratio=0.1,
                    eval_strategy="epoch",
                    save_strategy="epoch",
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_f1",
                    greater_is_better=True,
                    report_to="none",
                    logging_steps=50,
                    save_total_limit=2,
                    dataloader_drop_last=False,
                    fp16=torch.cuda.is_available(),
                    seed=42,
                )
                
                trainer = self.WeightedTrainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=tokenized_datasets["train"],
                    eval_dataset=tokenized_datasets["validation"],
                    tokenizer=self.tokenizer,
                    data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
                    compute_metrics=self.compute_metrics,
                    class_weights=self.class_weights,
                )
                
                logger.info("Starting training...")
                train_result = trainer.train()
                trainer.save_model(save_dir)
                self.tokenizer.save_pretrained(save_dir)
                
                eval_results = trainer.evaluate()
                for key, value in {**train_result.metrics, **eval_results}.items():
                    mlflow.log_metric(key, value)
                
                logger.info(f"Training complete. Model saved to {save_dir}")
                return eval_results
                
        except Exception as e:
            logger.error(f"Training failed: {e}")
            if mlflow.active_run():
                mlflow.end_run(status="FAILED")
            raise

def train_sentiment_model(
    data_dir: str = None,
    save_dir: str = None,
    **kwargs
):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "../../../data/datasets/finetune_data")
    
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), "saved/finetuned_model")
    
    train_path = os.path.join(data_dir, "train.parquet")
    val_path = os.path.join(data_dir, "val.parquet")
    
    trainer = SentimentTrainer()
    return trainer.fine_tune(train_path, val_path, save_dir, **kwargs)

if __name__ == "__main__":
    results = train_sentiment_model(
        num_epochs=4,
        batch_size=16,
        run_name="local_training"
    )
    print(f"Training results: {results}")