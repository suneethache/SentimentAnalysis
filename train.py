import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification,TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, TaskType, get_peft_model
import evaluate

# Load the dataset

imdb_data = load_dataset("imdb",download_mode="force_redownload")
train_ds = imdb_data["train"]
test_ds = imdb_data["test"]

# Tokenisation

model_name = "distilbert-base-uncased"
tokeniser = AutoTokenizer.from_pretrained(model_name)

def tokenise(batch):
    return tokeniser(batch["text"], truncation=True, padding="max_length", max_length=256)


train_ds = train_ds.map(tokenise, batched=True, batch_size=1000)
test_ds = test_ds.map(tokenise, batched=True, batch_size=1000)
train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Model and LoRA

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8,
                         lora_alpha=16, lora_dropout=0.1, target_modules=["q_lin", "v_lin"])

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# Accuracy metric

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy: ": accuracy.compute(predictions=preds, references=labels)["accuracy"]}


# Training args

training_args = TrainingArguments(output_dir="finetuned-distilbert-imdb",
                                  overwrite_output_dir=True,
                                  evaluation_strategy="epoch",
                                  save_strategy="epoch",
                                  num_train_epochs=3,
                                  per_device_train_batch_size=8,
                                  per_device_eval_batch_size=8,
                                  logging_strategy="epoch",
                                  learning_rate=5e-5,
                                  report_to=["mlflow"],
                                  run_name="DistilBERT-Imdb_LoRA")

# Trainer

trainer = Trainer(model=model, args=training_args, train_dataset=train_ds,
                  eval_dataset=test_ds, data_collator=DataCollatorWithPadding(tokeniser),
                  compute_metrics=compute_metrics)

trainer.train()
trainer.evaluate(test_ds)
trainer.save_model("models/imdb_model")


