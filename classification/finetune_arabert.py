import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("libyan_clean.csv")

# تنظيف احتياطي
df = df.dropna(subset=['text', 'label'])

df['label'] = pd.to_numeric(df['label'], errors='coerce')
df = df.dropna(subset=['label'])

df['label'] = df['label'].astype(int)

# =========================
# 2. تقسيم البياناتtest /train
# =========================
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

print("Train:", train_df.shape)
print("Test:", test_df.shape)
# =========================
# 3 تحميل Arabic BERT
# =========================
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

model_name = "aubmindlab/bert-base-arabertv02"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)
# =========================
# 4. تجهيز البيانات للتدريب
# =========================
train_texts = train_df['text'].tolist()
train_labels = train_df['label'].tolist()

test_texts = test_df['text'].tolist()
test_labels = test_df['label'].tolist()
# =========================
# 5. ترميز البيانات 
# =========================
train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=128
)

test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding=True,
    max_length=128
)
# =========================
# 6. تحويل البيانات إلى تنسورات 
# =========================
import torch

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }

        item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.labels)
    
train_dataset = CustomDataset(
    train_encodings,
    train_labels
)

test_dataset = CustomDataset(
    test_encodings,
    test_labels
)
# =========================
# 7. إعداد التدريب      
# =========================
from transformers import Trainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./finetuned_results",
    num_train_epochs=8,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    logging_steps=10,
    save_strategy="epoch"
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)
trainer.train()
# =========================
# التقييم 
# =========================
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

predictions = trainer.predict(test_dataset)

preds = predictions.predictions.argmax(axis=1)

acc = accuracy_score(
    test_labels,
    preds
)

print("Accuracy:", acc)

print(
    classification_report(
        test_labels,
        preds
    )
)