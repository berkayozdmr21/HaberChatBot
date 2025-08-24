from flask import Flask, render_template, request
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os

app = Flask(__name__)

# Veri kümesi yükleniyor
df = pd.read_csv(r"C:\Users\User\Desktop\haber_chatbot_final\guncellenmis_haber_veriseti_130k.csv")
df = df.dropna().reset_index(drop=True)

# Etiketleri sayısal değerlere dönüştür
label_dict = {label: idx for idx, label in enumerate(df["Konu"].unique())}
label_dict_inv = {v: k for k, v in label_dict.items()}
df["Konu"] = df["Konu"].map(label_dict)

# Eğitim ve doğrulama verileri
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Haber Başlığı"], df["Konu"], test_size=0.2, random_state=42
)

# Tokenizer yükleniyor
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=256)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=256)

# Dataset sınıfı
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels.iloc[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encodings, train_labels)
val_dataset = NewsDataset(val_encodings, val_labels)

# Eğitim ayarları (her iki eğitim için kullanılabilir)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=200,
    weight_decay=0.05,
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=2e-5
)

# Modeli sıfırdan eğitme fonksiyonu
def train_model(save_dir="./saved_model"):
    print(">>> İlk eğitim başlatılıyor...")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_dict))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    trainer.train()
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f">>> Model {save_dir} dizinine kaydedildi.")
    return model

# Daha önce eğitilmiş modeli tekrar fine-tune etme fonksiyonu
def fine_tune_model(load_dir="./saved_model", save_dir="./saved_model_finetuned"):
    print(">>> Fine-tuning başlatılıyor...")
    model = BertForSequenceClassification.from_pretrained(load_dir, num_labels=len(label_dict))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    trainer.train()
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f">>> Fine-tuned model {save_dir} dizinine kaydedildi.")
    return model

# Model eğitilmiş mi kontrol et
if os.path.exists("./saved_model_finetuned"):
    print(">>> Fine-tuned model bulundu. Yükleniyor...")
    model = BertForSequenceClassification.from_pretrained("./saved_model_finetuned")
    tokenizer = BertTokenizer.from_pretrained("./saved_model_finetuned")
elif os.path.exists("./saved_model"):
    print(">>> Eğitilmiş model bulundu. Fine-tuning yapılacak.")
    model = fine_tune_model()
    tokenizer = BertTokenizer.from_pretrained("./saved_model_finetuned")
else:
    print(">>> Hiç model bulunamadı. Eğitim başlatılıyor...")
    model = train_model()
    model = fine_tune_model()
    tokenizer = BertTokenizer.from_pretrained("./saved_model_finetuned")

# Ana sayfa
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        title = request.form["title"].strip()
        if len(title.split()) < 3:
            prediction = "Lütfen en az 3 kelimelik bir başlık giriniz."
        else:
            inputs = tokenizer(title, return_tensors="pt", padding=True, truncation=True, max_length=256)
            with torch.no_grad():
                outputs = model(**inputs)
            _, predicted = torch.max(outputs.logits, dim=1)
            predicted_label = label_dict_inv[predicted.item()]
            prediction = predicted_label

    return render_template("index.html", prediction=prediction)

# --- BURASI YENİ: Başarım metriklerini hesaplayıp gösteren route ---

@app.route("/metrics")
def metrics():
    all_preds = []
    all_labels = []

    for i in range(len(val_dataset)):
        item = val_dataset[i]
        inputs = {key: val.unsqueeze(0) for key, val in item.items() if key != "labels"}
        label = item["labels"].item()
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
        all_preds.append(pred)
        all_labels.append(label)

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=list(label_dict.keys()), zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # Confusion matrix için HTML tablo oluştur
    cm_html = "<table border='1' style='border-collapse: collapse; margin: auto;'>"
    cm_html += "<tr><th></th>" + "".join(f"<th>{label}</th>" for label in label_dict.keys()) + "</tr>"
    for i, row in enumerate(cm):
        cm_html += f"<tr><th>{list(label_dict.keys())[i]}</th>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
    cm_html += "</table>"

    return render_template(
        "metrics.html",
        accuracy=accuracy,
        report=report.replace("\n", "<br>"),
        confusion_matrix=cm_html
    )

if __name__ == "__main__":
    app.run(debug=True)
