
# -*- coding: utf-8 -*-

# === Désactiver complètement Weights & Biases (W&B) ===
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "dryrun"  # mode inactif complet
os.environ["WANDB_SILENT"] = "true"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
print("✅ WandB désactivé avec succès.")

# === Cellule 0 : Vérifier GPU (exécuter en premier) ===
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    try:
        print("Device:", torch.cuda.get_device_name(0))
    except:
        pass
else:
    print("Pas de GPU — l'entraînement sera lent sur CPU.")

# === Cellule 1 : Installer dépendances (Colab) ===
!pip install -q transformers datasets evaluate peft bitsandbytes scikit-learn sentencepiece gradio
# bitsandbytes est optionnel, utile pour quantization 8-bit sur certains GPU
print("Install terminé")

# === Cellule 2 : Imports & configuration ===
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import evaluate
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import matplotlib.pyplot as plt
import gradio as gr
print("Imports OK")

# === Cellule 3 : Monter Google Drive pour sauvegarder ===
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
SAVE_DIR = "/content/drive/MyDrive/sti_project"
os.makedirs(SAVE_DIR, exist_ok=True)
print("Sauvegardes dans :", SAVE_DIR)

# === Cellule 4 : Charger ou créer le dataset ===
create_fake = True  # ← Mettre False pour charger ton CSV réel

if create_fake:
    # Dataset factice pour test
    texts = [
        "Douleur à la miction et écoulement vaginal anormal",
        "Aucun symptôme, consultation de routine",
        "Brûlures et pertes malodorantes",
        "Douleur pelvienne sans autres symptômes",
        "Écoulement purulent, douleur, fièvre",
        "Pas de plainte, test négatif",
        "Démangeaisons et irritation génitale",
        "Symptômes grippaux avec éruption cutanée",
        "Examen médical normal",
        "Contrôle post-traitement sans symptômes"
    ] * 50  # 500 échantillons
    labels = [1, 0, 1, 0, 1, 0, 1, 1, 0, 0] * 50
    df = pd.DataFrame({"symptoms_text": texts, "has_sti": labels})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print("Dataset factice créé, taille:", len(df))
    print("Répartition des labels:")
    print(df["has_sti"].value_counts(normalize=True))
else:
    # Charger ton CSV réel
    data_path = "/content/drive/MyDrive/sti_project/stis.csv"
    df = pd.read_csv(data_path)
    print("CSV chargé :", data_path)

display(df.head())

# === Cellule 5 : Anonymisation et préparation des données ===
# Supprimer les colonnes d'identification
to_drop = ["patient_id", "name", "address", "phone", "email", "ssn"]
for c in to_drop:
    if c in df.columns:
        df = df.drop(columns=c)

# Gérer la date de naissance si présente
if "date_of_birth" in df.columns:
    df["age_years"] = pd.to_datetime("today").year - pd.to_datetime(df["date_of_birth"]).dt.year
    df["age_group"] = pd.cut(df["age_years"], bins=[0, 18, 25, 35, 50, 120],
                            labels=["<18", "18-25", "26-35", "36-50", "50+"])
    df = df.drop(columns=["date_of_birth", "age_years"])

# Créer la colonne has_sti si elle n'existe pas
if "has_sti" not in df.columns:
    if "diagnosis" in df.columns:
        sti_set = {"chlamydia", "gonorrhea", "syphilis", "trichomoniasis", "herpes"}
        df["has_sti"] = df["diagnosis"].astype(str).str.lower().apply(
            lambda x: 1 if any(sti in x.lower() for sti in sti_set) else 0
        )
        print("Colonne has_sti créée depuis diagnosis.")
    else:
        raise ValueError("Colonne 'has_sti' manquante et impossible à créer depuis 'diagnosis'")

print("Colonnes disponibles :", df.columns.tolist())
display(df.head())

# === Cellule 6 : Split train / valid / test ===
text_col = "symptoms_text"

if text_col not in df.columns:
    raise ValueError(f"Colonne texte '{text_col}' introuvable.")

# Garder uniquement les colonnes utiles
keep_cols = [text_col, "has_sti"] + [c for c in ["age_group", "gender"] if c in df.columns]
df = df[keep_cols].dropna(subset=[text_col, "has_sti"]).reset_index(drop=True)

# Split stratifié
train_df, test_df = train_test_split(df, test_size=0.15, stratify=df["has_sti"], random_state=42)
train_df, valid_df = train_test_split(train_df, test_size=0.15, stratify=train_df["has_sti"], random_state=42)

# Créer le DatasetDict
ds = DatasetDict({
    "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
    "validation": Dataset.from_pandas(valid_df.reset_index(drop=True)),
    "test": Dataset.from_pandas(test_df.reset_index(drop=True))
})
print("Tailles splits:", {k: len(ds[k]) for k in ds})

# === Cellule 7 : Charger le tokenizer ===
model_name = "camembert-base"
num_labels = 2

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
print("Tokeniser chargé:", model_name, "vocab_size:", tokenizer.vocab_size)

# === Cellule 8 : Tokenization avec métadonnées ===
max_len = 256

def preprocess_and_label(batch):
    texts = []
    for i in range(len(batch[text_col])):
        t = str(batch[text_col][i])
        meta = []
        if "age_group" in batch and batch["age_group"][i] is not None:
            meta.append(f"AGE:{batch['age_group'][i]}")
        if "gender" in batch and batch["gender"][i] is not None:
            meta.append(f"GENDER:{batch['gender'][i]}")
        prefix = " ".join(meta)
        if prefix:
            texts.append(prefix + " | " + t)
        else:
            texts.append(t)
    
    enc = tokenizer(texts, truncation=True, padding="max_length", max_length=max_len)
    enc["labels"] = [int(x) for x in batch["has_sti"]]
    return enc

tokenized = ds.map(preprocess_and_label, batched=True)
print("Tokenization + labels OK")
print("Colonnes du dataset tokenisé (train):", tokenized["train"].column_names)

# === Cellule 9 : Configuration LoRA et chargement du modèle ===
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "key", "value", "dense"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)
model = get_peft_model(model, lora_config)
print("Modèle + LoRA prêt — paramètres LoRA ajoutés.")

# Afficher le nombre de paramètres entraînables
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Paramètres entraînables: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")

# === Cellule 10 : Configuration de l'entraînement (CORRIGÉE) ===
# Version compatible avec les nouvelles versions de transformers
training_args = TrainingArguments(
    output_dir="/content/outputs/sti_camembert_lora",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    eval_strategy="epoch",  # CORRECTION: 'evaluation_strategy' -> 'eval_strategy'
    save_strategy="epoch",
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    save_total_limit=3,
    report_to="none"  # Désactive complètement W&B
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Métriques d'évaluation
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy.compute(predictions=preds, references=labels)["accuracy"]),
        "f1": float(f1.compute(predictions=preds, references=labels, average="weighted")["f1"])
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
print("Trainer prêt")

# === Cellule 11 : Entraînement du modèle ===
print("Début de l'entraînement...")
train_result = trainer.train()
print("Entraînement terminé.")

# Sauvegarder le modèle
OUT = os.path.join(SAVE_DIR, "sti_camembert_lora")
os.makedirs(OUT, exist_ok=True)
trainer.save_model(OUT)
tokenizer.save_pretrained(OUT)
model.save_pretrained(OUT)  # Sauvegarde spécifique PEFT
print(f"Modèle sauvegardé dans : {OUT}")

# Évaluation finale sur validation
eval_metrics = trainer.evaluate(eval_dataset=tokenized["validation"])
print("Metrics validation finale:", eval_metrics)

# === Cellule 12 : Évaluation sur le jeu de test ===
print("Évaluation sur le jeu de test...")
preds_output = trainer.predict(tokenized["test"])
logits = preds_output.predictions

if logits.ndim == 2:
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    y_pred = np.argmax(logits, axis=-1)
    y_proba_pos = probs[:, 1]
else:
    y_pred = (logits > 0).astype(int)
    y_proba_pos = y_pred.astype(float)

y_true = np.array(tokenized["test"]["labels"])

print("\n=== Classification report (test) ===")
print(classification_report(y_true, y_pred, digits=4))

print("\n=== Matrice de confusion ===")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# Métriques détaillées
metrics_test = {
    "accuracy": float((y_pred == y_true).mean()),
    "f1": float(f1.compute(predictions=y_pred, references=y_true, average="weighted")["f1"])
}
try:
    metrics_test["roc_auc"] = float(roc_auc_score(y_true, y_proba_pos))
except Exception as e:
    metrics_test["roc_auc"] = None
    print(f"ROC AUC non calculable: {e}")

print("\nMetrics résumé test:", metrics_test)

# === Cellule 13 : Visualisations ===
# Matrice de confusion
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Matrice de confusion - Test")
plt.colorbar()
classes = ["Sans IST (0)", "Avec IST (1)"]
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel("Prédiction")
plt.ylabel("Vrai label")

# Ajouter les valeurs dans les cases
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.show()

# Courbe ROC si binaire
if len(np.unique(y_true)) == 2 and metrics_test["roc_auc"] is not None:
    fpr, tpr, _ = roc_curve(y_true, y_proba_pos)
    auc = metrics_test["roc_auc"]
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Courbe ROC - Test')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()
else:
    print("Courbe ROC non tracée (problème binaire requis)")

# === Cellule 14 : Sauvegarde des prédictions ===
# Reconstruire le DataFrame test avec prédictions
test_pd = test_df.reset_index(drop=True).copy()
test_pd["pred"] = y_pred
test_pd["proba_pos"] = y_proba_pos
test_pd["proba_neg"] = 1 - y_proba_pos
test_pd["correct"] = test_pd["has_sti"] == test_pd["pred"]

# Sauvegarder toutes les prédictions
preds_fp = os.path.join(OUT, "test_predictions_full.csv")
test_pd.to_csv(preds_fp, index=False)
print("Prédictions complètes sauvegardées :", preds_fp)

# Exemples mal classés
bad_predictions = test_pd[~test_pd["correct"]]
bad_fp = os.path.join(OUT, "mauvaise_pred_examples.csv")
bad_predictions.to_csv(bad_fp, index=False)
print(f"Exemples mal classés ({len(bad_predictions)}) sauvegardés :", bad_fp)

# Statistiques des mauvaises prédictions
if len(bad_predictions) > 0:
    print("\nAnalyse des mauvaises prédictions:")
    print(bad_predictions["has_sti"].value_counts().rename("Mauvaises prédictions par classe"))

# === Cellule 15 : Interface Gradio ===
print("Préparation de l'interface Gradio...")

# Recharger le modèle pour l'inférence (meilleure pratique)
try:
    tokenizer_inf = AutoTokenizer.from_pretrained(OUT)
    base_model_inf = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model_inf = PeftModel.from_pretrained(base_model_inf, OUT)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_inf.eval()
    model_inf.to(device)
    print(f"Modèle rechargé pour inférence sur {device}")
    
except Exception as e:
    print(f"Erreur rechargement modèle: {e}")
    print("Utilisation du modèle existant...")
    model_inf = model
    tokenizer_inf = tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_inf.to(device)
    model_inf.eval()

def predict_sti(text, age_group=None, gender=None):
    """
    Prédit si le texte décrit des symptômes d'IST
    """
    if not text or text.strip() == "":
        return "❌ Veuillez entrer une description des symptômes", 0.0, 0.0
    
    try:
        # Construire le texte avec métadonnées optionnelles
        meta_parts = []
        if age_group and age_group != "Non spécifié":
            meta_parts.append(f"AGE:{age_group}")
        if gender and gender != "Non spécifié":
            meta_parts.append(f"GENDER:{gender}")
        
        if meta_parts:
            full_text = " ".join(meta_parts) + " | " + text.strip()
        else:
            full_text = text.strip()
        
        # Tokenization
        inputs = tokenizer_inf(full_text, return_tensors="pt", truncation=True,
                             padding=True, max_length=max_len)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Prédiction
        with torch.no_grad():
            outputs = model_inf(**inputs)
            logits = outputs.logits
        
        # Conversion en probabilités
        probs = torch.softmax(logits, dim=-1)
        prob_positive = probs[0][1].item()
        prob_negative = probs[0][0].item()
        
        # Interprétation
        if prob_positive > 0.7:
            prediction = "🟡 IST probable - Consultation recommandée"
            confidence = prob_positive
        elif prob_positive > 0.3:
            prediction = "🟠 Suspicion d'IST - Consultation conseillée"
            confidence = prob_positive
        else:
            prediction = "🟢 Aucun signe d'IST détecté"
            confidence = prob_negative
        
        return prediction, prob_positive, prob_negative
        
    except Exception as e:
        return f"❌ Erreur lors de la prédiction: {str(e)}", 0.0, 0.0

# Test de la fonction
print("Test de la fonction de prédiction...")
test_text = "Douleur à la miction et écoulement vaginal"
result, prob_pos, prob_neg = predict_sti(test_text)
print(f"Test: '{test_text}' -> {result} (prob IST: {prob_pos:.3f})")

# Création de l'interface Gradio
age_options = ["Non spécifié", "<18", "18-25", "26-35", "36-50", "50+"]
gender_options = ["Non spécifié", "F", "M", "Autre"]

with gr.Blocks(title="Détecteur d'IST - Analyse de symptômes", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔍 Détecteur d'IST - Analyse de symptômes
    Cet outil analyse la description de symptômes pour détecter des signes d'Infections Sexuellement Transmissibles (IST).
    
    **⚠️ Attention**: Ceci est un outil d'aide à la décision, pas un diagnostic médical. Consultez toujours un professionnel de santé.
    """)
    
    with gr.Row():
        with gr.Column():
            symptoms_input = gr.Textbox(
                label="Description des symptômes",
                placeholder="Ex: Douleur à la miction, écoulement anormal, brûlures...",
                lines=3,
                max_lines=6
            )
            
            age_dropdown = gr.Dropdown(
                label="Groupe d'âge (optionnel)",
                choices=age_options,
                value="Non spécifié"
            )
            
            gender_dropdown = gr.Dropdown(
                label="Genre (optionnel)",
                choices=gender_options,
                value="Non spécifié"
            )
            
            analyze_btn = gr.Button("Analyser les symptômes", variant="primary")
        
        with gr.Column():
            prediction_output = gr.Textbox(
                label="Résultat de l'analyse",
                interactive=False,
                lines=2
            )
            
            with gr.Row():
                prob_positive = gr.Number(
                    label="Probabilité IST",
                    interactive=False,
                    precision=3
                )
                prob_negative = gr.Number(
                    label="Probabilité absence IST",
                    interactive=False,
                    precision=3
                )
    
    # Exemples rapides
    gr.Markdown("### Exemples rapides:")
    examples = gr.Examples(
        examples=[
            ["Douleur à la miction et écoulement vaginal anormal", "26-35", "F"],
            ["Aucun symptôme, consultation de routine", "Non spécifié", "Non spécifié"],
            ["Brûlures et pertes malodorantes", "18-25", "F"],
            ["Démangeaisons et irritation génitale", "Non spécifié", "M"]
        ],
        inputs=[symptoms_input, age_dropdown, gender_dropdown],
        outputs=[prediction_output, prob_positive, prob_negative],
        fn=predict_sti,
        cache_examples=False
    )
    
    # Liaison du bouton
    analyze_btn.click(
        fn=predict_sti,
        inputs=[symptoms_input, age_dropdown, gender_dropdown],
        outputs=[prediction_output, prob_positive, prob_negative]
    )
    
    # Disclaimer
    gr.Markdown("""
    ---
    **Disclaimer médical**: 
    - Cet outil utilise l'IA pour analyser les symptômes décrits
    - Il ne remplace pas une consultation médicale professionnelle
    - En cas de symptômes, consultez un médecin ou un centre de santé
    - Les résultats sont fournis à titre informatif seulement
    """)

print("Interface Gradio créée avec succès!")
print("Pour lancer l'interface, exécutez: demo.launch(share=True)")

# Lancer l'interface
try:
    demo.launch(share=True, debug=True)
except Exception as e:
    print(f"Erreur lancement Gradio: {e}")
    print("Tentative sans partage...")
    demo.launch(share=False, debug=True)

print("✅ Code exécuté avec succès!")
