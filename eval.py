import pandas as pd
from bixdata.dataloading import load_meta_data
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score
)

# Deine bisherigen Zeilen
meta, _ = load_meta_data("audio_data_bix/metadata3")
result_picture = pd.read_csv("results/bix_llm_story_reading_Mistral-Small-3.2-24B-Instruct-2506.csv")
merged = meta.merge(result_picture, on='subject', how='inner').sort_values(by=['subject']).reset_index(drop=True)

# -------------------------
# 1️⃣ Werte in Spalte "type" umbenennen
# -------------------------
rename_map = {
    "patient_dat": "DEM",
    "patient_mci": "MCI",
    "control": "NCI"
}

merged["type"] = merged["type"].replace(rename_map)

# -------------------------
# 2️⃣ Klassifikationsmetriken berechnen
# -------------------------
y_true = merged["type"]
y_pred = merged["cognitive_status"]

# (a) Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=["NCI", "MCI", "DEM"])
cm_df = pd.DataFrame(cm, index=["True_NCI", "True_MCI", "True_DEM"], columns=["Pred_NCI", "Pred_MCI", "Pred_DEM"])

# (b) Report
report = classification_report(y_true, y_pred, labels=["NCI", "MCI", "DEM"], digits=3, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# (c) Accuracy
acc = accuracy_score(y_true, y_pred)
bal_acc = balanced_accuracy_score(y_true, y_pred)

# -------------------------
# 3️⃣ Ausgabe
# -------------------------
print("✅ Klassifikationsmetriken (Mistral Geschichte Lesen)\n")
print("Confusion Matrix:")
print(cm_df, "\n")
print("Classification Report:")
print(report_df, "\n")
print(f"Accuracy: {acc:.3f}")
print(f"Balanced Accuracy: {bal_acc:.3f}")
