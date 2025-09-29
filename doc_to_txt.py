import os
from docx import Document

# === Configure these paths ===
input_dir = "/Users/franziskabraun/Documents/bix-docx/"      # folder where your .docx files are
output_dir = "/Users/franziskabraun/Documents/bix-txt/"      # folder where you want to save .txt files

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through all files in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".docx"):
        docx_path = os.path.join(input_dir, filename)
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_filename)

        # Read the .docx file
        doc = Document(docx_path)

        # Write text to .txt file
        with open(txt_path, "w", encoding="utf-8") as out:
            for para in doc.paragraphs:
                out.write(para.text + "\n")

        print(f"Converted: {filename} → {txt_filename}")

print("All files converted successfully!")
