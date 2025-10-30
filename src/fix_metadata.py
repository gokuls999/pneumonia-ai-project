import os
import pandas as pd

# Set paths from project root
base_dir = 'data/raw/sarscov2-ct'
covid_dir = os.path.join(base_dir, 'COVID')
non_covid_dir = os.path.join(base_dir, 'non-COVID')

# Check folders
if not os.path.exists(covid_dir):
    print("COVID folder not found:", covid_dir)
if not os.path.exists(non_covid_dir):
    print("non-COVID folder not found:", non_covid_dir)

# Collect relative paths (relative to data/raw)
ct_paths = []
labels = []
for dir_path, label in [(covid_dir, 1), (non_covid_dir, 0)]:
    for img in os.listdir(dir_path):
        if img.endswith('.png'):
            full_path = os.path.join(dir_path, img)
            rel_path = os.path.relpath(full_path, 'data/raw')  # Relative: sarscov2-ct/non-COVID/img.png
            ct_paths.append(rel_path)
            labels.append(label)

# Dummy data
dummy_texts = ["Fever, cough, pneumonia symptoms" if l == 1 else "Normal checkup, no symptoms" for l in labels]
severities = [7.5 if l == 1 else 1.0 for l in labels]
cxr_paths = ["dummy_cxr.jpg"] * len(ct_paths)

# DataFrame
df = pd.DataFrame({
    'patient_id': range(len(ct_paths)),
    'notes': dummy_texts,
    'cxr_path': cxr_paths,
    'ct_path': ct_paths,
    'severity': severities,
    'pneumonia_type': labels
})
df.to_csv('data/raw/metadata.csv', index=False)
print("Fixed metadata CSV created with", len(df), "entries!")