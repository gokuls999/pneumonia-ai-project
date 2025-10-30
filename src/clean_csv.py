import os
import pandas as pd

base_dir = 'data/raw/sarscov2-ct'
covid_dir = os.path.join(base_dir, 'COVID')
non_covid_dir = os.path.join(base_dir, 'non-COVID')

ct_paths = []
labels = []
for dir_path, label in [(covid_dir, 1), (non_covid_dir, 0)]:
    for img in os.listdir(dir_path):
        if img.endswith('.png'):
            rel_path = os.path.relpath(os.path.join(dir_path, img), 'data/raw')  # Clean: sarscov2-ct/non-COVID/img.png
            ct_paths.append(rel_path)
            labels.append(label)

dummy_texts = ["Fever, cough, pneumonia symptoms" if l == 1 else "Normal checkup, no symptoms" for l in labels]
severities = [7.5 if l == 1 else 1.0 for l in labels]
cxr_paths = ["dummy_cxr.jpg"] * len(ct_paths)

df = pd.DataFrame({
    'patient_id': range(len(ct_paths)),
    'notes': dummy_texts,
    'cxr_path': cxr_paths,
    'ct_path': ct_paths,
    'severity': severities,
    'pneumonia_type': labels
})
df.to_csv('data/raw/metadata.csv', index=False)
print("Clean CSV created with", len(df), "entries!")
print("Sample ct_path:", df['ct_path'].iloc[0])