# Splits

Copy these two files here from your Google Drive after running script 01:

- `MyDrive/test/rev5_results/split_assignment.csv`  (subject_id, study_id, dicom_id, split)
- `MyDrive/test/rev5_results/split_summary.csv`

They contain only MIMIC-CXR identifiers (no images, reports or labels), which allows other
credentialed users to reproduce the exact patient-level split. Check your MIMIC-CXR data use
agreement before publishing; if in doubt, keep only `split_summary.csv` public and share the
assignment file on request.
