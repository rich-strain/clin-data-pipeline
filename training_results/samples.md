# Before/after generation samples

## Sample 1

**Instruction:**
```
Patient: [PATIENT NAME], DOB 2000-10-02, Female, MRN [MRN]
Address: [ADDRESS]
Visit Date: 2026-12-31

CC: Presents for routine follow-up of T2DM.

HPI: Continues management of T2DM, first diagnosed 2024-11-11. No acute concerns reported. Patient reports doing well on current regimen for GERD, diagnosed 2022-03-15. Continues management of Chronic obstructive pulmonary disease, unspecified, first diagnosed 2023-03-19. No acute concerns reported.

PMH: T2DM (dx 2024-11-11), GERD (dx 2022-03-15), COPD (dx 2023-03-19).

Meds: Omeprazole 20 MG Oral Capsule - 20mg PO QD.

VS: HR 71.6, Temp 99.3, Wt 109.9, BP 106.0 mm[Hg], Ht 163.9 cm.

A/P: Continue current management of T2DM. No changes to GERD management at this time. Continue current management of COPD.
```

**Ground truth response:**
```json
{"date_of_birth": "2000-10-02", "diagnoses": [{"name": "Type 2 Diabetes Mellitus"}, {"name": "Gastro-esophageal reflux disease without esophagitis"}, {"name": "Chronic obstructive pulmonary disease, unspecified"}], "medications": [{"name": "Omeprazole 20 MG Oral Capsule", "dosage": "Take 20 mg by mouth once daily"}], "vitals": [{"name": "Heart rate", "value": 71.6, "unit": "beats/minute"}, {"name": "Body temperature", "value": 37.4, "unit": "Cel"}, {"name": "Body weight", "value": 109.9, "unit": "kg"}, {"name": "Systolic blood pressure", "value": 106.0, "unit": "mm[Hg]"}, {"name": "Body height", "value": 163.9, "unit": "cm"}]}
```

**Base model (no adapter) output:**
```
```json
{
  "date_of_birth": "2000-10-02",
  "diagnoses": ["T2DM", "GERD", "COPD"],
  "medications": ["Omeprazole 20 MG Oral Capsule - 20mg PO QD"],
  "vitals": {
    "HR": 71.6,
    "Temp": 99.3,
    "Wt": 109.9,
    "BP": 106.0
  }
}
```
```

**Fine-tuned model (with adapter) output:**
```
{"date_of_birth": "2000-10-02", "diagnoses": [{"name": "Type 2 diabetes mellitus without complications"}, {"name": "Gastro-esophageal reflux disease without esophagitis"}, {"name": "Chronic obstructive pulmonary disease, unspecified"}], "medications": [{"name": "Omeprazole 20 MG Oral Capsule", "dosage": "Take 20 mg by mouth once daily"}], "vitals": [{"name": "Heart rate", "value": 71.6, "unit": "beats/minute"}, {"name": "Body temperature", "value": 37.3, "unit": "Cel"}, {"name": "Body weight", "value": 98.0, "unit": "kg"}, {"name": "Systolic blood pressure", "value": 106.0, "unit": "mm[Hg]"}, {"name": "Body height", "value": 163.9, "unit": "cm"}]}
```

## Sample 2

**Instruction:**
```
Patient: [PATIENT NAME], DOB 1946-02-19, Male, MRN [MRN]
Address: [ADDRESS]
Visit Date: 2024-09-20

CC: Here today to discuss T2DM.

HPI: T2DM (dx 2019-10-10) remains stable; no new complaints today.

PMH: Type 2 diabetes mellitus without complications (dx 2019-10-10).

Meds: Lisinopril 10 MG Oral Tablet - 10mg PO QD.

A/P: Continue present treatment for T2DM; recheck at next visit.
```

**Ground truth response:**
```json
{"date_of_birth": "1946-02-19", "diagnoses": [{"name": "Type 2 diabetes mellitus without complications"}], "medications": [{"name": "Lisinopril 10 MG Oral Tablet", "dosage": "Take 10 mg by mouth once daily"}], "vitals": []}
```

**Base model (no adapter) output:**
```
{
  "date_of_birth": "1946-02-19",
  "diagnoses": ["T2DM", "Type 2 diabetes mellitus without complications"],
  "medications": ["Lisinopril 10 MG Oral Tablet"],
  "vitals": {
    "blood_pressure": "135/85 mmHg",
    "heart_rate": "72 bpm"
  }
}
```

**Fine-tuned model (with adapter) output:**
```
{"date_of_birth": "1946-02-19", "diagnoses": [{"name": "Type 2 diabetes mellitus without complications"}], "medications": [{"name": "Lisinopril 10 MG Oral Tablet", "dosage": "Take 10 mg by mouth once daily"}], "vitals": []}
```

## Sample 3

**Instruction:**
```
Patient: [PATIENT NAME], DOB 1953-10-20, Male, MRN [MRN]
Address: [ADDRESS]
Visit Date: 2024-03-20

CC: Here today to discuss HTN.

HPI: Continues management of HTN, first diagnosed 2022-02-12. No acute concerns reported. Patient reports doing well on current regimen for HLD, diagnosed 2024-03-20.

PMH: HTN (dx 2022-02-12), HLD (dx 2024-03-20).

Meds: Metformin 500 MG Oral Tablet - Take 500 mg by mouth twice daily.

VS: HR 59.1, Glu 136.4, Temp 36.4.

A/P: Continue present treatment for HTN; recheck at next visit. Continue present treatment for Hyperlipidemia, unspecified; recheck at next visit.
```

**Ground truth response:**
```json
{"date_of_birth": "1953-10-20", "diagnoses": [{"name": "Essential (primary) hypertension"}, {"name": "Hyperlipidemia, unspecified"}], "medications": [{"name": "Metformin 500 MG Oral Tablet", "dosage": "Take 500 mg by mouth twice daily"}], "vitals": [{"name": "Heart rate", "value": 59.1, "unit": "beats/minute"}, {"name": "Glucose", "value": 136.4, "unit": "mg/dL"}, {"name": "Body temperature", "value": 36.4, "unit": "Cel"}]}
```

**Base model (no adapter) output:**
```
```json
{
  "date_of_birth": "1953-10-20",
  "diagnoses": ["HTN", "HLD"],
  "medications": ["Metformin 500 MG Oral Tablet"],
  "vitals": {
    "HR": "59.1",
    "Glu": "136.4",
    "Temp": "36.4"
  }
}
```
```

**Fine-tuned model (with adapter) output:**
```
{"date_of_birth": "1953-10-20", "diagnoses": [{"name": "Essential (primary) hypertension"}, {"name": "Hyperlipidemia, unspecified"}], "medications": [{"name": "Metformin 500 MG Oral Tablet", "dosage": "Take 500 mg by mouth twice daily"}], "vitals": [{"name": "Heart rate", "value": 59.1, "unit": "beats/minute"}, {"name": "Glucose", "value": 136.4, "unit": "mg/dL"}, {"name": "Body temperature", "value": 36.4, "unit": "Cel"}]}
```
