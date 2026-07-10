# Before/after generation samples

## Sample 1

**Instruction:**
```
Patient: [PATIENT NAME], DOB 1996-06-03, Female, MRN [MRN]
Address: [ADDRESS]
Visit Date: 2026-01-10

CC: Presents for routine follow-up of obesity.

HPI: obesity (dx 2021-05-11) remains stable; no new complaints today.

PMH: obesity (dx 2021-05-11).

Meds: Sertraline 50 MG Oral Tablet - 50mg PO QD; Atorvastatin 20 MG Oral Tablet - Take 20 mg by mouth at bedtime.

VS: Wt 236.6, Glu 80.2 mg/dL, Ht 165.9.

A/P: Continue current management of obesity.
```

**Ground truth response:**
```json
{"date_of_birth": "1996-06-03", "diagnoses": [{"name": "Obesity, unspecified"}], "medications": [{"name": "Sertraline 50 MG Oral Tablet", "dosage": "Take 50 mg by mouth once daily"}, {"name": "Atorvastatin 20 MG Oral Tablet", "dosage": "Take 20 mg by mouth at bedtime"}], "vitals": [{"name": "Body weight", "value": 107.3, "unit": "kg"}, {"name": "Glucose", "value": 80.2, "unit": "mg/dL"}, {"name": "Body height", "value": 165.9, "unit": "cm"}]}
```

**Base model (no adapter) output:**
```
```json
{
  "date_of_birth": "1996-06-03",
  "diagnoses": ["obesity"],
  "medications": ["Sertraline", "Atorvastatin"],
  "vitals": {
    "weight": "236.6 kg",
    "blood_pressure": "80.2 mmHg",
    "heart_rate": "110 bpm"
  }
}
```
```

**Fine-tuned model (with adapter) output:**
```
{"date_of_birth": "1996-06-03", "diagnoses": [{"name": "Obesity"}], "medications": [{"name": "Sertraline 50 MG Oral Tablet", "dosage": "Take 50 mg by mouth once daily"}, {"name": "Atorvastatin 20 MG Oral Tablet", "dosage": "Take 20 mg by mouth at bedtime"}], "vitals": [{"name": "Body weight", "value": 236.6, "unit": "kg"}, {"name": "Glucose", "value": 80.2, "unit": "mg/dL"}, {"name": "Body height", "value": 165.9, "unit": "cm"}]}
```

## Sample 2

**Instruction:**
```
Patient: [PATIENT NAME], DOB 1996-06-10, Male, MRN [MRN]
Address: [ADDRESS]
Visit Date: 2024-10-02

CC: Follow-up visit for Essential (primary) hypertension management.

HPI: Patient reports doing well on current regimen for HTN, diagnosed 2024-10-02.

PMH: HTN (dx 2024-10-02).

Meds: Lisinopril 10 MG Oral Tablet - dosage not recorded.

VS: Wt 55.0, BP 135.3, Temp 97.3, Glu 88.2 mg/dL.

A/P: Continue present treatment for Essential (primary) hypertension; recheck at next visit.
```

**Ground truth response:**
```json
{"date_of_birth": "1996-06-10", "diagnoses": [{"name": "Essential (primary) hypertension"}], "medications": [{"name": "Lisinopril 10 MG Oral Tablet", "dosage": null}], "vitals": [{"name": "Body weight", "value": 55.0, "unit": "kg"}, {"name": "Systolic blood pressure", "value": 135.3, "unit": "mm[Hg]"}, {"name": "Body temperature", "value": 36.3, "unit": "Cel"}, {"name": "Glucose", "value": 88.2, "unit": "mg/dL"}]}
```

**Base model (no adapter) output:**
```
```json
{
  "date_of_birth": "1996-06-10",
  "diagnoses": ["Essential (primary)", "HTN"],
  "medications": ["Lisinopril", "No medication recorded"],
  "vitals": {
    "weight": "55.0 kg",
    "blood_pressure": {
      "systolic": "135.3 mmHg",
      "diastolic": "97.3 mmHg"
    },
    "temperature": "97.3 °C",
    "glucose": "88.2 mg/dL"
  }
}
```
```

**Fine-tuned model (with adapter) output:**
```
{"date_of_birth": "1996-06-10", "diagnoses": [{"name": "Essential (primary) hypertension"}], "medications": [{"name": "Lisinopril 10 MG Oral Tablet", "dosage": null}], "vitals": [{"name": "Body weight", "value": 55.0, "unit": "kg"}, {"name": "Systolic blood pressure", "value": 135.3, "unit": "mm[Hg]"}, {"name": "Body temperature", "value": 37.4, "unit": "Cel"}, {"name": "Glucose", "value": 88.2, "unit": "mg/dL"}]}
```

## Sample 3

**Instruction:**
```
Patient: [PATIENT NAME], DOB 1941-08-21, Male, MRN [MRN]
Address: [ADDRESS]
Visit Date: 2024-08-07

CC: Here today to discuss obesity.

HPI: obesity (dx 2022-06-11) remains stable; no new complaints today.

PMH: obesity (dx 2022-06-11).

Meds: Atorvastatin 20 MG Oral Tablet - dosage not recorded; Metformin 500 MG Oral Tablet - Take 500 mg by mouth twice daily.

VS: BP 115.3 mm[Hg], HR 69.1.

A/P: Continue current management of obesity.
```

**Ground truth response:**
```json
{"date_of_birth": "1941-08-21", "diagnoses": [{"name": "Obesity, unspecified"}], "medications": [{"name": "Atorvastatin 20 MG Oral Tablet", "dosage": null}, {"name": "Metformin 500 MG Oral Tablet", "dosage": "Take 500 mg by mouth twice daily"}], "vitals": [{"name": "Systolic blood pressure", "value": 115.3, "unit": "mm[Hg]"}, {"name": "Heart rate", "value": 69.1, "unit": "beats/minute"}]}
```

**Base model (no adapter) output:**
```
```json
{
  "date_of_birth": "1941-08-21",
  "diagnoses": ["obesity"],
  "medications": {
    "Atorvastatin": null,
    "Metformin": {
      "Dosage": null
    }
  },
  "vitals": {
    "BP": "115.3 mm[Hg]",
    "HR": "69.1"
  }
}
```
```

**Fine-tuned model (with adapter) output:**
```
{"date_of_birth": "1941-08-21", "diagnoses": [{"name": "Obesity"}], "medications": [{"name": "Atorvastatin 20 MG Oral Tablet", "dosage": null}, {"name": "Metformin 500 MG Oral Tablet", "dosage": "Take 500 mg by mouth twice daily"}], "vitals": [{"name": "Systolic blood pressure", "value": 115.3, "unit": "mm[Hg]"}, {"name": "Heart rate", "value": 69.1, "unit": "beats/minute"}]}
```
