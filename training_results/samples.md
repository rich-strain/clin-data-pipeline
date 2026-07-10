# Before/after generation samples

## Sample 1

**Instruction:**
```
Patient: [PATIENT NAME], DOB 1941-11-12, Female, MRN [MRN]
Visit Date: 2024-05-25

CC: Presents for routine follow-up of obesity.

PMH: obesity (dx 2024-05-25), COPD (dx 2022-11-05), GERD (dx 2016-07-23).

Meds: Atorvastatin 20 MG Oral Tablet - Take 20 mg by mouth at bedtime; Atorvastatin 20 MG Oral Tablet - 20mg PO QHS.

VS: Glu 170.8, BP 134.1 mm[Hg], Temp 37.2.

A/P: Continue current management of obesity. Continue current management of COPD. No changes to GERD management at this time.
```

**Ground truth response:**
```json
{"date_of_birth": "1941-11-12", "diagnoses": [{"name": "Obesity, unspecified"}, {"name": "Chronic obstructive pulmonary disease, unspecified"}, {"name": "Gastro-esophageal reflux disease without esophagitis"}], "medications": [{"name": "Atorvastatin 20 MG Oral Tablet", "dosage": "Take 20 mg by mouth at bedtime"}], "vitals": [{"name": "Glucose", "value": 170.8, "unit": "mg/dL"}, {"name": "Systolic blood pressure", "value": 134.1, "unit": "mm[Hg]"}, {"name": "Body temperature", "value": 37.2, "unit": "Cel"}]}
```

**Base model (no adapter) output:**
```
```json
{
  "date_of_birth": "1941-11-12",
  "diagnoses": ["obesity", "COPD", "GERD"],
  "medications": ["Atorvastatin 20 MG Oral Tablet - Take 20 mg by mouth at bedtime; Atorvastatin 20 MG Oral Tablet - 20mg PO QHS"],
  "vitals": {
    "glucose": 170.8,
    "blood_pressure": 134.1 mm[Hg],
    "temperature": 37.2
  }
}
```
```

**Fine-tuned model (with adapter) output:**
```
{"date_of_birth": "1941-11-12", "diagnoses": [{"name": "Obesity"}, {"name": "Chronic obstructive pulmonary disease without pneumonia"}, {"name": "Gastro-esophageal reflux disease without esophagitis"}], "medications": [{"name": "Atorvastatin 20 MG Oral Tablet", "dosage": "Take 20 mg by mouth at bedtime; Atorvastatin 20 MG Oral Tablet, 20 mg oral, take 20 mg by mouth every night"}], "vitals": [{"name": "Glucose", "value": 170.8, "unit": "mg/dL"}, {"name": "Systolic blood pressure", "value": 134.1, "unit": "mm[Hg]"}, {"name": "Body temperature", "value": 37.2, "unit": "Cel"}]}
```

## Sample 2

**Instruction:**
```
Patient: [PATIENT NAME], DOB 1979-03-13, Male, MRN [MRN]
Address: [ADDRESS]
Visit Date: 2023-12-02

CC: Presents for routine follow-up of HTN.

PMH: Essential (primary) hypertension (dx 2020-01-03).

Meds: Atorvastatin 20 MG Oral Tablet - Take 20 mg by mouth at bedtime; Sertraline 50 MG Oral Tablet - dosage not recorded; Metformin 500 MG Oral Tablet - 500mg PO BID.

A/P: Continue current management of HTN.
```

**Ground truth response:**
```json
{"date_of_birth": "1979-03-13", "diagnoses": [{"name": "Essential (primary) hypertension"}], "medications": [{"name": "Atorvastatin 20 MG Oral Tablet", "dosage": "Take 20 mg by mouth at bedtime"}, {"name": "Sertraline 50 MG Oral Tablet", "dosage": null}, {"name": "Metformin 500 MG Oral Tablet", "dosage": "Take 500 mg by mouth twice daily"}], "vitals": []}
```

**Base model (no adapter) output:**
```
```json
{
  "date_of_birth": "1979-03-13",
  "diagnoses": ["HTN"],
  "medications": {
    "Atorvastatin": {
      "Dosage": "20 MG Oral Tablet"
    },
    "Sertraline": {},
    "Metformin": {}
  },
  "vitals": {
    "Blood Pressure": "140/80 mmHg"
  }
}
```
```

**Fine-tuned model (with adapter) output:**
```
{"date_of_birth": "1979-03-13", "diagnoses": [{"name": "Essential (primary) hypertension"}], "medications": [{"name": "Atorvastatin 20 MG Oral Tablet", "dosage": "Take 20 mg by mouth at bedtime"}, {"name": "Sertraline 50 MG Oral Tablet", "dosage": null}, {"name": "Metformin 500 MG Oral Tablet", "dosage": "Take 500 mg by mouth twice daily"}]}
```

## Sample 3

**Instruction:**
```
Patient: [PATIENT NAME], DOB 1949-07-06, Female, MRN [MRN]
Address: [ADDRESS]
Visit Date: 2023-06-18

CC: Here today to discuss MDD.

HPI: Patient reports doing well on current regimen for MDD, diagnosed 2019-01-10. Unspecified asthma, uncomplicated (dx 2023-06-18) remains stable; no new complaints today.

PMH: MDD (dx 2019-01-10), asthma (dx 2023-06-18).

Meds: Amlodipine 5 MG Oral Tablet - Take 5 mg by mouth once daily; Omeprazole 20 MG Oral Capsule - Take 20 mg by mouth once daily; Omeprazole 20 MG Oral Capsule - dosage not recorded.

VS: BP 130.5 mm[Hg], Wt 82.8 kg, HR 57.3 beats/minute.

A/P: Continue current management of MDD. No changes to asthma management at this time.
```

**Ground truth response:**
```json
{"date_of_birth": "1949-07-06", "diagnoses": [{"name": "Major depressive disorder, single episode, unspecified"}, {"name": "Unspecified asthma, uncomplicated"}], "medications": [{"name": "Amlodipine 5 MG Oral Tablet", "dosage": "Take 5 mg by mouth once daily"}, {"name": "Omeprazole 20 MG Oral Capsule", "dosage": "Take 20 mg by mouth once daily"}, {"name": "Omeprazole 20 MG Oral Capsule", "dosage": null}], "vitals": [{"name": "Systolic blood pressure", "value": 130.5, "unit": "mm[Hg]"}, {"name": "Body weight", "value": 82.8, "unit": "kg"}, {"name": "Heart rate", "value": 57.3, "unit": "beats/minute"}]}
```

**Base model (no adapter) output:**
```
```json
{
  "date_of_birth": "1949-07-06",
  "diagnoses": ["MDD", "asthma"],
  "medications": {
    "amlodipine": {
      "Dosage": "5 mg by mouth once daily"
    },
    "omeprazole": {
      "Dosage": "20 mg by mouth once daily"
    }
  },
  "vitals": {
    "BP": "130.5 mm[Hg]",
    "Wt": "82.8 kg",
    "HR": "57.3 beats/minute"
  }
}
```
```

**Fine-tuned model (with adapter) output:**
```
{"date_of_birth": "1949-07-06", "diagnoses": [{"name": "Major depressive disorder, single episode, unspecified"}, {"name": "Unspecified asthma, uncomplicated"}], "medications": [{"name": "Amlodipine 5 MG Oral Tablet", "dosage": "Take 5 mg by mouth once daily"}, {"name": "Omeprazole 20 MG Oral Capsule", "dosage": null}, {"name": "Omeprazole 20 MG Oral Capsule", "dosage": "Take 20 mg by mouth once daily"}], "vitals": [{"name": "Systolic blood pressure", "value": 130.5, "unit": "mm[Hg]"}, {"name": "Body weight", "value": 58.8, "unit": "kg"}, {"name": "Heart rate", "value": 57.3, "unit": "beats/minute"}]}
```
