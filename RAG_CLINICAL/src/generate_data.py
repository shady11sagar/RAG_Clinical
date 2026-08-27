import json, random, os, re

random.seed(42)
OUT = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Condition-specific sentence banks (all synthetic / authored, not copied)
# ---------------------------------------------------------------------------
CONDITIONS = {
    "hypertension": {
        "subjective": [
            "I've been having occasional headaches in the morning.",
            "I feel a bit dizzy when I stand up quickly.",
            "No chest pain, but I do feel tired more than usual.",
            "I ran out of my blood pressure medication about two weeks ago.",
            "I've been under a lot of stress at work lately.",
            "I haven't been checking my blood pressure at home.",
        ],
        "objective": [
            "Blood pressure today is {sbp}/{dbp} mmHg, heart rate {hr} bpm.",
            "Repeat blood pressure reading after five minutes rest is {sbp2}/{dbp2} mmHg.",
            "Cardiovascular exam reveals regular rate and rhythm, no murmurs.",
            "No lower extremity edema noted on exam.",
            "Weight is {weight} lbs, BMI {bmi}.",
        ],
        "guideline_ids": ["htn_1", "htn_2"],
        "assessment": "Essential hypertension, currently {control} controlled on current regimen.",
        "plan_grounded": "Continue lifestyle modification (sodium restriction, exercise) and {med_change}, per first-line antihypertensive guidance for stage {stage} hypertension.",
    },
    "diabetes": {
        "subjective": [
            "My blood sugar readings at home have been running a bit high.",
            "I've noticed increased thirst over the past month.",
            "No numbness or tingling in my feet.",
            "I've been trying to follow the diet plan but it's been difficult.",
            "I missed a few doses of my metformin this month.",
        ],
        "objective": [
            "Fasting glucose today is {glucose} mg/dL.",
            "Hemoglobin A1c is {a1c}%.",
            "Foot exam shows intact sensation bilaterally, no ulcers.",
            "Weight is {weight} lbs, unchanged from last visit.",
        ],
        "guideline_ids": ["dm_1", "dm_2"],
        "assessment": "Type 2 diabetes mellitus, {control} controlled with A1c of {a1c}%.",
        "plan_grounded": "{med_change}, reinforce dietary counseling, and recheck A1c in 3 months, consistent with ADA glycemic-target guidance.",
    },
    "uri": {
        "subjective": [
            "I've had a runny nose and sore throat for about four days.",
            "No fever at home, just some nasal congestion.",
            "I have a mild cough, mostly dry.",
            "No shortness of breath.",
            "My symptoms seem to be improving slightly since yesterday.",
        ],
        "objective": [
            "Temperature is {temp} F, oxygen saturation {spo2}% on room air.",
            "Oropharynx is mildly erythematous without exudate.",
            "Lungs are clear to auscultation bilaterally.",
            "No cervical lymphadenopathy noted.",
        ],
        "guideline_ids": ["uri_1", "uri_2"],
        "assessment": "Acute uncomplicated viral upper respiratory infection.",
        "plan_grounded": "Symptomatic management with fluids and rest; antibiotics are not indicated given absence of bacterial features, consistent with URI antibiotic-stewardship guidance.",
    },
    "back_pain": {
        "subjective": [
            "I strained my lower back lifting something heavy three days ago.",
            "The pain is worse with bending forward.",
            "No numbness or weakness in my legs.",
            "No loss of bowel or bladder control.",
            "Over-the-counter ibuprofen has helped a little.",
        ],
        "objective": [
            "Lumbar paraspinal tenderness noted on palpation.",
            "Straight leg raise test is negative bilaterally.",
            "Lower extremity strength and sensation are intact.",
            "Gait is normal.",
        ],
        "guideline_ids": ["back_1", "back_2"],
        "assessment": "Acute nonspecific low back pain without red-flag features.",
        "plan_grounded": "Continue NSAIDs as needed, encourage early mobilization, and avoid routine imaging in the absence of red flags, per acute low back pain guidance.",
    },
    "asthma": {
        "subjective": [
            "I've been using my rescue inhaler more often this week.",
            "I wake up coughing about twice a week.",
            "No recent emergency room visits.",
            "Pollen season seems to be triggering my symptoms.",
        ],
        "objective": [
            "Peak flow today is {peakflow} L/min, {pct}% of personal best.",
            "Mild expiratory wheeze noted bilaterally.",
            "Oxygen saturation is {spo2}% on room air.",
            "Respiratory rate is {rr} breaths per minute.",
        ],
        "guideline_ids": ["asthma_1", "asthma_2"],
        "assessment": "Mild persistent asthma, currently {control} controlled.",
        "plan_grounded": "{med_change} inhaled corticosteroid therapy and reassess symptom control in 4 weeks, per stepwise asthma management guidance.",
    },
    "gerd": {
        "subjective": [
            "I've had heartburn most nights over the past two weeks.",
            "Symptoms are worse after eating spicy or fatty food.",
            "No difficulty swallowing.",
            "No unintentional weight loss.",
        ],
        "objective": [
            "Abdomen is soft, non-tender, non-distended.",
            "No epigastric tenderness on palpation.",
            "Weight is {weight} lbs, stable.",
        ],
        "guideline_ids": ["gerd_1", "gerd_2"],
        "assessment": "Gastroesophageal reflux disease, symptomatic.",
        "plan_grounded": "Start a trial of once-daily proton pump inhibitor and recommend lifestyle modification (weight management, avoiding late meals), per GERD management guidance.",
    },
    "uti": {
        "subjective": [
            "I've had burning with urination for the past two days.",
            "I've also noticed increased urinary frequency.",
            "No fever or chills.",
            "No flank pain.",
        ],
        "objective": [
            "Temperature is {temp} F.",
            "No costovertebral angle tenderness.",
            "Urinalysis shows leukocyte esterase and nitrites positive.",
        ],
        "guideline_ids": ["uti_1", "uti_2"],
        "assessment": "Uncomplicated acute cystitis.",
        "plan_grounded": "Start a short course of first-line oral antibiotic therapy per local antibiogram and uncomplicated-UTI treatment guidance.",
    },
    "anxiety": {
        "subjective": [
            "I've been feeling more anxious and having trouble sleeping.",
            "My worries feel hard to control most days.",
            "No thoughts of self-harm.",
            "Symptoms have been present for at least six weeks now.",
        ],
        "objective": [
            "Affect is mildly anxious but appropriate.",
            "Speech is normal rate and rhythm.",
            "PHQ-9 score is {phq9}, GAD-7 score is {gad7}.",
        ],
        "guideline_ids": ["anx_1", "anx_2"],
        "assessment": "Generalized anxiety disorder, {control} controlled.",
        "plan_grounded": "{med_change} and referral for cognitive behavioral therapy, consistent with first-line generalized anxiety disorder management guidance.",
    },
}

# ---------------------------------------------------------------------------
# 2. Guideline knowledge base (short, paraphrased general clinical knowledge —
#    authored for this project, not copied verbatim from any single source)
# ---------------------------------------------------------------------------
GUIDELINES = [
    {"id": "htn_1", "condition": "hypertension", "text": "First-line therapy for stage 1 hypertension without compelling indications includes thiazide diuretics, ACE inhibitors, ARBs, or calcium channel blockers, alongside sodium restriction and aerobic exercise."},
    {"id": "htn_2", "condition": "hypertension", "text": "For stage 2 hypertension or blood pressure persistently above goal on monotherapy, combination therapy with two agents from different classes is recommended, with home blood pressure monitoring to confirm control."},
    {"id": "dm_1", "condition": "diabetes", "text": "Metformin remains first-line pharmacotherapy for type 2 diabetes; an A1c goal of below 7% is appropriate for most non-pregnant adults without significant comorbidity."},
    {"id": "dm_2", "condition": "diabetes", "text": "When A1c remains above target despite maximized metformin and lifestyle therapy, addition of a second agent (e.g., GLP-1 receptor agonist or SGLT2 inhibitor) should be considered, particularly with cardiovascular or renal comorbidity."},
    {"id": "uri_1", "condition": "uri", "text": "Antibiotics are not indicated for uncomplicated viral upper respiratory infections; management is supportive, including fluids, rest, and symptomatic treatment."},
    {"id": "uri_2", "condition": "uri", "text": "Red-flag features warranting further evaluation of a respiratory complaint include high fever beyond several days, shortness of breath, or focal exam findings suggestive of pneumonia."},
    {"id": "back_1", "condition": "back_pain", "text": "For acute low back pain without red-flag features (no trauma, no neurologic deficit, no bowel/bladder involvement), routine imaging is not recommended in the first six weeks."},
    {"id": "back_2", "condition": "back_pain", "text": "First-line management of acute nonspecific low back pain includes NSAIDs, encouragement of normal activity as tolerated, and avoidance of prolonged bed rest."},
    {"id": "asthma_1", "condition": "asthma", "text": "Stepwise asthma management escalates inhaled corticosteroid dose or adds a long-acting beta agonist when symptom control or rescue-inhaler use indicates inadequate control."},
    {"id": "asthma_2", "condition": "asthma", "text": "Peak expiratory flow below 80% of personal best, or rescue inhaler use more than twice weekly, suggests inadequately controlled asthma requiring step-up therapy."},
    {"id": "gerd_1", "condition": "gerd", "text": "A once-daily proton pump inhibitor trial for 4 to 8 weeks is first-line therapy for symptomatic GERD without alarm features."},
    {"id": "gerd_2", "condition": "gerd", "text": "Alarm features for reflux symptoms warranting endoscopic evaluation include dysphagia, unintentional weight loss, or gastrointestinal bleeding."},
    {"id": "uti_1", "condition": "uti", "text": "Uncomplicated acute cystitis in otherwise healthy patients is typically treated with a short course of nitrofurantoin, trimethoprim-sulfamethoxazole, or fosfomycin, guided by local resistance patterns."},
    {"id": "uti_2", "condition": "uti", "text": "Flank pain, fever, or costovertebral angle tenderness suggest possible pyelonephritis and warrant broader evaluation and treatment rather than management as uncomplicated cystitis."},
    {"id": "anx_1", "condition": "anxiety", "text": "First-line pharmacotherapy for generalized anxiety disorder includes SSRIs or SNRIs, typically alongside cognitive behavioral therapy referral."},
    {"id": "anx_2", "condition": "anxiety", "text": "Screening tools such as the GAD-7 can help quantify anxiety symptom severity and monitor response to treatment over time."},
]

OTHER_SENTENCES = [
    "Thanks for coming in today, how have you been overall?",
    "Do you have any questions before we wrap up?",
    "Let's go ahead and schedule your follow-up for next month.",
    "The front desk will help you with your next appointment.",
    "Any changes to your insurance information since your last visit?",
    "It's good to see you again.",
    "I'll send the referral over to the specialist's office.",
    "How has your family been doing lately?",
    "Did you find parking okay today?",
    "We'll get you checked out at the front desk.",
    "Let me know if you need a note for work.",
    "Take care, and we'll see you at the next visit.",
    "Is there a good number to reach you for lab results?",
    "The nurse will be in shortly to draw some blood.",
    "Do you have any other concerns before you go?",
]

# ---------------------------------------------------------------------------
# 3. Slot-filling ranges for realistic numeric variation
# ---------------------------------------------------------------------------
def rand_vitals():
    return {
        "sbp": random.randint(128, 168), "dbp": random.randint(78, 100),
        "sbp2": random.randint(122, 150), "dbp2": random.randint(74, 92),
        "hr": random.randint(62, 96), "weight": random.randint(140, 230),
        "bmi": round(random.uniform(23.0, 34.0), 1),
        "glucose": random.randint(110, 210), "a1c": round(random.uniform(6.8, 9.4), 1),
        "temp": round(random.uniform(97.8, 100.9), 1), "spo2": random.randint(95, 99),
        "peakflow": random.randint(280, 480), "pct": random.randint(62, 92),
        "rr": random.randint(14, 22), "phq9": random.randint(6, 16), "gad7": random.randint(6, 16),
    }

def stage_and_control(sbp):
    if sbp >= 160: return "2", "poorly"
    if sbp >= 140: return "2", "moderately"
    return "1", "well"

FAKE_FIRST = ["James","Maria","Robert","Linda","David","Susan","Michael","Karen","John","Nancy",
              "Thomas","Patricia","Charles","Barbara","Daniel","Jessica","Mark","Sandra","Paul","Betty"]
FAKE_LAST = ["Nguyen","Smith","Garcia","Johnson","Williams","Brown","Davis","Miller","Wilson","Moore",
             "Taylor","Anderson","Thomas","Jackson","White","Harris","Martin","Thompson","Young","Clark"]

def fake_name():
    return f"{random.choice(FAKE_FIRST)} {random.choice(FAKE_LAST)}"

def fake_phone():
    return f"({random.randint(200,989)}) {random.randint(200,989)}-{random.randint(1000,9999)}"

def fake_mrn():
    return f"MRN{random.randint(100000,999999)}"

def fake_ssn():
    return f"{random.randint(100,899)}-{random.randint(10,99)}-{random.randint(1000,9999)}"

def fake_date():
    return f"{random.randint(1,12)}/{random.randint(1,28)}/{random.choice(['2023','2024','2025'])}"

def fake_address():
    return f"{random.randint(100,9999)} {random.choice(['Oak','Maple','5th','River','Elm','Sunset'])} {random.choice(['St','Ave','Rd','Ln'])}, {random.choice(['Tuscaloosa','Birmingham','Huntsville','Mobile'])}, AL"

# ---------------------------------------------------------------------------
# 4. Build labeled sentence bank for the SOAP-section classifier
# ---------------------------------------------------------------------------
labeled_rows = []
for cond, bank in CONDITIONS.items():
    for s in bank["subjective"]:
        labeled_rows.append((s, "Subjective"))
    for o in bank["objective"]:
        v = rand_vitals()
        labeled_rows.append((o.format(**v), "Objective"))
    v = rand_vitals()
    stage, control = stage_and_control(v["sbp"])
    labeled_rows.append((bank["assessment"].format(control=control, a1c=v["a1c"]), "Assessment"))
    labeled_rows.append((bank["plan_grounded"].format(
        med_change=random.choice(["increase the current dose", "start an additional agent", "switch to an alternative agent"]),
        stage=stage), "Plan"))
for extra in OTHER_SENTENCES:
    labeled_rows.append((extra, "Other"))
    labeled_rows.append((extra, "Other"))  # light oversampling of minority class

# augment with several randomized re-fills to get a reasonably sized training set
for _ in range(6):
    for cond, bank in CONDITIONS.items():
        for o in bank["objective"]:
            v = rand_vitals()
            labeled_rows.append((o.format(**v), "Objective"))
        v = rand_vitals()
        stage, control = stage_and_control(v["sbp"])
        labeled_rows.append((bank["assessment"].format(control=control, a1c=v["a1c"]), "Assessment"))
        labeled_rows.append((bank["plan_grounded"].format(
            med_change=random.choice(["increase the current dose", "start an additional agent", "switch to an alternative agent", "continue the current regimen"]),
            stage=stage), "Plan"))

random.shuffle(labeled_rows)

import csv
with open(os.path.join(OUT, "sentences_labeled.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["text", "label"])
    for text, label in labeled_rows:
        w.writerow([text, label])

# ---------------------------------------------------------------------------
# 5. Build synthetic encounters (dialogue transcript + gold SOAP note)
# ---------------------------------------------------------------------------
encounters = []
enc_id = 0
for cond, bank in CONDITIONS.items():
    for variant in range(7):
        v = rand_vitals()
        stage, control = stage_and_control(v["sbp"])
        med_change = random.choice(["increase the current dose", "start an additional agent",
                                     "switch to an alternative agent", "continue the current regimen"])
        subj_lines = random.sample(bank["subjective"], k=min(3, len(bank["subjective"])))
        obj_lines = [o.format(**v) for o in bank["objective"]]
        assessment = bank["assessment"].format(control=control, a1c=v["a1c"])
        plan = bank["plan_grounded"].format(med_change=med_change, stage=stage)

        transcript = []
        transcript.append({"speaker": "Doctor", "text": "What brings you in today, and how have things been?"})
        for s in subj_lines:
            transcript.append({"speaker": "Patient", "text": s})
        transcript.append({"speaker": "Doctor", "text": "Okay, let's take a look and check a few things."})
        for o in obj_lines:
            transcript.append({"speaker": "Doctor", "text": o})
        transcript.append({"speaker": "Doctor", "text": random.choice(OTHER_SENTENCES)})

        encounters.append({
            "id": f"enc_{enc_id:03d}",
            "condition": cond,
            "transcript": transcript,
            "gold_soap": {
                "subjective": subj_lines,
                "objective": obj_lines,
                "assessment": assessment,
                "plan": plan,
            },
            "guideline_ids": bank["guideline_ids"],
        })
        enc_id += 1

random.shuffle(encounters)
split = int(len(encounters) * 0.75)
train_enc, test_enc = encounters[:split], encounters[split:]

with open(os.path.join(OUT, "encounters.json"), "w") as f:
    json.dump({"train": train_enc, "test": test_enc}, f, indent=2)

with open(os.path.join(OUT, "guidelines.json"), "w") as f:
    json.dump(GUIDELINES, f, indent=2)

# ---------------------------------------------------------------------------
# 6. Synthetic PHI test set for de-identification evaluation
#    Each example: raw text with embedded fake PHI + list of gold PHI strings
# ---------------------------------------------------------------------------
phi_examples = []
for _ in range(50):
    name = fake_name()
    phone = fake_phone()
    mrn = fake_mrn()
    date = fake_date()
    templates = [
        f"Patient {name} (MRN {mrn}) was seen in clinic on {date} for follow-up.",
        f"Please call {name} back at {phone} to confirm the {date} appointment.",
        f"{name}'s chart, medical record number {mrn}, was reviewed prior to the visit on {date}.",
        f"Contact number on file for {name} is {phone}; date of birth is {date}.",
    ]
    text = random.choice(templates)
    gold_phi = [name, phone, mrn, date]
    gold_phi = [g for g in gold_phi if g in text]
    phi_examples.append({"text": text, "phi_spans": gold_phi})

# add SSN and address variants too
for _ in range(15):
    name = fake_name(); ssn = fake_ssn(); addr = fake_address()
    text = f"{name} (SSN {ssn}) resides at {addr}."
    phi_examples.append({"text": text, "phi_spans": [name, ssn, addr]})

with open(os.path.join(OUT, "phi_test.json"), "w") as f:
    json.dump(phi_examples, f, indent=2)

print(f"Encounters: train={len(train_enc)} test={len(test_enc)}")
print(f"Labeled sentences: {len(labeled_rows)}")
print(f"Guidelines: {len(GUIDELINES)}")
print(f"PHI test examples: {len(phi_examples)}")
