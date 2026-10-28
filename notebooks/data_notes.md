# Data Notes (Day 1 and Day 2)

## UNSW-NB15 (primary)
- Rows (train / test): 175,341 / 82,332 (raw files)
- Rows after cleaning (train / val / test): 91,579 / 16,161 / 82,332
- Columns: 45
- Target columns: label (binary), attack_cat (multi-class)
- Classes: 10 (Normal, Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms)
- Most common class (train, after cleaning): Normal (44,107)
- Smallest classes (train, after cleaning): Worms (108), Shellcode (927), Backdoor (1,305), Analysis (1,355)
- Missing values: 0
- Duplicates: 0 when the id column is counted, but 67,601 duplicate rows in train once id is dropped (removed in Day 2; test duplicates are kept)
- Categorical columns: proto, service, state
- Columns to drop: id

## NSL-KDD (secondary)
- Rows (train / test): 125,973 / 22,544
- Columns: 43 (41 features + label + difficulty)
- Target: label (attack name), grouped into normal / DoS / Probe / R2L / U2R
- Most common group: normal (67,343)
- Smallest groups: U2R (52), R2L (995)
- Missing values: 0
- Duplicates: 0
- Categorical columns: protocol_type, service, flag
- Columns to drop: difficulty

## Decision
- Binary or multi-class: multi-class
- Reason (write in your own words): The model should identify which type of attack it detects (Generic, Exploits, DoS, and so on), rather than only saying attack or normal. This is more useful for a security system and matches the project title. The small classes like Worms (108 training rows) are handled with class weights.

## To tell Member B
- Class names: Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Normal, Reconnaissance, Shellcode, Worms (10 classes)
- Number of features: 42 raw input features; 30 are kept after Mutual Information feature selection