## Results on the test data

| Model | Accuracy | Precision (w) | Recall (w) | F1 (w) | F1 (macro) |
|---|---|---|---|---|---|
| DNN | 0.651 | 0.846 | 0.651 | 0.706 | 0.414 |
| CNN | 0.659 | 0.837 | 0.659 | 0.708 | 0.422 |
| RNN-LSTM | 0.636 | 0.816 | 0.636 | 0.679 | 0.359 |
| Ensemble | 0.658 | 0.849 | 0.658 | 0.708 | 0.433 |

## Results on the validation data

| Model | Accuracy | Precision (w) | Recall (w) | F1 (w) | F1 (macro) |
|---|---|---|---|---|---|
| DNN | 0.759 | 0.837 | 0.759 | 0.782 | 0.541 |
| CNN | 0.770 | 0.833 | 0.770 | 0.788 | 0.551 |
| RNN-LSTM | 0.720 | 0.790 | 0.720 | 0.736 | 0.469 |
| Ensemble | 0.766 | 0.836 | 0.766 | 0.783 | 0.553 |

## F1 per class (test)

| Class | DNN | CNN | RNN-LSTM | Ensemble |
|---|---|---|---|---|
| Analysis | 0.034 | 0.059 | 0.125 | 0.048 |
| Backdoor | 0.109 | 0.079 | 0.095 | 0.102 |
| DoS | 0.121 | 0.109 | 0.039 | 0.098 |
| Exploits | 0.650 | 0.657 | 0.598 | 0.665 |
| Fuzzers | 0.342 | 0.327 | 0.354 | 0.344 |
| Generic | 0.977 | 0.980 | 0.979 | 0.981 |
| Normal | 0.732 | 0.742 | 0.728 | 0.733 |
| Reconnaissance | 0.755 | 0.694 | 0.388 | 0.736 |
| Shellcode | 0.208 | 0.255 | 0.151 | 0.234 |
| Worms | 0.210 | 0.320 | 0.127 | 0.391 |
