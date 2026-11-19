# Web application test results

Mode: real ensemble

| Test case | Result |
|---|---|
| Register a new user | PASS |
| Password is stored hashed, not as plain text | PASS |
| Register with an existing username shows an error | PASS |
| Register with a short password shows an error | PASS |
| Register with non-matching passwords shows an error | PASS |
| Log in with a wrong password shows an error | PASS |
| Log in with the correct password works | PASS |
| Predict page without login redirects to the login page | PASS |
| Valid feature row gives a predicted class and confidence | PASS |
| Missing value shows a validation error | PASS |
| Non-numeric value shows a validation error | PASS |
| Unknown category value shows a validation error | PASS |
| Upload a valid CSV gives predictions for each row | PASS |
| Predictions can be downloaded as CSV | PASS |
| Upload of a wrong file type shows an error | PASS |
| Upload with wrong columns shows an error | PASS |
| Upload with non-numeric values shows an error | PASS |
| Upload of an empty file shows an error | PASS |
| Form without a CSRF token is rejected | PASS |
| Log out works | PASS |
| Home page loads | PASS |
| Results page loads | PASS |
| Unknown page shows a 404 page | PASS |
| Same input gives the same prediction in the app and in the notebook | PASS |
