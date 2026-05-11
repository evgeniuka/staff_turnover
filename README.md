# Staff Turnover Data Cleaning

This is a small HR data-cleaning project. It takes two employee CSV files, joins exit reasons onto the main employee table, standardizes missing values and dates, and produces a cleaner `employee_clean.csv` file that can be used for turnover analysis.

The project is intentionally simple, but useful: it shows a realistic data preparation step before dashboarding or analysis.

## What It Does

- Reads the main employee file: `employee_data_Ch.csv`.
- Reads the exit-reason lookup: `employee_data_exit_view.csv`.
- Normalizes `EmpID` so both files can be joined safely.
- Adds `FireReason` and `ExitManager` to the employee table.
- Marks employees with a fire/exit reason as `Exited`.
- Standardizes `DOB`, `StartDate`, and `ExitDate` to `YYYY-MM-DD`.
- Removes duplicate employee rows by name and date of birth.
- Writes the final output to `employee_clean.csv`.

## Files

```text
main.py                       cleaning script and CLI
employee_data_Ch.csv           main employee dataset
employee_data_exit_view.csv    exit reason lookup
employee_clean.csv             generated cleaned output
tests/                         small regression tests for the cleaning logic
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py --summary
```

The script rewrites `employee_clean.csv` and prints a short summary.

## Example Summary

```text
Rows: 3,000
Columns: 28

Employee status:
Exited    1533
Active    1467
```

## Resume Framing

Good one-liner:

> Cleaned and merged HR employee datasets with pandas to prepare staff turnover data for analysis.

What I would mention in an interview:

- I kept the original supervisor data instead of overwriting it with synthetic values.
- I separated data cleaning into small reusable functions.
- I added tests around merge behavior, status updates, date formatting, and deduplication.

## Next Improvements

- Add a notebook or dashboard with turnover rates by department, manager, location, and performance score.
- Add anonymization if this dataset ever contains real employee information.
- Add validation checks for impossible dates and missing key fields.
