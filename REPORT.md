# Staff Turnover Cleaning Notes

This project prepares a cleaned HR dataset for staff turnover analysis.

## Current Output

After cleaning, the output file has:

- 3,000 employee rows
- 28 columns
- 1,533 rows marked as `Exited`
- 1,467 rows marked as `Active`

The generated file is:

`employee_clean.csv`

## Cleaning Decisions

- `EmpID` is converted to a numeric nullable integer before merging.
- Exit reasons are joined from `employee_data_exit_view.csv`.
- `Manager` from the exit file is kept as `ExitManager`.
- Original `Supervisor` values are preserved. Missing supervisors are filled from `ExitManager` only when possible.
- `DOB`, `StartDate`, and `ExitDate` are standardized to `YYYY-MM-DD`.
- Duplicate people are removed using `FirstName`, `LastName`, and `DOB`.

## Why This Matters

This is not a prediction model yet. It is the data preparation step that would come before a turnover dashboard or analysis. For a resume, it is best framed as a pandas cleaning and ETL project.

