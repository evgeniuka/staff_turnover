import argparse
from pathlib import Path

import pandas as pd


BASE = Path(".")
EMPLOYEE_SRC = BASE / "employee_data_Ch.csv"
EXIT_SRC = BASE / "employee_data_exit_view.csv"
OUTPUT_CSV = BASE / "employee_clean.csv"

NA_LIKE = {"", "na", "n/a", "none", "null", "-", "--", "nan"}


def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text_columns = [
        column
        for column in df.columns
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column])
    ]
    for column in text_columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .replace({value: pd.NA for value in NA_LIKE}, regex=False)
        )
    return df


def normalize_emp_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EmpID"] = pd.to_numeric(df["EmpID"], errors="coerce").astype("Int64")
    return df


def format_date(series: pd.Series, date_format: str) -> pd.Series:
    parsed = pd.to_datetime(series, format=date_format, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d").astype("string").replace("<NA>", pd.NA)


def build_exit_lookup(exit_df: pd.DataFrame) -> pd.DataFrame:
    exit_df = normalize_emp_id(normalize_missing_values(exit_df))
    exit_lookup = (
        exit_df[["EmpID", "FireReason", "Manager"]]
        .rename(columns={"Manager": "ExitManager"})
        .drop_duplicates(subset=["EmpID"], keep="first")
    )
    return exit_lookup


def clean_employee_data(employee_df: pd.DataFrame, exit_df: pd.DataFrame) -> pd.DataFrame:
    employees = normalize_emp_id(normalize_missing_values(employee_df))
    exits = build_exit_lookup(exit_df)

    cleaned = employees.merge(exits, on="EmpID", how="left")

    if "Supervisor" in cleaned.columns:
        cleaned["Supervisor"] = cleaned["Supervisor"].fillna(cleaned["ExitManager"])

    exited_mask = cleaned["FireReason"].notna()
    cleaned.loc[exited_mask, "EmployeeStatus"] = "Exited"

    cleaned["DOB"] = format_date(cleaned["DOB"], "%d.%m.%Y")
    cleaned["StartDate"] = format_date(cleaned["StartDate"], "%d-%b-%y")
    cleaned["ExitDate"] = format_date(cleaned["ExitDate"], "%d-%b-%y")

    cleaned = cleaned.drop_duplicates(subset=["FirstName", "LastName", "DOB"], keep="first")
    return cleaned


def summarize(cleaned: pd.DataFrame) -> str:
    status_counts = cleaned["EmployeeStatus"].value_counts(dropna=False)
    fire_reasons = cleaned["FireReason"].value_counts(dropna=True).head(5)
    missing_counts = cleaned.isna().sum().sort_values(ascending=False).head(8)

    lines = [
        "Staff turnover cleaning summary",
        "-" * 36,
        f"Rows: {len(cleaned):,}",
        f"Columns: {len(cleaned.columns):,}",
        "",
        "Employee status:",
        status_counts.to_string(),
        "",
        "Top exit reasons:",
        fire_reasons.to_string() if not fire_reasons.empty else "No exit reasons found",
        "",
        "Most missing values:",
        missing_counts.to_string(),
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and merge HR staff turnover CSV files.")
    parser.add_argument("--employees", type=Path, default=EMPLOYEE_SRC)
    parser.add_argument("--exits", type=Path, default=EXIT_SRC)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--summary", action="store_true", help="Print a short data quality summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    employee_df = pd.read_csv(args.employees, sep=";", encoding="utf-8-sig")
    exit_df = pd.read_csv(args.exits, encoding="utf-8-sig")

    cleaned = clean_employee_data(employee_df, exit_df)
    cleaned.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"[OK] {args.output} | rows={len(cleaned):,} | columns={len(cleaned.columns):,}")
    if args.summary:
        print()
        print(summarize(cleaned))


if __name__ == "__main__":
    main()
