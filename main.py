import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(".")
EXIT_SRC = BASE / "employee_data_exit_view.csv"
CH_SRC   = BASE / "employee_data_Ch.csv"

def clean_dataframe_keep_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    na_like = {"", "na", "n/a", "none", "null", "-", "--", "nan"}
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip().replace({v: pd.NA for v in na_like}, regex=False)
    return df

def main():
    ex_raw = pd.read_csv(EXIT_SRC, encoding="utf-8-sig")
    ch_raw = pd.read_csv(CH_SRC,   sep=";", encoding="utf-8-sig")

    ex_raw["EmpID"] = pd.to_numeric(ex_raw["EmpID"], errors="coerce").astype("Int64")
    ch_raw["EmpID"] = pd.to_numeric(ch_raw["EmpID"], errors="coerce").astype("Int64")

    ex_fire = ex_raw[["EmpID", "FireReason"]].drop_duplicates(subset=["EmpID"], keep="first")
    merged = ch_raw.merge(ex_fire, on="EmpID", how="left")

    rng = np.random.default_rng(42)
    managers = ex_raw["Manager"].dropna().astype(str).str.strip().unique()
    weights  = rng.gamma(shape=0.6, scale=1.0, size=len(managers))
    probs    = weights / weights.sum()
    merged["Supervisor"] = rng.choice(managers, size=len(merged), replace=True, p=probs)

    merged = clean_dataframe_keep_names(merged)

    leaver_mask = merged["FireReason"].notna()
    merged.loc[leaver_mask, "EmployeeStatus"] = "Exited"


    merged["DOB"] = pd.to_datetime(merged["DOB"], format="%d.%m.%Y", errors="coerce").dt.strftime("%Y-%m-%d")
    merged = merged.drop_duplicates(subset=["FirstName", "LastName", "DOB"], keep="first")

    merged.to_csv("employee_clean.csv", index=False, encoding="utf-8-sig")

    print(f"[OK] employee_clean.csv | rows={len(merged)} | managers={len(managers)}")

if __name__ == "__main__":
    main()
