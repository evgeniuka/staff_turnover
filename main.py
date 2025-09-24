import pandas as pd
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
    ex_raw = pd.read_csv(EXIT_SRC, sep=";", encoding="utf-8-sig")
    ch_raw = pd.read_csv(CH_SRC,   sep=";", encoding="utf-8-sig")

    ex_raw["EmpID"] = pd.to_numeric(ex_raw["EmpID"], errors="coerce").astype("Int64")
    ch_raw["EmpID"] = pd.to_numeric(ch_raw["EmpID"], errors="coerce").astype("Int64")

    merged = ch_raw.merge(ex_raw[["EmpID", "Manager", "FireReason"]], on="EmpID", how="left")

    mask = merged["Manager"].notna()
    merged.loc[mask, "Supervisor"] = merged.loc[mask, "Manager"]
    merged = merged.drop(columns=["Manager"])

    merged = clean_dataframe_keep_names(merged)

    counts = (merged["FireReason"]
              .dropna()
              .value_counts()
              .rename_axis("FireReason")
              .reset_index(name="Count"))
    counts["Share"] = (counts["Count"] / counts["Count"].sum()).round(4)

    merged["FireReasonCount"] = (
        merged["FireReason"].map(counts.set_index("FireReason")["Count"]).fillna(0).astype("Int64")
    )

    leaver_mask = merged["FireReason"].notna()
    merged.loc[leaver_mask, "EmployeeStatus"] = "Exited"

    merged.to_csv("employee_clean.csv", index=False, encoding="utf-8-sig")
    counts.to_csv("fire_reason_counts.csv", index=False, encoding="utf-8-sig")

    print(f"[OK] employee_master.csv | rows={len(merged)} | matched={int(mask.sum())}")
    print(f"[OK] fire_reason_counts.csv | reasons={len(counts)}")

if __name__ == "__main__":
    main()
