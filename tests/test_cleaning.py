import pandas as pd

from main import clean_employee_data


def test_clean_employee_data_preserves_supervisor_and_merges_exit_reason():
    employees = pd.DataFrame(
        [
            {
                "EmpID": "1",
                "FirstName": "Alex",
                "LastName": "Stone",
                "Supervisor": "Original Manager",
                "EmployeeStatus": "Active",
                "DOB": "01.02.1990",
                "StartDate": "05-Jan-20",
                "ExitDate": "",
            }
        ]
    )
    exits = pd.DataFrame(
        [
            {
                "EmpID": "1",
                "FireReason": "Better opportunity",
                "Manager": "Exit Manager",
            }
        ]
    )

    cleaned = clean_employee_data(employees, exits)

    row = cleaned.iloc[0]
    assert row["Supervisor"] == "Original Manager"
    assert row["ExitManager"] == "Exit Manager"
    assert row["FireReason"] == "Better opportunity"
    assert row["EmployeeStatus"] == "Exited"
    assert row["DOB"] == "1990-02-01"
    assert row["StartDate"] == "2020-01-05"


def test_clean_employee_data_deduplicates_by_name_and_dob():
    employees = pd.DataFrame(
        [
            {
                "EmpID": "1",
                "FirstName": "Alex",
                "LastName": "Stone",
                "Supervisor": "A",
                "EmployeeStatus": "Active",
                "DOB": "01.02.1990",
                "StartDate": "05-Jan-20",
                "ExitDate": "",
            },
            {
                "EmpID": "2",
                "FirstName": "Alex",
                "LastName": "Stone",
                "Supervisor": "B",
                "EmployeeStatus": "Active",
                "DOB": "01.02.1990",
                "StartDate": "05-Jan-20",
                "ExitDate": "",
            },
        ]
    )
    exits = pd.DataFrame(columns=["EmpID", "FireReason", "Manager"])

    cleaned = clean_employee_data(employees, exits)

    assert len(cleaned) == 1
