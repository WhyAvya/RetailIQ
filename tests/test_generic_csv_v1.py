from uuid import uuid4

import pandas as pd

from backend.dataset_selection import SelectedDataset
from backend.generic_analytics import generic_dashboard, infer_schema
from backend.generic_query import execute_generic_query, plan_generic_query
from backend.rag_pipeline import answer_question


def test_employee_schema_and_dashboard_are_not_retail_specific():
    frame = pd.DataFrame({"Employee": ["Ana", "Ben"], "Department": ["Ops", "Sales"], "Salary": [60000, 80000]})
    schema = infer_schema(frame)
    result = generic_dashboard(frame)
    assert {item["name"] for item in schema} == {"Employee", "Department", "Salary"}
    assert result["kind"] == "generic"
    assert result["summary"]["numeric"][0]["mean"] == 70000
    assert result["charts"]["categorical_distributions"]


def test_generic_planner_filters_and_counts_inventory():
    frame = pd.DataFrame({"Product": ["A", "B", "C"], "Stock": [4, 12, 8]})
    plan, issue = plan_generic_query(frame, "How many products have Stock below 10?")
    assert issue is None
    answer, calculation = execute_generic_query(frame, plan)
    assert calculation["value"] == 2
    assert "2" in answer


def test_generic_grouped_average_is_deterministic():
    frame = pd.DataFrame({"Department": ["Ops", "Ops", "Sales"], "Salary": [10, 20, 40]})
    plan, issue = plan_generic_query(frame, "What is the average Salary by Department?")
    assert issue is None
    _, calculation = execute_generic_query(frame, plan)
    assert calculation["records"] == [
        {"Department": "Ops", "Salary": 15.0},
        {"Department": "Sales", "Salary": 40.0},
    ]


def test_generic_ranking_and_categorical_filter_use_schema_values():
    frame = pd.DataFrame({"Category": ["A", "B", "A"], "Revenue": [10, 30, 15]})
    ranking, issue = plan_generic_query(frame, "Show the top five Category by Revenue")
    assert issue is None
    _, ranked = execute_generic_query(frame, ranking)
    assert ranked["records"][0] == {"Category": "B", "Revenue": 30.0}
    filtered, issue = plan_generic_query(frame, "How many rows are in Category B?")
    assert issue is None
    _, counted = execute_generic_query(frame, filtered)
    assert counted["value"] == 1


def test_generic_pipeline_uses_actual_uploaded_schema_without_llm():
    upload_id = uuid4()
    frame = pd.DataFrame({"Student": ["A", "B"], "Marks": [70, 90]})
    result = answer_question(
        upload_id,
        "Which Student has the highest Marks?",
        metadata_lookup=lambda _id: {"upload_id": str(upload_id)},
        dataset_loader=lambda _id: SelectedDataset(upload_id, "students.csv", frame),
    )
    assert result.mode == "analytics"
    assert result.calculation["records"][0]["Marks"] == 90
