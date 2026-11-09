"""Build and validate Benchmark B1 directly from the full Superstore CSV."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from backend.embeddings import generate_embeddings
from backend.rag_preparation import DocumentChunk, dataframe_to_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_PATH = PROJECT_ROOT / "data" / "sample_superstore.csv"
DEFAULT_BENCHMARK_PATH = PROJECT_ROOT / "evaluation" / "retailiq_benchmark_b1.json"
DEFAULT_COVERAGE_REPORT_PATH = (
    PROJECT_ROOT / "evaluation" / "reports" / "benchmark_b1_dataset_coverage.json"
)
DEFAULT_CHROMA_PATH = PROJECT_ROOT / "backend" / "storage" / "chroma"
FULL_DATASET_UPLOAD_ID = "11111111-1111-1111-1111-111111111111"
SOURCE_FILENAME = "sample_superstore.csv"
EXPECTED_ROW_COUNT = 9_994
EXPECTED_TYPE_COUNTS = {
    "semantic_product_discovery": 20,
    "metadata_filtering": 15,
    "numerical": 15,
    "mixed": 10,
}
EXPECTED_SPLIT_COUNTS = {"development": 40, "held_out_test": 20}


class BenchmarkB1ValidationError(ValueError):
    """Raised when a generated or loaded B1 artifact is invalid."""


def load_source_dataframe(source_path: Path | str = DEFAULT_SOURCE_PATH) -> pd.DataFrame:
    """Load the unchanged source using the encoding accepted by RetailIQ uploads."""
    frame = pd.read_csv(Path(source_path), encoding="cp1252")
    if len(frame) != EXPECTED_ROW_COUNT:
        raise BenchmarkB1ValidationError(
            f"expected {EXPECTED_ROW_COUNT} source rows, found {len(frame)}"
        )
    frame = frame.copy()
    frame.index = pd.RangeIndex(1, len(frame) + 1, name="source_row")
    frame["_order_date"] = pd.to_datetime(frame["Order Date"], format="mixed")
    frame["_ship_date"] = pd.to_datetime(frame["Ship Date"], format="mixed")
    return frame


def _json_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return round(value, 6)
    return value


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "source_row": int(row_number),
            **{column: _json_value(row[column]) for column in columns},
        }
        for row_number, row in frame.iterrows()
    ]


def _question(
    question_id: str,
    question_type: str,
    split: str,
    text: str,
    expected_answer: str,
    ground_truth: dict[str, Any],
    relevant_rows: list[int],
    relevance_definition: str,
) -> dict[str, Any]:
    return {
        "id": question_id,
        "type": question_type,
        "split": split,
        "question": text,
        "expected_answer": expected_answer,
        "ground_truth": ground_truth,
        "relevant_source_rows": relevant_rows,
        "relevance_definition": relevance_definition,
    }


def _semantic_questions(frame: pd.DataFrame) -> list[dict[str, Any]]:
    specs = [
        ("Which five Technology Accessories line items made the most profit?", (frame["Category"] == "Technology") & (frame["Sub-Category"] == "Accessories"), "Profit", False, ["Product Name", "Sales", "Profit"], "Category == Technology and Sub-Category == Accessories"),
        ("Which five Furniture line items had the largest losses?", (frame["Category"] == "Furniture") & (frame["Profit"] < 0), "Profit", True, ["Product Name", "Sales", "Profit"], "Category == Furniture and Profit < 0"),
        ("Which five products produced the highest sales line items in the West region?", frame["Region"] == "West", "Sales", False, ["Product Name", "Category", "Sales", "Profit"], "Region == West"),
        ("Which five chair line items had the highest sales?", frame["Sub-Category"] == "Chairs", "Sales", False, ["Product Name", "Sales", "Profit"], "Sub-Category == Chairs"),
        ("Which five copier line items earned the most profit?", frame["Sub-Category"] == "Copiers", "Profit", False, ["Product Name", "Sales", "Profit"], "Sub-Category == Copiers"),
        ("Which five product line items containing 'wireless' in the product name had the highest sales?", frame["Product Name"].str.contains("wireless", case=False, na=False), "Sales", False, ["Product Name", "Sales", "Profit"], "Product Name contains wireless, case-insensitive"),
        ("Which five product line items containing 'recycled' in the product name made the most profit?", frame["Product Name"].str.contains("recycled", case=False, na=False), "Profit", False, ["Product Name", "Sales", "Profit"], "Product Name contains recycled, case-insensitive"),
        ("Which five ergonomic product line items had the highest sales?", frame["Product Name"].str.contains("ergonomic", case=False, na=False), "Sales", False, ["Product Name", "Sub-Category", "Sales", "Profit"], "Product Name contains ergonomic, case-insensitive"),
        ("Which five profitable phone line items in the South earned the most profit?", (frame["Region"] == "South") & (frame["Sub-Category"] == "Phones") & (frame["Profit"] > 0), "Profit", False, ["Product Name", "Sales", "Profit"], "Region == South, Sub-Category == Phones, and Profit > 0"),
        ("Which five machine line items generated the largest losses?", (frame["Sub-Category"] == "Machines") & (frame["Profit"] < 0), "Profit", True, ["Product Name", "Sales", "Profit"], "Sub-Category == Machines and Profit < 0"),
        ("Among storage line items with quantity of at least 8, which five had the highest sales?", (frame["Sub-Category"] == "Storage") & (frame["Quantity"] >= 8), "Sales", False, ["Product Name", "Quantity", "Sales", "Profit"], "Sub-Category == Storage and Quantity >= 8"),
        ("Which five paper line items sold in December 2017 had the highest sales?", (frame["Sub-Category"] == "Paper") & frame["_order_date"].between("2017-12-01", "2017-12-31"), "Sales", False, ["Product Name", "Order Date", "Sales", "Profit"], "Sub-Category == Paper and Order Date is in December 2017"),
        ("Which five profitable bookcase line items in the East earned the most profit?", (frame["Region"] == "East") & (frame["Sub-Category"] == "Bookcases") & (frame["Profit"] > 0), "Profit", False, ["Product Name", "Sales", "Profit"], "Region == East, Sub-Category == Bookcases, and Profit > 0"),
        ("Which five appliance line items in the Central region earned the most profit?", (frame["Region"] == "Central") & (frame["Sub-Category"] == "Appliances"), "Profit", False, ["Product Name", "Sales", "Profit"], "Region == Central and Sub-Category == Appliances"),
        ("Which five envelope line items bought by Consumer customers had the highest sales?", (frame["Segment"] == "Consumer") & (frame["Sub-Category"] == "Envelopes"), "Sales", False, ["Product Name", "Quantity", "Sales", "Profit"], "Segment == Consumer and Sub-Category == Envelopes"),
        ("Which five fastener line items bought by Corporate customers earned the most profit?", (frame["Segment"] == "Corporate") & (frame["Sub-Category"] == "Fasteners"), "Profit", False, ["Product Name", "Sales", "Profit"], "Segment == Corporate and Sub-Category == Fasteners"),
        ("Which five art line items bought by Home Office customers had the highest sales?", (frame["Segment"] == "Home Office") & (frame["Sub-Category"] == "Art"), "Sales", False, ["Product Name", "Sales", "Profit"], "Segment == Home Office and Sub-Category == Art"),
        ("Which five table line items discounted by at least 30% had the highest sales?", (frame["Sub-Category"] == "Tables") & (frame["Discount"] >= 0.3), "Sales", False, ["Product Name", "Discount", "Sales", "Profit"], "Sub-Category == Tables and Discount >= 0.30"),
        ("Which five furnishing line items shipped Same Day earned the most profit?", (frame["Ship Mode"] == "Same Day") & (frame["Sub-Category"] == "Furnishings"), "Profit", False, ["Product Name", "Sales", "Profit"], "Ship Mode == Same Day and Sub-Category == Furnishings"),
        ("Which five label line items in the West had the highest sales?", (frame["Region"] == "West") & (frame["Sub-Category"] == "Labels"), "Sales", False, ["Product Name", "Sales", "Profit"], "Region == West and Sub-Category == Labels"),
    ]
    questions = []
    for number, (text, mask, sort_column, ascending, columns, predicate) in enumerate(specs, 1):
        matches = frame.loc[mask]
        selected = matches.sort_values(
            [sort_column, "Row ID"], ascending=[ascending, True], kind="stable"
        ).head(5)
        records = _records(selected, columns)
        answer = "; ".join(
            f"row {record['source_row']}: {record['Product Name']} "
            f"({sort_column}={_money(float(record[sort_column])) if sort_column in {'Sales', 'Profit'} else record[sort_column]})"
            for record in records
        )
        questions.append(
            _question(
                f"B1-SEM-{number:03d}",
                "semantic_product_discovery",
                "development" if number <= 13 else "held_out_test",
                text,
                answer,
                {
                    "verification": f"Pandas filter: {predicate}; stable sort by {sort_column} {'ascending' if ascending else 'descending'}, then Row ID ascending; take 5.",
                    "matching_row_count_before_top_k": int(len(matches)),
                    "records": records,
                },
                [int(value) for value in selected.index],
                f"Exactly the first five rows after applying ({predicate}) and sorting by {sort_column} {'ascending' if ascending else 'descending'}, with Row ID ascending as the tie-breaker.",
            )
        )
    return questions


def _metadata_questions(frame: pd.DataFrame) -> list[dict[str, Any]]:
    specs = [
        ("How many line items are from the West region?", frame["Region"] == "West", "Region == West"),
        ("How many Technology line items are from the East region?", (frame["Region"] == "East") & (frame["Category"] == "Technology"), "Region == East and Category == Technology"),
        ("How many Furniture line items are from the South region?", (frame["Region"] == "South") & (frame["Category"] == "Furniture"), "Region == South and Category == Furniture"),
        ("How many Office Supplies line items are from the Central region?", (frame["Region"] == "Central") & (frame["Category"] == "Office Supplies"), "Region == Central and Category == Office Supplies"),
        ("How many line items were ordered in California?", frame["State"] == "California", "State == California"),
        ("How many line items were ordered in New York City?", frame["City"] == "New York City", "City == New York City"),
        ("How many line items were ordered during 2017?", frame["_order_date"].dt.year == 2017, "Order Date year == 2017"),
        ("How many line items were ordered in the fourth quarter of 2016?", frame["_order_date"].between("2016-10-01", "2016-12-31"), "2016-10-01 <= Order Date <= 2016-12-31"),
        ("How many Corporate line items used Standard Class shipping?", (frame["Segment"] == "Corporate") & (frame["Ship Mode"] == "Standard Class"), "Segment == Corporate and Ship Mode == Standard Class"),
        ("How many Home Office line items used First Class shipping?", (frame["Segment"] == "Home Office") & (frame["Ship Mode"] == "First Class"), "Segment == Home Office and Ship Mode == First Class"),
        ("How many Furniture line items are in the Chairs sub-category?", (frame["Category"] == "Furniture") & (frame["Sub-Category"] == "Chairs"), "Category == Furniture and Sub-Category == Chairs"),
        ("How many line items had no discount?", frame["Discount"] == 0, "Discount == 0"),
        ("How many line items used Same Day shipping?", frame["Ship Mode"] == "Same Day", "Ship Mode == Same Day"),
        ("How many line items took more than seven calendar days to ship?", (frame["_ship_date"] - frame["_order_date"]).dt.days > 7, "Ship Date minus Order Date > 7 days"),
        ("How many Office Supplies line items were ordered in Texas?", (frame["State"] == "Texas") & (frame["Category"] == "Office Supplies"), "State == Texas and Category == Office Supplies"),
    ]
    questions = []
    for number, (text, mask, predicate) in enumerate(specs, 1):
        rows = [int(value) for value in frame.index[mask]]
        count = len(rows)
        questions.append(
            _question(
                f"B1-META-{number:03d}",
                "metadata_filtering",
                "development" if number <= 10 else "held_out_test",
                text,
                f"{count:,} line items match.",
                {
                    "verification": f"Pandas boolean filter: {predicate}; count matching rows.",
                    "matching_row_count": count,
                },
                rows,
                f"Every CSV data row satisfying ({predicate}) is relevant; no top-k truncation is applied.",
            )
        )
    return questions


def _numerical_questions(frame: pd.DataFrame) -> list[dict[str, Any]]:
    order_sales = frame.groupby("Order ID", sort=False)["Sales"].sum()
    sales_by_region = frame.groupby("Region")["Sales"].sum().sort_index()
    profit_by_category = frame.groupby("Category")["Profit"].sum().sort_index()
    sales_by_state = frame.groupby("State")["Sales"].sum()
    profit_by_subcategory = frame.groupby("Sub-Category")["Profit"].sum()
    sales_by_year = frame.groupby(frame["_order_date"].dt.year)["Sales"].sum().sort_index()
    entries = [
        ("What are total sales across the full dataset?", "SUM(Sales)", round(float(frame["Sales"].sum()), 6), f"Total sales are {_money(frame['Sales'].sum())}."),
        ("What is total profit across the full dataset?", "SUM(Profit)", round(float(frame["Profit"].sum()), 6), f"Total profit is {_money(frame['Profit'].sum())}."),
        ("What is the average sales value per line item?", "MEAN(Sales)", round(float(frame["Sales"].mean()), 6), f"Average sales per line item are {_money(frame['Sales'].mean())}."),
        ("What is the median profit per line item?", "MEDIAN(Profit)", round(float(frame["Profit"].median()), 6), f"Median profit per line item is {_money(frame['Profit'].median())}."),
        ("How many distinct orders are in the full dataset?", "NUNIQUE(Order ID)", int(frame["Order ID"].nunique()), f"There are {frame['Order ID'].nunique():,} distinct orders."),
        ("How many distinct customers are in the full dataset?", "NUNIQUE(Customer ID)", int(frame["Customer ID"].nunique()), f"There are {frame['Customer ID'].nunique():,} distinct customers."),
        ("What is the average order value, after combining all line items for each order?", "GROUP BY Order ID, SUM(Sales), then MEAN", round(float(order_sales.mean()), 6), f"Average order value is {_money(order_sales.mean())}."),
        ("What is the overall profit margin as a percentage of sales?", "SUM(Profit) / SUM(Sales) * 100", round(float(frame["Profit"].sum() / frame["Sales"].sum() * 100), 6), f"Overall profit margin is {frame['Profit'].sum() / frame['Sales'].sum() * 100:.2f}%."),
        ("How many units were sold in total?", "SUM(Quantity)", int(frame["Quantity"].sum()), f"A total of {frame['Quantity'].sum():,} units were sold."),
        ("How many line items were profitable?", "COUNT rows where Profit > 0", int((frame["Profit"] > 0).sum()), f"{(frame['Profit'] > 0).sum():,} line items had profit greater than zero."),
        ("What were total sales in each region?", "GROUP BY Region, SUM(Sales)", {key: round(float(value), 6) for key, value in sales_by_region.items()}, "; ".join(f"{key}: {_money(value)}" for key, value in sales_by_region.items()) + "."),
        ("What was total profit in each product category?", "GROUP BY Category, SUM(Profit)", {key: round(float(value), 6) for key, value in profit_by_category.items()}, "; ".join(f"{key}: {_money(value)}" for key, value in profit_by_category.items()) + "."),
        ("Which state had the highest total sales, and what was the amount?", "GROUP BY State, SUM(Sales), take maximum", {"state": str(sales_by_state.idxmax()), "sales": round(float(sales_by_state.max()), 6)}, f"{sales_by_state.idxmax()} had the highest total sales at {_money(sales_by_state.max())}."),
        ("Which sub-category had the lowest total profit, and what was the amount?", "GROUP BY Sub-Category, SUM(Profit), take minimum", {"sub_category": str(profit_by_subcategory.idxmin()), "profit": round(float(profit_by_subcategory.min()), 6)}, f"{profit_by_subcategory.idxmin()} had the lowest total profit at {_money(profit_by_subcategory.min())}."),
        ("What were total sales in each order year?", "GROUP BY year(Order Date), SUM(Sales)", {str(key): round(float(value), 6) for key, value in sales_by_year.items()}, "; ".join(f"{key}: {_money(value)}" for key, value in sales_by_year.items()) + "."),
    ]
    questions = []
    for number, (text, operation, value, answer) in enumerate(entries, 1):
        questions.append(
            _question(
                f"B1-NUM-{number:03d}",
                "numerical",
                "development" if number <= 10 else "held_out_test",
                text,
                answer,
                {"verification": f"Pandas calculation over the original CSV: {operation}.", "value": value},
                [],
                "This is a structured aggregate. The declared source scope is all 9,994 CSV data rows; individual rows are not retrieval judgments.",
            )
        )
    return questions


def _mixed_questions(frame: pd.DataFrame) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []

    def add(
        number: int,
        text: str,
        mask: pd.Series,
        predicate: str,
        calculation: Callable[[pd.DataFrame], tuple[str, dict[str, Any]]],
    ) -> None:
        matches = frame.loc[mask]
        answer, result = calculation(matches)
        questions.append(
            _question(
                f"B1-MIX-{number:03d}",
                "mixed",
                "development" if number <= 7 else "held_out_test",
                text,
                answer,
                {
                    "verification": f"Pandas filter: {predicate}; then perform the stated structured calculation.",
                    "matching_row_count": int(len(matches)),
                    **result,
                },
                [int(value) for value in matches.index],
                f"Every row satisfying ({predicate}) is relevant because the aggregate uses the complete filtered set.",
            )
        )

    def top_and_total(matches: pd.DataFrame, sort: str, total: str) -> tuple[str, dict[str, Any]]:
        top = matches.sort_values([sort, "Row ID"], ascending=[False, True], kind="stable").head(1)
        record = _records(top, ["Product Name", sort])[0]
        total_value = float(matches[total].sum())
        return (
            f"The top line item is row {record['source_row']}, {record['Product Name']} ({sort}={_money(record[sort])}); filtered {total.lower()} total is {_money(total_value)}.",
            {"top_record": record, f"total_{total.lower()}": round(total_value, 6)},
        )

    wireless = frame["Product Name"].str.contains("wireless", case=False, na=False)
    add(1, "Among products containing 'wireless', which line item made the most profit and what were total sales for all matching rows?", wireless, "Product Name contains wireless, case-insensitive", lambda x: top_and_total(x, "Profit", "Sales"))
    west_chairs = (frame["Region"] == "West") & (frame["Sub-Category"] == "Chairs")
    add(2, "For chair line items in the West, which had the highest sales and what was average profit across the matching rows?", west_chairs, "Region == West and Sub-Category == Chairs", lambda x: (f"The highest-sales line is row {int(x['Sales'].idxmax())}, {x.loc[x['Sales'].idxmax(), 'Product Name']} at {_money(x['Sales'].max())}; average profit is {_money(x['Profit'].mean())}.", {"top_source_row": int(x["Sales"].idxmax()), "top_sales": round(float(x["Sales"].max()), 6), "average_profit": round(float(x["Profit"].mean()), 6)}))
    tech_2017 = (frame["Category"] == "Technology") & (frame["_order_date"].dt.year == 2017)
    add(3, "Within 2017 Technology rows, which sub-category had the most sales and what was its sales total?", tech_2017, "Category == Technology and Order Date year == 2017", lambda x: (lambda totals: (f"{totals.idxmax()} had the most sales at {_money(totals.max())}.", {"sub_category": str(totals.idxmax()), "sales": round(float(totals.max()), 6)}))(x.groupby("Sub-Category")["Sales"].sum()))
    furniture_losses = (frame["Category"] == "Furniture") & (frame["Profit"] < 0)
    add(4, "Which three Furniture line items had the largest losses, and what was the total loss across all loss-making Furniture rows?", furniture_losses, "Category == Furniture and Profit < 0", lambda x: (lambda top: ("; ".join(f"row {r['source_row']}: {r['Product Name']} ({_money(r['Profit'])})" for r in _records(top, ["Product Name", "Profit"])) + f"; all matching rows total {_money(x['Profit'].sum())}.", {"top_records": _records(top, ["Product Name", "Profit"]), "total_profit": round(float(x["Profit"].sum()), 6)}))(x.sort_values(["Profit", "Row ID"], kind="stable").head(3)))
    ca_corp_office = (frame["State"] == "California") & (frame["Segment"] == "Corporate") & (frame["Category"] == "Office Supplies")
    add(5, "For Corporate Office Supplies rows in California, which line item had the highest sales and how many rows matched?", ca_corp_office, "State == California, Segment == Corporate, and Category == Office Supplies", lambda x: (f"Row {int(x['Sales'].idxmax())}, {x.loc[x['Sales'].idxmax(), 'Product Name']}, had the highest sales at {_money(x['Sales'].max())}; {len(x):,} rows matched.", {"top_source_row": int(x["Sales"].idxmax()), "top_sales": round(float(x["Sales"].max()), 6), "count": int(len(x))}))
    q4_same_day = frame["_order_date"].between("2016-10-01", "2016-12-31") & (frame["Ship Mode"] == "Same Day")
    add(6, "For Same Day rows ordered in Q4 2016, which line item had the highest sales and what were total sales?", q4_same_day, "2016-10-01 <= Order Date <= 2016-12-31 and Ship Mode == Same Day", lambda x: top_and_total(x, "Sales", "Sales"))
    recycled = frame["Product Name"].str.contains("recycled", case=False, na=False)
    add(7, "For products containing 'recycled', which region generated the most sales and which matching line item had the highest sales?", recycled, "Product Name contains recycled, case-insensitive", lambda x: (lambda totals, top: (f"{totals.idxmax()} generated the most sales at {_money(totals.max())}; row {int(top.name)}, {top['Product Name']}, was the highest-sales line at {_money(top['Sales'])}.", {"region": str(totals.idxmax()), "region_sales": round(float(totals.max()), 6), "top_source_row": int(top.name), "top_sales": round(float(top["Sales"]), 6)}))(x.groupby("Region")["Sales"].sum(), x.sort_values(["Sales", "Row ID"], ascending=[False, True], kind="stable").iloc[0]))
    central_discounted_appliances = (frame["Region"] == "Central") & (frame["Sub-Category"] == "Appliances") & (frame["Discount"] > 0)
    add(8, "For discounted appliance rows in the Central region, which three made the most profit and what was total profit?", central_discounted_appliances, "Region == Central, Sub-Category == Appliances, and Discount > 0", lambda x: (lambda top: ("; ".join(f"row {r['source_row']}: {r['Product Name']} ({_money(r['Profit'])})" for r in _records(top, ["Product Name", "Profit"])) + f"; total profit {_money(x['Profit'].sum())}.", {"top_records": _records(top, ["Product Name", "Profit"]), "total_profit": round(float(x["Profit"].sum()), 6)}))(x.sort_values(["Profit", "Row ID"], ascending=[False, True], kind="stable").head(3)))
    home_office = frame["Segment"] == "Home Office"
    add(9, "Among Home Office customers, who had the highest total sales and how many distinct orders did that customer place?", home_office, "Segment == Home Office", lambda x: (lambda totals: (lambda customer: (f"{customer} had the highest sales at {_money(totals.loc[customer])} across {x.loc[x['Customer ID'] == customer, 'Order ID'].nunique()} distinct orders.", {"customer_id": str(customer), "sales": round(float(totals.loc[customer]), 6), "distinct_orders": int(x.loc[x["Customer ID"] == customer, "Order ID"].nunique())}))(totals.idxmax()))(x.groupby("Customer ID")["Sales"].sum()))
    south_profitable_phones = (frame["Region"] == "South") & (frame["Sub-Category"] == "Phones") & (frame["Profit"] > 0)
    add(10, "For profitable phone rows in the South, which line item made the most profit and what was the aggregate profit margin?", south_profitable_phones, "Region == South, Sub-Category == Phones, and Profit > 0", lambda x: (lambda top, margin: (f"Row {int(top.name)}, {top['Product Name']}, made the most profit at {_money(top['Profit'])}; aggregate profit margin was {margin:.2f}%.", {"top_source_row": int(top.name), "top_profit": round(float(top["Profit"]), 6), "profit_margin_percent": round(float(margin), 6)}))(x.sort_values(["Profit", "Row ID"], ascending=[False, True], kind="stable").iloc[0], x["Profit"].sum() / x["Sales"].sum() * 100))
    return questions


def build_benchmark(source_path: Path | str = DEFAULT_SOURCE_PATH) -> dict[str, Any]:
    """Calculate all B1 answers and relevance judgments with Pandas."""
    source_path = Path(source_path)
    frame = load_source_dataframe(source_path)
    questions = (
        _semantic_questions(frame)
        + _metadata_questions(frame)
        + _numerical_questions(frame)
        + _mixed_questions(frame)
    )
    return {
        "benchmark": "RetailIQ Benchmark B1",
        "version": 1,
        "source": {
            "path": "data/sample_superstore.csv",
            "filename": SOURCE_FILENAME,
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "encoding": "cp1252",
            "row_count": len(frame),
            "column_count": len([column for column in frame.columns if not column.startswith("_")]),
            "row_numbering": "1-based CSV data rows; header excluded",
            "upload_id": FULL_DATASET_UPLOAD_ID,
        },
        "ground_truth_method": "All answers, structured ground truth, and relevance sets were calculated directly from the original CSV with Pandas. No LLM, vector search, or generated answer was used.",
        "split_policy": "Development and held-out test IDs are fixed before any model run. Held-out questions must not be used for tuning.",
        "questions": questions,
    }


def validate_benchmark(
    benchmark: dict[str, Any] | None = None,
    source_path: Path | str = DEFAULT_SOURCE_PATH,
) -> dict[str, Any]:
    """Validate structure and independently reproduce every ground-truth field."""
    expected = build_benchmark(source_path)
    if benchmark is None:
        benchmark = json.loads(DEFAULT_BENCHMARK_PATH.read_text(encoding="utf-8"))
    errors: list[str] = []
    questions = benchmark.get("questions", [])
    ids = [entry.get("id") for entry in questions]
    if len(questions) != 60:
        errors.append(f"expected 60 questions, found {len(questions)}")
    if len(ids) != len(set(ids)):
        errors.append("question IDs are not unique")
    type_counts = Counter(entry.get("type") for entry in questions)
    split_counts = Counter(entry.get("split") for entry in questions)
    if dict(type_counts) != EXPECTED_TYPE_COUNTS:
        errors.append(f"incorrect type counts: {dict(type_counts)}")
    if dict(split_counts) != EXPECTED_SPLIT_COUNTS:
        errors.append(f"incorrect split counts: {dict(split_counts)}")
    required = {
        "id", "type", "split", "question", "expected_answer", "ground_truth",
        "relevant_source_rows", "relevance_definition",
    }
    for entry in questions:
        if required - entry.keys():
            errors.append(f"{entry.get('id')} is missing required fields")
        rows = entry.get("relevant_source_rows", [])
        if len(rows) != len(set(rows)) or any(
            not isinstance(row, int) or isinstance(row, bool) or not 1 <= row <= EXPECTED_ROW_COUNT
            for row in rows
        ):
            errors.append(f"{entry.get('id')} has invalid relevant source rows")
    if benchmark != expected:
        errors.append("artifact does not match independently recomputed Pandas ground truth")
    if errors:
        raise BenchmarkB1ValidationError("; ".join(errors))
    return benchmark


def expected_chunks(
    source_path: Path | str = DEFAULT_SOURCE_PATH,
    upload_id: str = FULL_DATASET_UPLOAD_ID,
) -> list[DocumentChunk]:
    frame = load_source_dataframe(source_path).drop(columns=["_order_date", "_ship_date"])
    return dataframe_to_chunks(frame, upload_id, SOURCE_FILENAME)


def audit_index_coverage(
    stored: dict[str, Any], chunks: list[DocumentChunk]
) -> dict[str, Any]:
    """Compare deterministic expected chunks with one Chroma upload snapshot."""
    expected_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    stored_ids = list(stored.get("ids") or [])
    metadatas = list(stored.get("metadatas") or [])
    rows_to_ids: dict[int, list[str]] = defaultdict(list)
    for chunk_id, metadata in zip(stored_ids, metadatas):
        row_number = metadata.get("row_number")
        if isinstance(row_number, int):
            rows_to_ids[row_number].append(chunk_id)
    expected_rows = {int(chunk.metadata["row_number"]) for chunk in chunks}
    stored_rows = set(rows_to_ids)
    missing_rows = sorted(expected_rows - stored_rows)
    duplicate_rows = sorted(row for row, ids in rows_to_ids.items() if len(ids) > 1)
    unexpected_ids = sorted(set(stored_ids) - set(expected_by_id))
    mismatched_ids = sorted(
        chunk.chunk_id
        for chunk in chunks
        if chunk.metadata["row_number"] in stored_rows and chunk.chunk_id not in stored_ids
    )
    return {
        "expected_rows": len(expected_rows),
        "expected_chunks": len(chunks),
        "stored_chunks": len(stored_ids),
        "unique_stored_ids": len(set(stored_ids)),
        "unique_stored_rows": len(stored_rows),
        "missing_rows": missing_rows,
        "duplicate_rows": duplicate_rows,
        "unexpected_chunk_ids": unexpected_ids,
        "row_id_conflicts": mismatched_ids,
        "complete": not missing_rows and not duplicate_rows and not unexpected_ids and not mismatched_ids and len(stored_ids) == len(chunks),
    }


def prepare_full_dataset_index(
    source_path: Path | str = DEFAULT_SOURCE_PATH,
    chroma_path: Path | str = DEFAULT_CHROMA_PATH,
    upload_id: str = FULL_DATASET_UPLOAD_ID,
) -> dict[str, Any]:
    """Audit B1 coverage and embed/upsert missing rows only."""
    import chromadb

    chunks = expected_chunks(source_path, upload_id)
    client = chromadb.PersistentClient(path=str(Path(chroma_path).resolve()))
    collection = client.get_or_create_collection(
        name="retailiq_chunks", metadata={"hnsw:space": "cosine"}
    )
    stored = collection.get(where={"upload_id": upload_id}, include=["metadatas"])
    before = audit_index_coverage(stored, chunks)
    if before["duplicate_rows"] or before["row_id_conflicts"]:
        raise BenchmarkB1ValidationError(
            "existing Chroma rows conflict with deterministic B1 IDs; refusing to create duplicates"
        )
    missing_set = set(before["missing_rows"])
    missing_chunks = [
        chunk for chunk in chunks if int(chunk.metadata["row_number"]) in missing_set
    ]
    for start in range(0, len(missing_chunks), 500):
        embedded = generate_embeddings(missing_chunks[start : start + 500])
        collection.upsert(
            ids=[chunk.chunk_id for chunk in embedded],
            documents=[chunk.text for chunk in embedded],
            metadatas=[chunk.metadata for chunk in embedded],
            embeddings=[chunk.embedding for chunk in embedded],
        )
    after_stored = collection.get(where={"upload_id": upload_id}, include=["metadatas"])
    after = audit_index_coverage(after_stored, chunks)
    return {
        "benchmark": "RetailIQ Benchmark B1",
        "source_path": "data/sample_superstore.csv",
        "source_sha256": hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
        "upload_id": upload_id,
        "indexing_action": "none" if not missing_chunks else "upserted_missing_rows_only",
        "rows_indexed_during_run": len(missing_chunks),
        "coverage_before": before,
        "coverage_after": after,
    }
