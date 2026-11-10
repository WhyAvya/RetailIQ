import pandas as pd
import pytest

from backend.query_planner import plan_query
from backend.structured_analytics import StructuredExecutionError, execute_plan


@pytest.fixture(scope="module")
def superstore():
    frame = pd.read_csv("data/sample_superstore.csv", encoding="cp1252")
    frame.index = pd.RangeIndex(1, len(frame) + 1)
    return frame


def test_executes_exact_filter_count(superstore):
    plan = plan_query("How many Technology line items are in the East region?")
    result = execute_plan(superstore, plan)
    expected = len(
        superstore.loc[
            (superstore["Category"] == "Technology")
            & (superstore["Region"] == "East")
        ]
    )

    assert result.value == expected
    assert result.rows_after_filtering == expected


def test_executes_row_level_threshold_date_and_top_n(superstore):
    plan = plan_query(
        "Which five Chairs line items in 2017 had the highest profit with sales at least 1000?"
    )
    result = execute_plan(superstore, plan)
    dates = pd.to_datetime(superstore["Order Date"], format="mixed")
    expected = (
        superstore.loc[
            (superstore["Sub-Category"] == "Chairs")
            & (dates.dt.year == 2017)
            & (superstore["Sales"] >= 1000)
        ]
        .sort_values("Profit", ascending=False, kind="stable")
        .head(5)
    )

    assert [record["Row ID"] for record in result.records] == expected["Row ID"].tolist()


def test_executes_order_and_product_level_grouping(superstore):
    average_order = execute_plan(superstore, plan_query("What is the average order value?"))
    top_products = execute_plan(superstore, plan_query("What are the top 3 products by sales?"))

    expected_average = superstore.groupby("Order ID")["Sales"].sum().mean()
    expected_products = (
        superstore.groupby(["Product ID", "Product Name"])["Sales"]
        .sum()
        .sort_values(ascending=False, kind="stable")
        .head(3)
    )
    assert average_order.value == pytest.approx(expected_average)
    assert [record["Sales"] for record in top_products.records] == pytest.approx(
        expected_products.tolist()
    )


def test_executes_grouped_aggregate_and_mixed_product_name_filter(superstore):
    grouped = execute_plan(superstore, plan_query("What are total sales by region?"))
    mixed = execute_plan(
        superstore,
        plan_query(
            "Among line items containing 'wireless' in the West, which five had highest sales?"
        ),
    )

    expected_grouped = superstore.groupby("Region")["Sales"].sum().to_dict()
    assert {record["Region"]: record["value"] for record in grouped.records} == pytest.approx(
        expected_grouped
    )
    assert len(mixed.records) == 5
    assert all("wireless" in record["Product Name"].casefold() for record in mixed.records)
    assert all(record["Region"] == "West" for record in mixed.records)


def test_grouped_averages_and_counts_respect_order_and_product_grains(superstore):
    order_average = execute_plan(
        superstore, plan_query("What is average order sales by region?")
    )
    product_count = execute_plan(
        superstore, plan_query("How many products are there by category?")
    )

    expected_order_average = (
        superstore.groupby(["Region", "Order ID"])["Sales"]
        .sum()
        .groupby("Region")
        .mean()
        .to_dict()
    )
    expected_product_count = (
        superstore.groupby("Category")["Product ID"].nunique().to_dict()
    )
    assert {
        record["Region"]: record["value"] for record in order_average.records
    } == pytest.approx(expected_order_average)
    assert {
        record["Category"]: record["value"] for record in product_count.records
    } == expected_product_count


def test_mixed_group_and_row_ranking_returns_both_results(superstore):
    result = execute_plan(
        superstore,
        plan_query(
            "For products containing 'recycled', which region generated the most sales "
            "and which matching line item had the highest sales?"
        ),
    )

    assert [record["result_kind"] for record in result.records] == [
        "group_ranking",
        "row_ranking",
    ]
    assert "Region" in result.records[0]
    assert "source_row" in result.records[1]


def test_rejects_ambiguous_plan_without_executing(superstore):
    with pytest.raises(StructuredExecutionError, match="Only valid query plans"):
        execute_plan(superstore, plan_query("What are the top products?"))


def test_executes_discount_and_shipping_duration_filters(superstore):
    discounted = execute_plan(
        superstore, plan_query("Show rows where discount is at least 30%")
    )
    slow_orders = execute_plan(
        superstore, plan_query("Which orders took more than seven calendar days to ship?")
    )
    shipping_days = (
        pd.to_datetime(superstore["Ship Date"], format="mixed")
        - pd.to_datetime(superstore["Order Date"], format="mixed")
    ).dt.days

    assert discounted.rows_after_filtering == int((superstore["Discount"] >= 0.3).sum())
    assert slow_orders.entity_count == superstore.loc[shipping_days > 7, "Order ID"].nunique()


def test_executes_grouped_totals_state_ranking_and_customer_order_count(superstore):
    yearly = execute_plan(
        superstore, plan_query("Show total profit for each order year")
    )
    states = execute_plan(
        superstore, plan_query("What are the top 3 states by total sales?")
    )
    customer = execute_plan(
        superstore,
        plan_query(
            "Among Home Office customers, who had the highest total sales and how many "
            "distinct orders did that customer place?"
        ),
    )

    expected_yearly = (
        superstore.assign(order_year=pd.to_datetime(superstore["Order Date"], format="mixed").dt.year)
        .groupby("order_year")["Profit"].sum().to_dict()
    )
    expected_states = superstore.groupby("State")["Sales"].sum().nlargest(3)
    home = superstore.loc[superstore["Segment"] == "Home Office"]
    expected_customer = home.groupby("Customer ID")["Sales"].sum().idxmax()
    expected_orders = home.loc[home["Customer ID"] == expected_customer, "Order ID"].nunique()

    assert {row["order_year"]: row["value"] for row in yearly.records} == pytest.approx(expected_yearly)
    assert [row["State"] for row in states.records] == expected_states.index.tolist()
    assert customer.records[0]["Customer ID"] == expected_customer
    assert customer.records[0]["distinct_order_count"] == expected_orders
    assert "Customer Name" in customer.records[0]


def test_customer_and_order_grouped_totals_do_not_double_count(superstore):
    by_customer = execute_plan(superstore, plan_query("What are total sales by customer?"))
    by_order = execute_plan(superstore, plan_query("What are total sales by order?"))

    expected_customers = superstore.groupby("Customer ID")["Sales"].sum().to_dict()
    expected_orders = superstore.groupby("Order ID")["Sales"].sum().to_dict()
    assert {row["Customer ID"]: row["value"] for row in by_customer.records} == pytest.approx(expected_customers)
    assert {row["Order ID"]: row["value"] for row in by_order.records} == pytest.approx(expected_orders)


def test_plural_category_rankings_match_full_dataset_aggregates(superstore):
    categories = execute_plan(
        superstore,
        plan_query("What are the top two product categories by total quantity?"),
    )
    sub_categories = execute_plan(
        superstore,
        plan_query("What are the top four sub-categories by total sales?"),
    )

    expected_categories = superstore.groupby("Category")["Quantity"].sum().nlargest(2)
    expected_sub_categories = superstore.groupby("Sub-Category")["Sales"].sum().nlargest(4)
    assert [record["Category"] for record in categories.records] == expected_categories.index.tolist()
    assert [record["Quantity"] for record in categories.records] == expected_categories.tolist()
    assert [record["Sub-Category"] for record in sub_categories.records] == expected_sub_categories.index.tolist()
    assert [record["Sales"] for record in sub_categories.records] == pytest.approx(expected_sub_categories.tolist())
