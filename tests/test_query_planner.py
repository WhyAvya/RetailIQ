from backend.query_planner import plan_query


def test_plans_pure_semantic_query_without_structured_constraints():
    plan = plan_query("Find items suitable for organizing a small office")

    assert plan.status == "valid"
    assert plan.query_type == "semantic"
    assert plan.operation == "semantic"
    assert plan.semantic_query == plan.original_question
    assert plan.filters.is_empty


def test_plans_exact_dimensions_dates_threshold_and_top_n():
    plan = plan_query(
        "Which five Technology Accessories line items in the West during 2017 "
        "had the most profit with sales at least 1000?"
    )

    assert plan.status == "valid"
    assert plan.query_type == "ranking"
    assert plan.grain == "row"
    assert plan.operation == "ranking"
    assert plan.metric == "profit"
    assert plan.limit == 5
    assert plan.filters.region == "West"
    assert plan.filters.category == "Technology"
    assert plan.filters.sub_category == "Accessories"
    assert plan.filters.start_date.isoformat() == "2017-01-01"
    assert plan.filters.end_date.isoformat() == "2017-12-31"
    assert plan.filters.numeric[0].operator == "gte"
    assert plan.filters.numeric[0].value == 1000


def test_plans_row_order_and_product_grains_with_appropriate_grouping():
    row_plan = plan_query("How many line items are in the East region?")
    order_plan = plan_query("What is the average order value?")
    product_plan = plan_query("What are the top 3 products by profit?")

    assert (row_plan.grain, row_plan.operation) == ("row", "count")
    assert (order_plan.grain, order_plan.operation, order_plan.metric) == (
        "order",
        "mean",
        "sales",
    )
    assert (product_plan.grain, product_plan.operation, product_plan.limit) == (
        "product",
        "ranking",
        3,
    )


def test_plans_mixed_keyword_and_structured_filter_query():
    plan = plan_query(
        "Among line items containing 'wireless' in the South, which five had highest sales?"
    )

    assert plan.status == "valid"
    assert plan.query_type == "mixed"
    assert plan.filters.product_name_query == "wireless"
    assert plan.filters.region == "South"
    assert plan.semantic_query == "wireless"
    assert plan.limit == 5


def test_marks_ambiguous_and_unsupported_queries_instead_of_guessing():
    missing_metric = plan_query("What are the top 5 products?")
    missing_grain = plan_query("Which had the highest profit?")
    unsupported = plan_query("Predict next year's sales")

    assert missing_metric.status == "ambiguous"
    assert "ranking metric" in missing_metric.issues[0]
    assert missing_grain.status == "ambiguous"
    assert any("rows, orders, or products" in issue for issue in missing_grain.issues)
    assert unsupported.status == "unsupported"
    assert unsupported.issues


def test_threshold_phrase_at_least_is_not_misclassified_as_a_ranking():
    plan = plan_query("Show Chairs records with sales at least 1000 in 2017")

    assert plan.status == "valid"
    assert plan.operation == "records"
    assert plan.query_type == "filter"
    assert plan.filters.numeric[0].operator == "gte"


def test_office_supplies_category_is_not_also_treated_as_supplies_subcategory():
    category_plan = plan_query(
        "Show Office Supplies records in the Central region with sales at least 500"
    )
    explicit_subcategory_plan = plan_query(
        "Show Office Supplies records in sub-category Supplies"
    )

    assert category_plan.filters.category == "Office Supplies"
    assert category_plan.filters.sub_category is None
    assert explicit_subcategory_plan.filters.category == "Office Supplies"
    assert explicit_subcategory_plan.filters.sub_category == "Supplies"


def test_plans_discount_and_shipping_duration_variations():
    discounted = plan_query("Show rows where the discount was at least 30%")
    undiscounted = plan_query("Find line items without a discount")
    slow = plan_query("Which orders took more than seven calendar days to ship?")
    discounted_by = plan_query("Which table rows were discounted by at least 30%?")
    any_discount = plan_query("Show discounted appliance rows")

    assert discounted.filters.numeric[0].metric == "discount"
    assert discounted.filters.numeric[0].value == 0.3
    assert undiscounted.filters.numeric[0].operator == "eq"
    assert slow.status == "valid"
    assert slow.filters.numeric[0].metric == "shipping_days"
    assert slow.filters.numeric[0].value == 7
    assert discounted_by.filters.numeric[0].value == 0.3
    assert any_discount.filters.numeric[0].operator == "gt"


def test_plans_grouped_totals_aggregate_rankings_and_customer_order_count():
    grouped = plan_query("Show total profit for each order year")
    states = plan_query("What are the top 3 states by total sales?")
    customer = plan_query(
        "Among Home Office customers, who had the highest total sales and how many "
        "distinct orders did that customer place?"
    )

    assert (grouped.operation, grouped.group_by, grouped.grain) == (
        "sum", "order_year", None
    )
    assert (states.operation, states.ranking_group_by, states.limit) == (
        "ranking", "State", 3
    )
    assert customer.status == "valid"
    assert customer.ranking_group_by == "Customer ID"
    assert customer.filters.segment == "Home Office"
    assert customer.include_distinct_order_count is True


def test_compound_ranking_and_total_keeps_both_intents():
    plan = plan_query(
        "For discounted appliance rows in the Central region, which three made the "
        "most profit and what was total profit?"
    )

    assert plan.operation == "sum"
    assert plan.ranking_metric == "profit"
    assert plan.limit == 3
    assert plan.query_type == "mixed"


def test_compound_rankings_preserve_requested_average_and_count_operations():
    average = plan_query(
        "For chair line items in the West, which had the highest sales and what was "
        "average profit across the matching rows?"
    )
    count = plan_query(
        "For Corporate Office Supplies rows in California, which line item had the "
        "highest sales and how many rows matched?"
    )

    assert (average.operation, average.ranking_metric) == ("mean", "sales")
    assert (count.operation, count.ranking_metric) == ("count", "sales")


def test_overlapping_location_names_prefer_the_complete_location_match():
    dimensions = {
        "state": ["New York", "Kansas"],
        "city": ["York", "Kansas City"],
    }

    new_york = plan_query(
        "How many Technology line items ordered in New York had sales above 750?",
        dimensions,
    )
    city_and_state = plan_query(
        "How many line items were ordered in Kansas City, Kansas?", dimensions
    )

    assert (new_york.filters.state, new_york.filters.city) == ("New York", None)
    assert (city_and_state.filters.state, city_and_state.filters.city) == (
        "Kansas", "Kansas City"
    )


def test_plural_category_rankings_use_grouped_dimensions_not_product_grain():
    categories = plan_query("What are the top two product categories by total quantity?")
    hyphenated = plan_query("What are the top four sub-categories by total sales?")
    unhyphenated = plan_query("Show the top 3 subcategories by total profit")

    assert (categories.status, categories.grain, categories.ranking_group_by) == (
        "valid", None, "Category"
    )
    assert (hyphenated.status, hyphenated.grain, hyphenated.ranking_group_by) == (
        "valid", None, "Sub-Category"
    )
    assert unhyphenated.ranking_group_by == "Sub-Category"
