from __future__ import annotations

from vifinqa.nlu.company import Company, CompanyResolver
from vifinqa.nlu.query_spec import parse_query_spec
from vifinqa.nlu.scope import detect_scope_intent
from vifinqa.nlu.temporal import extract_temporal_mentions
from vifinqa.nlu.unit_target import detect_target_unit


def test_target_unit_prefers_longest_phrase() -> None:
    assert detect_target_unit("Giá trị là bao nhiêu nghìn tỷ đồng?").name == "TRILLION_VND"
    assert detect_target_unit("Giá trị là bao nhiêu trăm tỷ đồng?").name == "HUNDRED_BILLION_VND"
    assert detect_target_unit("Có bao nhiêu triệu cổ phiếu?").name == "MILLION_SHARES"


def test_target_unit_uses_answer_clause_instead_of_filter_percent() -> None:
    cases = {
        "Các công ty có biên lợi nhuận trên 10%, tổng doanh thu là bao nhiêu nghìn tỷ đồng?": (
            "TRILLION_VND"
        ),
        "Nếu doanh thu giảm 10%, có bao nhiêu đơn vị có hệ số dưới 1,5 lần?": "COUNT",
        "Nếu EBIT giảm 15%, hệ số thanh toán lãi vay là bao nhiêu lần?": "RATIO",
        "Nếu VND biến động 5%, mức giảm lợi nhuận là bao nhiêu tỷ đồng?": "BILLION_VND",
        "Năm nào có tỷ trọng chi phí lãi (%) cao nhất?": "YEAR",
        "Biên lợi nhuận sau khi doanh thu giảm 5% là bao nhiêu phần trăm?": "PERCENT",
        "Tỷ lệ tiền gửi bằng VND trên tổng tiền gửi là bao nhiêu %?": "PERCENT",
        "Tính tăng trưởng (%) của số dư bằng VND giữa hai năm.": "PERCENT",
        "Cổ phiếu ABC có số năm dòng tiền âm là bao nhiêu trong giai đoạn 2020-2024?": ("COUNT"),
    }
    for question, expected in cases.items():
        assert detect_target_unit(question).name == expected


def test_scope_is_not_defaulted_when_unspecified() -> None:
    assert detect_scope_intent("Tổng tài sản của VNM năm 2024?").scope is None
    assert detect_scope_intent("Tổng tài sản công ty mẹ VNM năm 2024?").scope == "separate"


def test_start_of_year_uses_prior_period_column_in_same_report() -> None:
    mentions = extract_temporal_mentions("Tài sản đầu năm 2019 là bao nhiêu?")
    assert mentions[0].preferred_report_year == 2019
    assert mentions[0].column_role == "prior_period"


def test_year_range_expands_intermediate_report_years() -> None:
    mentions = extract_temporal_mentions("Giai doan 2020-2024")
    assert {mention.preferred_report_year for mention in mentions} == {
        2020,
        2021,
        2022,
        2023,
        2024,
    }


def test_long_company_alias_shadows_ticker_inside_another_company_name() -> None:
    resolver = CompanyResolver(
        (
            Company("FPT", "CTCP FPT"),
            Company("FTS", "CTCP Chứng khoán FPT"),
        )
    )
    matches = resolver.resolve("Lợi nhuận của CTCP Chứng khoán FPT năm 2023?")
    assert [match.company.ticker for match in matches] == ["FTS"]


def test_separate_ticker_tokens_are_preserved_for_multi_entity_question() -> None:
    resolver = CompanyResolver(
        (
            Company("FPT", "CTCP FPT"),
            Company("FTS", "CTCP Chứng khoán FPT"),
        )
    )
    matches = resolver.resolve("So sánh FPT và FTS năm 2023")
    assert {match.company.ticker for match in matches} == {"FPT", "FTS"}


def test_counterparty_is_not_routed_as_primary_entity() -> None:
    resolver = CompanyResolver(
        (
            Company("GAS", "Tổng Công ty Khí Việt Nam - CTCP"),
            Company("POW", "Tổng Công ty Điện lực Dầu khí Việt Nam - CTCP"),
        )
    )
    spec = parse_query_spec(
        "Của Tổng Công ty Khí Việt Nam - CTCP, giá trị bán hàng với "
        "Tổng Công ty Điện lực Dầu khí Việt Nam là bao nhiêu?",
        resolver,
    )
    assert [entity.ticker for entity in spec.entities] == ["GAS"]


def test_company_parent_marker_selects_primary_over_payee() -> None:
    resolver = CompanyResolver(
        (
            Company("EVF", "Công ty Tài chính Tổng hợp Cổ phần Điện lực"),
            Company("POW", "Tổng Công ty Điện lực Dầu khí Việt Nam - CTCP"),
        )
    )
    spec = parse_query_spec(
        "Khoản phải trả cho Tập đoàn Điện lực Việt Nam của công ty mẹ "
        "Tổng Công ty Điện lực Dầu khí Việt Nam - CTCP năm 2024?",
        resolver,
    )
    assert [entity.ticker for entity in spec.entities] == ["POW"]


def test_cohort_ownership_phrase_is_not_collapsed_to_first_ticker() -> None:
    resolver = CompanyResolver(
        tuple(Company(ticker, f"CTCP {ticker}") for ticker in ("HPX", "KBC", "NVL", "PDR", "SCR"))
    )
    spec = parse_query_spec(
        "Của các doanh nghiệp nhóm HPX, KBC, NVL, PDR và SCR, có bao nhiêu doanh nghiệp "
        "có biên dòng tiền từ hoạt động kinh doanh âm?",
        resolver,
    )
    assert {entity.ticker for entity in spec.entities} == {"HPX", "KBC", "NVL", "PDR", "SCR"}


def test_common_verb_phrase_does_not_trigger_short_company_alias() -> None:
    resolver = CompanyResolver(
        (
            Company("HPG", "CTCP Tap doan Hoa Phat"),
            Company("PDR", "CTCP Phat trien Bat dong san Phat Dat"),
        )
    )
    matches = resolver.resolve("Tai nam ma Hoa Phat dat doanh thu cao nhat")
    assert [match.company.ticker for match in matches] == ["HPG"]
    assert [match.company.ticker for match in resolver.resolve("Cong ty Phat Dat")] == ["PDR"]


def test_comparison_of_two_parent_companies_preserves_both_entities() -> None:
    resolver = CompanyResolver(
        (
            Company("CTG", "Ngan hang TMCP Cong Thuong Viet Nam"),
            Company("MBB", "Ngan hang TMCP Quan doi"),
        )
    )
    spec = parse_query_spec(
        "Chenh lech cua cong ty me Ngan hang TMCP Cong Thuong Viet Nam va cong ty me "
        "Ngan hang TMCP Quan doi nam 2024?",
        resolver,
    )
    assert {entity.ticker for entity in spec.entities} == {"CTG", "MBB"}
