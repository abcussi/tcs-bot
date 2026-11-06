"""Seed the database with synthetic, anonymized vendors + documents (AC-001.5).

``seed(path)`` loads ``SEED_FILE`` (JSON); if it is missing, unreadable or invalid it logs a
warning and falls back to :data:`BUILTIN_SEED`. Vendors are upserted by ``external_id`` and
documents are inserted deduped by sha256, all inside ONE ``transaction()``. Seeding does NOT
chunk or embed documents (the ingestion/rag layer does that).

All companies below are FICTIONAL. ``ticker`` values point at real listed companies used only as
*market proxies* for the vendor's sector (so the quote crawler has data to show) — they are not
the vendor itself.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from tcs_bot.config import get_settings
from tcs_bot.db.mysql import transaction
from tcs_bot.db.repositories import add_document_cur, upsert_vendor_cur
from tcs_bot.logging_setup import get_logger
from tcs_bot.schemas.domain import DocumentCreate, VendorCreate

log = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

SEED_META: dict[str, Any] = {
    "description": (
        "Synthetic, anonymized capital-markets vendor profiles and documents for Vendor Risk AI. "
        "All companies, people, figures and events are fictional."
    ),
    "ticker_note": (
        "Tickers are listed MARKET PROXIES for the vendor's sector (Google Finance "
        "'SYMBOL:EXCHANGE' "
        "pairs scraped by the crawler). They do not identify the fictional vendor."
    ),
    "version": 1,
}

BUILTIN_SEED: dict[str, Any] = {
    "_meta": SEED_META,
    "vendors": [
        # ------------------------------------------------------------------ market data
        {
            "external_id": "VND-001",
            "name": "Meridian Quotestream Data Ltd",
            "sector": "Market data provider",
            "country": "United Kingdom",
            "ticker": "ICE:NYSE",
            "criticality": "critical",
            "services": (
                "Consolidated real-time equities and fixed-income price feeds, end-of-day "
                "reference data and benchmark index calculation delivered via low-latency API."
            ),
            "documents": [
                {
                    "doc_type": "financial_report",
                    "title": "Meridian Quotestream FY2025 Annual Report (extract)",
                    "source": "synthetic://meridian/annual-report-2025",
                    "published_at": "2026-03-14",
                    "content": (
                        "Meridian Quotestream Data Ltd reported revenue of GBP 412 million for the "
                        "fiscal year ended 31 December 2025, an increase of 9.8% year on year, "
                        "driven by growth in real-time feed subscriptions and new benchmark "
                        "licensing agreements. Adjusted EBITDA rose to GBP 131 million, a margin "
                        "of 31.8%, compared with 30.2% in the prior year. Operating cash flow was "
                        "GBP 104 million and the company closed the year with cash and equivalents "
                        "of GBP 88 million and an undrawn revolving credit facility of GBP 75 "
                        "million. Net debt to EBITDA improved to 1.4x from 1.7x. Recurring "
                        "subscription revenue represents 93% of total revenue and gross retention "
                        "remained above 96%. However, customer concentration remains elevated: the "
                        "three largest clients, all global investment banks, accounted for 41% of "
                        "revenue, and the single largest client contributed 18%. Management notes "
                        "that two of these contracts are due for renewal in the second half of "
                        "2026 and that one client has issued a request for proposal to alternative "
                        "providers. Capital expenditure increased 22% to GBP 39 million as the "
                        "company migrated its ticker plant to a new co-location facility. The "
                        "board proposed a final dividend of 6.1 pence per share. The auditor "
                        "issued "
                        "an unqualified opinion. Principal risks identified include data licensing "
                        "costs from exchanges, which rose 12% during the year, cyber security, and "
                        "dependence on a small number of senior engineers for the proprietary "
                        "normalisation engine."
                    ),
                },
                {
                    "doc_type": "news",
                    "title": "Meridian feed outage delays opening auctions for several brokers",
                    "source": "synthetic://newswire/meridian-outage",
                    "published_at": "2026-05-06",
                    "content": (
                        "Several mid-sized brokers reported stale prices at the London market open "
                        "on Tuesday after Meridian Quotestream Data suffered a 47-minute "
                        "disruption "
                        "to its consolidated equities feed. According to a status notice published "
                        "by the company, a configuration change deployed to its primary ticker "
                        "plant during the overnight maintenance window caused a cascading failure "
                        "in the message normalisation layer. The automatic failover to the "
                        "secondary site did not complete because the same faulty configuration had "
                        "already been replicated. Clients that relied solely on Meridian were "
                        "forced to suspend algorithmic order routing until prices were restored. "
                        "One broker said it would review its single-source dependency. Meridian "
                        "said no data was lost or corrupted and that it had rolled back the change "
                        "within 40 minutes of detection. The company has committed to publishing "
                        "a full root-cause analysis within ten business days and to introducing "
                        "staggered deployment across sites. It is the second notable incident in "
                        "twelve months; a shorter 12-minute outage affecting fixed-income "
                        "reference data occurred last September. Industry observers noted that "
                        "operational resilience regulation now expects firms to identify and test "
                        "important business services that depend on third-party data vendors, and "
                        "that customers may request service credits under their contracts."
                    ),
                },
                {
                    "doc_type": "regulatory_filing",
                    "title": "Meridian Quotestream benchmark administrator compliance statement",
                    "source": "synthetic://regulator/meridian-bmr-2026",
                    "published_at": "2026-02-02",
                    "content": (
                        "This statement is filed by Meridian Quotestream Data Ltd in its capacity "
                        "as an authorised benchmark administrator. During the reporting period the "
                        "administrator calculated and published 214 indices, of which 11 are "
                        "classified as significant benchmarks. The oversight committee met six "
                        "times and reviewed the methodology of all significant benchmarks. Two "
                        "methodology changes were consulted on publicly before implementation. The "
                        "administrator identified one input data error in March 2025 that resulted "
                        "in the republication of a corporate bond index for a single day; affected "
                        "users were notified within four hours and the incident was reported to "
                        "the competent authority. Internal audit completed a review of the control "
                        "framework and raised three medium-rated findings relating to segregation "
                        "of duties for manual index adjustments, documentation of expert judgement "
                        "and timeliness of conflict-of-interest declarations. Remediation of two "
                        "findings has been completed; the remaining finding on expert judgement "
                        "documentation is scheduled to close by the third quarter of 2026. The "
                        "administrator maintains a business continuity plan tested annually, "
                        "and has no outstanding enforcement actions."
                    ),
                },
            ],
        },
        # ------------------------------------------------------------------ cloud / hosting
        {
            "external_id": "VND-002",
            "name": "Nimbus Ledger Cloud Inc.",
            "sector": "Cloud infrastructure and hosting",
            "country": "United States",
            "ticker": "MSFT:NASDAQ",
            "criticality": "high",
            "services": (
                "Managed private cloud, bare-metal hosting for trading and risk systems, "
                "disaster-recovery-as-a-service and managed Kubernetes in three US regions."
            ),
            "documents": [
                {
                    "doc_type": "financial_report",
                    "title": "Nimbus Ledger Cloud Q2 2026 quarterly results",
                    "source": "synthetic://nimbus/q2-2026-results",
                    "published_at": "2026-08-04",
                    "content": (
                        "Nimbus Ledger Cloud Inc. reported second-quarter revenue of USD 286 "
                        "million, up 17% year over year, with financial services customers "
                        "representing 58% of bookings. Gross margin was 61%, stable versus the "
                        "prior quarter. The company remained profitable on a GAAP basis with net "
                        "income of USD 21 million, and free cash flow reached USD 44 million. Cash "
                        "and marketable securities totalled USD 610 million against USD 300 "
                        "million of convertible notes maturing in 2029, leaving the company in a "
                        "net cash position. Remaining performance obligations grew 24% to USD 1.9 "
                        "billion, providing good revenue visibility. Management raised full-year "
                        "guidance. Capital expenditure for new data-centre capacity is expected "
                        "to reach 28% of revenue this year, above the historical range of 18-20%, "
                        "as the company builds a fourth region to satisfy customer demand for "
                        "geographic diversity. Net revenue retention was 118%. No single customer "
                        "accounted for more than 6% of revenue. Management highlighted rising "
                        "power "
                        "costs and longer lead times for GPU and networking hardware as headwinds, "
                        "and disclosed that its primary insurer reduced cyber coverage limits at "
                        "renewal, increasing the company's self-insured retention."
                    ),
                },
                {
                    "doc_type": "news",
                    "title": "Nimbus Ledger region-wide outage disrupts post-trade processing",
                    "source": "synthetic://newswire/nimbus-us-east-outage",
                    "published_at": "2026-06-18",
                    "content": (
                        "A power distribution fault at Nimbus Ledger Cloud's US-East facility "
                        "caused a region-wide outage lasting three hours and twenty minutes on "
                        "Wednesday, disrupting post-trade reconciliation and risk reporting for "
                        "several asset managers and a regional broker-dealer hosted on the "
                        "platform. Customers that had configured active-passive replication to "
                        "the company's US-Central region recovered within 45 minutes, while those "
                        "without cross-region replication waited for full restoration. Nimbus "
                        "said the uninterruptible power supply on one of two redundant feeds had "
                        "failed a maintenance test two weeks earlier and was awaiting replacement "
                        "parts, leaving the hall on a single feed when the utility fault occurred. "
                        "The company apologised and said it would issue service credits under its "
                        "99.95% availability commitment. Analysts said the financial impact on "
                        "Nimbus would be modest but that the incident may slow new wins among "
                        "regulated customers who must evidence resilience of critical ICT third "
                        "parties. Nimbus stated that recovery time objectives for its "
                        "disaster-recovery product were met for all customers who had enabled it."
                    ),
                },
                {
                    "doc_type": "regulatory_filing",
                    "title": "Nimbus Ledger SOC 2 Type II report summary (FY2026)",
                    "source": "synthetic://audit/nimbus-soc2-2026",
                    "published_at": "2026-04-30",
                    "content": (
                        "The independent service auditor examined Nimbus Ledger Cloud's "
                        "description of its managed hosting system and the suitability of design "
                        "and operating effectiveness of controls relevant to security, "
                        "availability and confidentiality for the period 1 April 2025 to 31 March "
                        "2026. The auditor expressed a qualified opinion. Two exceptions were "
                        "noted. First, in the change management control, 4 of 40 sampled "
                        "production changes lacked documented peer approval before deployment. "
                        "Second, in the logical access control, the quarterly user access review "
                        "for the infrastructure administration console was not completed for one "
                        "quarter, and three terminated contractors retained active accounts for "
                        "between 19 and 41 days after departure. No evidence of misuse was "
                        "identified. Management responded that an automated approval gate has "
                        "been added to the deployment pipeline and that access reviews are now "
                        "tracked in the governance tool with escalation to the CISO. All other "
                        "controls, including encryption at rest, vulnerability management, "
                        "backup and restoration testing and incident response, operated "
                        "effectively without exception. Complementary user entity controls "
                        "require customers to manage their own identity federation and to "
                        "configure cross-region replication for workloads with strict recovery "
                        "objectives."
                    ),
                },
            ],
        },
        # ------------------------------------------------------------------ clearing / settlement
        {
            "external_id": "VND-003",
            "name": "Clearpath Settlement Technologies AG",
            "sector": "Clearing and settlement technology",
            "country": "Germany",
            "ticker": "IBM:NYSE",
            "criticality": "critical",
            "services": (
                "Post-trade clearing, collateral management and securities settlement "
                "messaging platform (SaaS and on-premise) connected to CSDs and CCPs."
            ),
            "documents": [
                {
                    "doc_type": "financial_report",
                    "title": "Clearpath Settlement Technologies 2025 financial statements",
                    "source": "synthetic://clearpath/fs-2025",
                    "published_at": "2026-04-22",
                    "content": (
                        "Clearpath Settlement Technologies AG recorded revenue of EUR 96 million "
                        "for 2025, down 7% from EUR 103 million, as two large clients completed "
                        "migrations to in-house platforms. The company reported an operating loss "
                        "of EUR 18 million and a net loss of EUR 27 million, its third consecutive "
                        "annual loss. Cash and equivalents fell to EUR 9 million from EUR 31 "
                        "million. Total borrowings increased to EUR 142 million following a EUR 40 "
                        "million term-loan top-up, and net debt to EBITDA rose to 7.9x from 4.2x a "
                        "year earlier. The company breached its leverage covenant at the September "
                        "test date and obtained a waiver from its lenders, conditional on a "
                        "capital raise by the end of the second quarter of 2026. In its report, "
                        "the auditor included a material uncertainty related to going concern, "
                        "stating that the company's ability to continue depends on completing the "
                        "planned equity raise or refinancing its facilities. Management believes "
                        "the going-concern basis remains appropriate given advanced discussions "
                        "with a strategic investor. On the positive side, the order backlog for "
                        "its next-generation settlement engine stood at EUR 64 million and "
                        "recurring maintenance revenue grew 5%. Headcount was reduced by 11% and "
                        "research spending was cut, which management acknowledges may delay the "
                        "delivery of regulatory change releases."
                    ),
                },
                {
                    "doc_type": "news",
                    "title": "Clearpath talks with strategic investor stall, sources say",
                    "source": "synthetic://newswire/clearpath-investor",
                    "published_at": "2026-07-09",
                    "content": (
                        "Negotiations between Clearpath Settlement Technologies and a prospective "
                        "strategic investor have stalled over valuation and governance rights, "
                        "according to two people familiar with the matter. The German "
                        "post-trade software firm had told lenders it expected to close a capital "
                        "injection by the end of June as a condition of a covenant waiver. A "
                        "company spokesperson said discussions were continuing with several "
                        "parties and that operations were unaffected. Separately, Clearpath's "
                        "chief technology officer and head of client delivery both left the "
                        "company in the past quarter, and recruitment for the roles is ongoing. "
                        "Clients contacted said settlement processing had remained stable, with "
                        "straight-through processing rates above 98%, but some reported slower "
                        "responses to support tickets and a delay to the release supporting the "
                        "shortened settlement cycle in European markets. One custodian said it had "
                        "requested source-code escrow verification and an updated exit plan from "
                        "Clearpath as part of its third-party risk review. Credit insurers have "
                        "reportedly reduced cover for suppliers to the company."
                    ),
                },
                {
                    "doc_type": "regulatory_filing",
                    "title": "Clearpath ad-hoc disclosure: lender standstill and liquidity update",
                    "source": "synthetic://regulator/clearpath-adhoc-2026-08",
                    "published_at": "2026-08-12",
                    "content": (
                        "Clearpath Settlement Technologies AG announces that its lenders have "
                        "agreed to a standstill until 30 November 2026, extending the deadline for "
                        "the equity raise required under the covenant waiver. In exchange, the "
                        "company has agreed to a margin increase of 250 basis points, enhanced "
                        "monthly liquidity reporting and the appointment of a chief restructuring "
                        "officer. The company currently projects sufficient liquidity to fund "
                        "operations until the end of the first quarter of 2027 on the basis of "
                        "its existing facilities and expected client receipts. Failure to complete "
                        "a capital measure would likely result in an event of default. The "
                        "management board continues to pursue a capital increase, a sale of the "
                        "on-premise maintenance business and other strategic options. Clearpath "
                        "confirms that client assets and settlement instructions are not held on "
                        "its balance sheet and that its software escrow arrangements with a "
                        "third-party agent remain in force. The company remains in compliance with "
                        "its information security certification, which was renewed in May 2026 "
                        "with no major non-conformities."
                    ),
                },
            ],
        },
        # ------------------------------------------------------------------ KYC / AML SaaS
        {
            "external_id": "VND-004",
            "name": "Aurelian Identity Labs SAS",
            "sector": "KYC / AML compliance SaaS",
            "country": "France",
            "ticker": None,
            "criticality": "high",
            "services": (
                "Client onboarding, KYC document verification, sanctions and PEP screening and "
                "transaction monitoring for investment firms (SaaS)."
            ),
            "documents": [
                {
                    "doc_type": "financial_report",
                    "title": "Aurelian Identity Labs 2025 investor update",
                    "source": "synthetic://aurelian/investor-update-2025",
                    "published_at": "2026-02-27",
                    "content": (
                        "Aurelian Identity Labs, a privately held company backed by growth equity "
                        "investors, reported annual recurring revenue of EUR 58 million at the end "
                        "of 2025, up 34% year over year. The company reached EBITDA breakeven in "
                        "the fourth quarter for the first time, while full-year EBITDA was a loss "
                        "of EUR 4 million compared with a loss of EUR 15 million in 2024. Cash on "
                        "hand of EUR 47 million provides more than three years of runway at the "
                        "current burn rate, and the company has no financial debt. Gross revenue "
                        "retention was 94% and net revenue retention 121%. The customer base grew "
                        "to 310 regulated institutions, with no customer exceeding 5% of revenue. "
                        "Investment in the screening data team increased by 40% to improve "
                        "sanctions list coverage and reduce false positives. The update discloses "
                        "that the company is responding to a supervisory inquiry and has made a "
                        "provision of EUR 1.2 million for a data protection penalty (see "
                        "regulatory filing). Management also flags dependence on a single "
                        "third-party watch-list data supplier and plans to onboard a second "
                        "supplier in 2026."
                    ),
                },
                {
                    "doc_type": "news",
                    "title": "Aurelian screening gap left sanctions updates unapplied for 36h",
                    "source": "synthetic://newswire/aurelian-screening-gap",
                    "published_at": "2026-03-19",
                    "content": (
                        "Aurelian Identity Labs has notified clients that a defect in its list "
                        "ingestion pipeline prevented two consecutive sanctions list updates from "
                        "being applied to its screening engine for approximately 36 hours. During "
                        "this window, newly designated individuals and entities would not have "
                        "generated alerts during onboarding or batch rescreening. The company said "
                        "it has rescreened all affected client populations and identified four "
                        "potential matches, which were escalated to the relevant clients for "
                        "review. Aurelian attributed the failure to a schema change by its "
                        "upstream data supplier that was not caught by validation checks, and has "
                        "since added completeness monitoring with automatic alerts when list "
                        "volumes deviate from expected ranges. Compliance officers at two "
                        "investment firms said they were assessing whether the gap must be "
                        "reported to their own regulators. Despite the incident, industry "
                        "analysts rate Aurelian's matching accuracy and onboarding experience "
                        "among the strongest in the segment, and the company recently won a "
                        "mandate from a large European asset manager."
                    ),
                },
                {
                    "doc_type": "regulatory_filing",
                    "title": "Data protection authority decision regarding Aurelian Identity Labs",
                    "source": "synthetic://regulator/aurelian-dpa-decision",
                    "published_at": "2026-01-15",
                    "content": (
                        "Following a complaint and subsequent investigation, the data protection "
                        "authority found that Aurelian Identity Labs retained identity document "
                        "images and biometric verification data for longer than necessary for the "
                        "purposes for which they were processed, in some cases more than five "
                        "years "
                        "after the end of the client relationship. The authority also found that "
                        "the company had not carried out an adequate data protection impact "
                        "assessment for its facial liveness check prior to launch, and that "
                        "information provided to data subjects was incomplete. The authority "
                        "imposed an administrative fine of EUR 1.2 million and ordered the company "
                        "to bring its retention practices into compliance within six months, "
                        "including deletion of data held beyond the defined retention period. The "
                        "authority noted the company's cooperation and the absence of evidence "
                        "that data had been accessed by unauthorised parties. Aurelian has stated "
                        "it will not appeal, has appointed a new data protection officer and has "
                        "engaged an external firm to verify completion of the deletion programme."
                    ),
                },
            ],
        },
        # ------------------------------------------------------------------ trading software
        {
            "external_id": "VND-005",
            "name": "Tidewater Execution Systems LLC",
            "sector": "Trading software (OMS/EMS)",
            "country": "United States",
            "ticker": "ORCL:NYSE",
            "criticality": "medium",
            "services": (
                "Multi-asset order and execution management system, FIX connectivity and "
                "pre-trade risk checks for buy-side and sell-side desks."
            ),
            "documents": [
                {
                    "doc_type": "financial_report",
                    "title": "Tidewater Execution Systems FY2025 audited financial summary",
                    "source": "synthetic://tidewater/fy2025-summary",
                    "published_at": "2026-03-03",
                    "content": (
                        "Tidewater Execution Systems LLC generated revenue of USD 174 million in "
                        "fiscal 2025, an increase of 11%, with 88% derived from multi-year "
                        "subscription contracts. Operating margin expanded to 24% and net income "
                        "reached USD 33 million. The company holds USD 96 million of cash, has no "
                        "outstanding borrowings and maintains an undrawn USD 50 million revolving "
                        "facility. Days sales outstanding improved to 48 days. The top ten clients "
                        "represented 29% of revenue, and the largest single client 5%. Research "
                        "and development spending was 21% of revenue, focused on fixed-income "
                        "execution workflows and cloud deployment. The auditor issued an "
                        "unqualified opinion with no key audit matters beyond revenue recognition "
                        "on multi-element contracts. The company is majority owned by its "
                        "founders, and the founder-CEO remains closely involved in product "
                        "decisions; the board has adopted a formal succession plan during the "
                        "year to address key-person risk. Management expects continued "
                        "double-digit growth and modest margin expansion in 2026."
                    ),
                },
                {
                    "doc_type": "news",
                    "title": "Tidewater wins industry award, expands fixed-income execution suite",
                    "source": "synthetic://newswire/tidewater-award",
                    "published_at": "2026-05-21",
                    "content": (
                        "Tidewater Execution Systems was named best buy-side execution platform at "
                        "an industry technology awards ceremony this week, citing low latency, "
                        "reliability and a straightforward upgrade process. The company also "
                        "announced the general availability of its fixed-income execution module, "
                        "which aggregates liquidity from multiple electronic venues and supports "
                        "request-for-quote workflows. Clients reported platform availability of "
                        "99.99% over the past year with no significant incidents. Separately, a "
                        "security researcher disclosed a moderate-severity vulnerability in an "
                        "open-source FIX parsing library used by many trading platforms, "
                        "including Tidewater's. The company said it had released a patch within "
                        "five days of disclosure and that exploitation would require "
                        "authenticated network access; it advised on-premise clients to upgrade "
                        "at their next maintenance window. Some clients running older on-premise "
                        "versions noted that upgrades still require several weeks of testing, "
                        "leaving a window of exposure. Tidewater said it would offer extended "
                        "support and a hot-fix package that can be applied without a full "
                        "regression cycle, and that its hosted clients were patched automatically "
                        "within forty-eight hours."
                    ),
                },
                {
                    "doc_type": "regulatory_filing",
                    "title": "Tidewater SOC 2 Type II and ISO 27001 surveillance summary",
                    "source": "synthetic://audit/tidewater-soc2-2026",
                    "published_at": "2026-06-10",
                    "content": (
                        "The independent service auditor issued an unqualified SOC 2 Type II "
                        "opinion on Tidewater Execution Systems' hosted order management platform "
                        "covering security, availability and processing integrity for the twelve "
                        "months ended 31 March 2026. No exceptions were noted in the testing of "
                        "operating effectiveness. The ISO 27001 surveillance audit completed in "
                        "May 2026 raised one minor non-conformity relating to the timeliness of "
                        "supplier security reviews for two low-risk subcontractors, which has "
                        "since been closed. Business continuity testing in the period included a "
                        "full failover of the hosted platform to the secondary data centre, "
                        "completed within the documented recovery time objective of 30 minutes "
                        "with zero data loss. Penetration testing by an external firm identified "
                        "no critical or high findings. The report notes that clients remain "
                        "responsible for configuring pre-trade risk limits and for reviewing "
                        "user entitlements within their own tenant. The company is not subject to "
                        "any regulatory enforcement action."
                    ),
                },
            ],
        },
    ],
}


class _SeedDocument(DocumentCreate):
    pass


class _SeedVendor(VendorCreate):
    documents: list[_SeedDocument] = []


def _validate(data: Any) -> dict[str, Any]:
    """Return a normalized copy or raise ``ValueError`` if the structure is not a valid seed."""
    if not isinstance(data, dict) or not isinstance(data.get("vendors"), list):
        raise ValueError("seed data must be an object with a 'vendors' list")
    try:
        vendors = [_SeedVendor.model_validate(v) for v in data["vendors"]]
    except ValidationError as exc:
        raise ValueError(f"invalid seed vendor: {exc.error_count()} validation error(s)") from exc
    out = {k: v for k, v in data.items() if k != "vendors"}
    out["vendors"] = [v.model_dump(mode="json") for v in vendors]
    return out


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute() and not p.exists() and (_PROJECT_ROOT / p).exists():
        return _PROJECT_ROOT / p
    return p


def load_seed_data(path: str | None = None) -> dict:
    """Load and validate the seed JSON; fall back to :data:`BUILTIN_SEED` with a warning."""
    target = path if path is not None else get_settings().seed_file
    resolved = _resolve(target)
    try:
        with resolved.open(encoding="utf-8") as fh:
            data = _validate(json.load(fh))
        log.info("db.seed.loaded_file", path=str(resolved), vendors=len(data["vendors"]))
        return data
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        log.warning(
            "db.seed.fallback_builtin",
            path=str(resolved),
            reason=type(exc).__name__,
            detail=str(exc)[:200],
        )
        return _validate(copy.deepcopy(BUILTIN_SEED))


def seed(path: str | None = None) -> dict[str, int]:
    """Upsert vendors and insert (deduped) documents in one transaction.

    Returns ``{"vendors": <vendors upserted>, "documents": <documents newly inserted>}``;
    a second run therefore reports ``documents == 0``.
    """
    data = load_seed_data(path)
    n_vendors = 0
    n_docs = 0
    with transaction() as cur:
        for raw in data["vendors"]:
            v = _SeedVendor.model_validate(raw)
            vendor_id = upsert_vendor_cur(cur, VendorCreate(**v.model_dump(exclude={"documents"})))
            n_vendors += 1
            for d in v.documents:
                _, inserted = add_document_cur(cur, vendor_id, DocumentCreate(**d.model_dump()))
                n_docs += int(inserted)
    log.info("db.seed.done", vendors=n_vendors, documents=n_docs)
    return {"vendors": n_vendors, "documents": n_docs}


def write_seed_file(path: str | Path) -> None:
    """Dump :data:`BUILTIN_SEED` to ``path`` (used to (re)generate data/seed/vendors.json)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(BUILTIN_SEED, indent=2, ensure_ascii=False) + "\n", "utf-8")
