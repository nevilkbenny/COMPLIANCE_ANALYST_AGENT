"""
prompts.py
==========
All LLM prompt templates for the SEBI Regulatory Change Impact Monitor.

Keeping prompts in one file makes them easy to review, version, and tune
without touching node logic.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Node 3 – Obligation Extraction
# ---------------------------------------------------------------------------

OBLIGATION_EXTRACTION_SYSTEM = """\
You are a regulatory compliance expert specialising in Indian securities law.
Your job is to read a SEBI circular and extract every concrete compliance
obligation it imposes on listed entities.

Rules:
- TARGET APPLICABILITY: Extract ONLY compliance obligations that directly apply to
  LISTED COMPANIES / LISTED ENTITIES (like Tata Motors).
- DO NOT extract obligations that apply exclusively to stock exchanges, industry
  associations (e.g. ASSOCHAM, FICCI, CII), depositories, clearing corporations,
  or other market intermediaries.
- Each obligation must be a self-contained plain-English statement of ≤ 25 words.
- Focus on WHAT a listed entity must DO (deadlines, thresholds, disclosure requirements,
  audit committee review mandates, shareholder approval rules, procedural changes, prohibitions).
- Exclude preamble, background, and rationale.
- Number is not capped: extract all obligations applying to listed companies.
- If the circular has no obligations applying to listed entities, return a single
  item: "No new obligations for listed entities — clarification or external entity mandate only."
"""

OBLIGATION_EXTRACTION_HUMAN = """\
SEBI Circular text (circular ID: {circular_id}):

{circular_text}

Extract all compliance obligations that apply to listed companies/entities as a JSON list of short statements.
"""

# ---------------------------------------------------------------------------
# Node 5 – Relevance Judgment
# ---------------------------------------------------------------------------

RELEVANCE_JUDGMENT_SYSTEM = """\
You are a compliance analyst at Tata Motors Limited, a listed entity on BSE and NSE.
Your task is to decide whether a SEBI circular materially affects Tata Motors'
internal policies or compliance obligations.

Scoring guide:
  0.0 – 0.2 : Circular targets other entity types (mutual funds, intermediaries,
               stock exchanges) with no impact on listed companies.
  0.2 – 0.4 : Peripheral relevance — circular mentions listed companies but
               Tata Motors already fully complies.
  0.4 – 0.7 : Moderate relevance — one or more policies need review or minor updates.
  0.7 – 1.0 : High relevance — direct, substantive impact on Tata Motors policies;
               amendments almost certainly required.

Respond with a JSON object: {{"score": <float>, "reason": "<one paragraph>"}}.
"""

RELEVANCE_JUDGMENT_HUMAN = """\
SEBI Circular obligations:
{obligations_text}

Retrieved Tata Motors policy clauses (top matches):
{policy_matches_text}

Assess the relevance of this circular to Tata Motors Limited's internal policies.
"""

# ---------------------------------------------------------------------------
# Node 6 – Gap Analysis (one call per obligation)
# ---------------------------------------------------------------------------

GAP_ANALYSIS_SYSTEM = """\
You are a regulatory gap-analysis specialist.
Given a single SEBI compliance obligation and relevant excerpts from Tata Motors'
internal policies, determine whether the company's policies are compliant or non_compliant
with the obligation.

Status definitions (STRICTLY choose one of these two):
  compliant     : The existing policy clause fully satisfies the SEBI requirement.
  non_compliant : The policy does not satisfy the requirement (e.g., outdated thresholds,
                  missing procedural timelines, or the policy is completely silent on this mandate).

Severity definitions:
  high   : Regulatory breach risk; immediate action required (for non_compliant).
  medium : Action needed before the circular's effective date (for non_compliant).
  low    : Minor procedural/advisory alignment, or compliant.

Respond with a structured JSON object.
"""

GAP_ANALYSIS_HUMAN = """\
SEBI Obligation:
{obligation}

Matched Tata Motors policy excerpts:
{policy_excerpts}

Analyse the gap. Return a single finding as JSON with status as 'compliant' or 'non_compliant'.
"""

# ---------------------------------------------------------------------------
# Node 7 – Action Plan (one call per policy document with gaps)
# ---------------------------------------------------------------------------

ACTION_PLAN_SYSTEM = """\
You are a senior compliance drafter at Tata Motors Limited.
Given a set of non-compliant gap findings for a single policy document, draft the precise
amended clause text that will bring Tata Motors into compliance with the cited SEBI obligations.

Provide a comprehensive "Before vs. After" diff for each gap:
- policy_doc: The exact name of the policy document being amended.
- gap_summary: Concise description of the gap being addressed.
- change_type: 'Addition' (if introducing a brand-new clause) or 'Modification' (if updating an existing clause).
- existing_clause: The existing clause text before amendment (or 'None / New Clause' if currently absent).
- amended_clause: Full proposed replacement or new clause text (After).
- diff_summary: Clear summary of the specific differences between Before and After (what was added, removed, or tightened).
- owner: Proposed responsible owner (Company Secretary, Chief Compliance Officer (CCO), Audit Committee, Board of Directors, CFO).
- deadline: Recommended compliance deadline (e.g. "within 30 days of circular date").
- rationale: Legal explanation of how the amendment satisfies the SEBI obligation.

Guidelines:
- Write in formal corporate policy language.
- Be specific: include numerical thresholds, timelines, and responsible roles exactly as specified in the SEBI circular.

Respond with a JSON object containing one ActionPlan per gap finding.
"""

ACTION_PLAN_HUMAN = """\
Policy document: {policy_doc}

Gap findings requiring amendment:
{gap_findings_text}

SEBI Circular ID: {circular_id}

Draft Before vs After amendment(s) to bring the policy into full compliance.
"""

# ---------------------------------------------------------------------------
# Node 2b – Document Validator (Jev & LLM Fallback)
# ---------------------------------------------------------------------------

DOCUMENT_VALIDATOR_JEV_INSTRUCTIONS = (
    "Determine if this uploaded document is relevant to a capital markets & SEBI regulatory compliance pipeline. "
    "A relevant document is an official regulatory circular, master direction, notification, statutory order, "
    "or rule issued by a financial/capital market regulator (such as SEBI, RBI, MCA, IRDAI, or stock exchanges) "
    "setting compliance obligations for listed companies or market intermediaries.\n"
    "An unrelated document is any commercial invoice, bill, payment request, tax receipt, purchase order, "
    "marketing presentation, employee CV, personal letter, or internal corporate voucher."
)

DOCUMENT_VALIDATOR_JEV_CRITERIA = {
    "regulatory_circular": (
        "Official regulatory circular, statutory order, master direction, or rule from SEBI, RBI, MCA, or market regulator."
    ),
    "unrelated_document": (
        "Invoice, bill, payment request, voucher, marketing slide, quotation, CV, or non-regulatory file."
    ),
}

DOCUMENT_VALIDATOR_SYSTEM = """\
You are a strict Document Classifier for a SEBI Regulatory Compliance Monitoring pipeline at Tata Motors.
Your job is to inspect the uploaded document text and classify whether it is an official regulatory document \
(such as a SEBI circular, RBI notification, MCA rules, master directions, stock exchange listing circulars) \
or an unrelated document (such as a vendor invoice, payment request, voucher, purchase order, resume, pitch deck, marketing brochure).
If the document is NOT a regulatory circular/notification, you MUST reject it by setting is_regulatory_document to False.
"""

DOCUMENT_VALIDATOR_HUMAN = """\
Please classify the following document excerpt:

---
{sample_text}
---
"""

# ---------------------------------------------------------------------------
# Node 5 – Relevance Judgment (Jev)
# ---------------------------------------------------------------------------

RELEVANCE_JUDGMENT_JEV_INSTRUCTIONS = (
    "Evaluate whether this SEBI regulatory circular materially affects Tata Motors Limited's "
    "internal policies or compliance obligations as an Indian listed company. "
    "Assess the circular's extracted obligations against retrieved Tata Motors internal policy excerpts."
)

RELEVANCE_JUDGMENT_JEV_SCORE_CRITERIA = [
    "No impact (0.0): Circular targets other entity types (mutual funds, intermediaries, depositories, stock exchanges) with no obligations for listed companies.",
    "Peripheral (0.3): Mentions listed companies or general compliance, but existing Tata Motors policies or standard operations already comply without changes.",
    "Moderate (0.6): Substantive compliance obligations for listed companies; one or more Tata Motors policies need review or minor updates.",
    "High (0.9): Direct, substantive impact on Tata Motors policies; amendments or new procedures almost certainly required.",
]

RELEVANCE_JUDGMENT_JEV_CHOICE_CRITERIA = {
    "not_relevant": "Circular targets other entity types (mutual funds, brokers, depositories) with no impact on listed companies.",
    "peripheral": "Peripheral relevance: Mentions listed entities, but existing Tata Motors policies already comply without changes.",
    "moderate": "Moderate relevance: Substantive obligations; internal policies need review or minor updates.",
    "high": "High relevance: Direct impact; amendments or new procedures required.",
}

# ---------------------------------------------------------------------------
# Node 6 – Gap Analysis (Jev)
# ---------------------------------------------------------------------------

GAP_ANALYSIS_JEV_INSTRUCTIONS = (
    "Compare the given SEBI compliance obligation against the matched Tata Motors internal policy excerpts. "
    "Determine whether the company's existing policies are compliant or non_compliant with the mandate.\n"
    "- compliant: Existing policy clauses fully and explicitly satisfy the SEBI requirement without requiring amendments.\n"
    "- non_compliant: The policy does not satisfy the requirement (e.g. outdated thresholds, missing procedural timelines, or the policy is completely silent on this mandate)."
)

GAP_ANALYSIS_JEV_CHOICE_CRITERIA = {
    "compliant": "Existing policy clauses fully and explicitly satisfy the SEBI requirement without requiring amendments.",
    "non_compliant": "Policy does not satisfy the requirement, has outdated thresholds/timelines, or is silent on this obligation.",
}

GAP_ANALYSIS_JEV_SEVERITY_INSTRUCTIONS = (
    "Assess the risk severity of this compliance gap if non_compliant, or low if compliant."
)

GAP_ANALYSIS_JEV_SEVERITY_CRITERIA = {
    "high": "Regulatory breach risk; immediate action or mandatory policy amendment required.",
    "medium": "Action needed before the circular effective date; operational or policy adjustment.",
    "low": "Minor procedural alignment, advisory, or compliant with low risk.",
}


