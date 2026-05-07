"""Generate MetaClaw-Bench Design Analysis Report as a Word document."""

from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

OUTPUT_PATH = "/home/user/MetaClaw/docs/MetaClaw-Bench_Design_Analysis.docx"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return h


def add_para(doc, text, bold=False, italic=False, size=None, indent=None):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.left_indent = Inches(indent)
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    return p


def add_bullet(doc, text, level=0):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Inches(0.25 * (level + 1))
    run = p.add_run(text)
    return p


def add_code_block(doc, code_text):
    """Add a shaded code block paragraph."""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.4)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    # Light grey shading
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), "F2F2F2")
    pPr.append(shd)
    run = p.add_run(code_text)
    run.font.name = "Courier New"
    run.font.size = Pt(9)
    return p


def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.LEFT

    # Header row
    hdr = table.rows[0]
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        # Header background
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), "D9E1F2")
        tcPr.append(shd)

    # Data rows
    for r_idx, row_data in enumerate(rows):
        row = table.rows[r_idx + 1]
        for c_idx, cell_text in enumerate(row_data):
            row.cells[c_idx].text = cell_text

    # Column widths
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Inches(w)
    return table


# ---------------------------------------------------------------------------
# Document build
# ---------------------------------------------------------------------------

def build_report():
    doc = Document()

    # Page margins
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(3.0)
        section.right_margin = Cm(2.5)

    # Default font
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)

    # ── Title ────────────────────────────────────────────────────────────────
    title = doc.add_heading("MetaClaw-Bench Design Analysis Report", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle = doc.add_paragraph("A Deep Dive into the 30-Day Benchmark Construction Methodology")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.runs[0].italic = True
    subtitle.runs[0].font.size = Pt(12)
    doc.add_paragraph()

    # ── Section 1 ────────────────────────────────────────────────────────────
    set_heading(doc, "1. What Does This Benchmark Actually Measure?", 1)

    add_para(doc,
        "MetaClaw-Bench does not test knowledge. It measures the acquisition and "
        "retention of behavioral preferences — specifically, whether an agent can "
        "internalize a set of workplace conventions through interaction and continue "
        "applying them autonomously in new tasks without any explicit reminders.")

    add_para(doc,
        "The benchmark constructs a fictional workplace scenario (Orion Tech, a B2B SaaS "
        "company; the agent serves as the AI assistant of Alex Zhang, Backend Tech Lead). "
        "Over 30 workdays, five conventions (P1–P5) are gradually revealed through task "
        "feedback. The core question is:")

    q = doc.add_paragraph()
    q.paragraph_format.left_indent = Inches(0.5)
    r = q.add_run(
        '"Can the agent, having observed a rule through failure and correction, '
        'apply it correctly in future tasks — without being told again?"')
    r.italic = True
    r.bold = True

    add_para(doc,
        "This makes it a behavioral alignment measurement tool, not a Q&A or coding "
        "capability benchmark.")

    # ── Section 2 ────────────────────────────────────────────────────────────
    set_heading(doc, "2. The Five Preferences (P1–P5)", 1)

    add_para(doc,
        "Five conventions are introduced incrementally. Each is designed to be "
        "programmatically verifiable — a deliberate engineering choice that eliminates "
        "LLM-as-judge noise from the scoring pipeline.")

    doc.add_paragraph()
    add_table(doc,
        headers=["#", "Name", "Rule"],
        rows=[
            ("P1", "Datetime Format",
             "All time fields must use ISO 8601 with +08:00 timezone offset.\n"
             "e.g. 2026-03-16T09:30:00+08:00"),
            ("P2", "File Naming",
             "Output files must follow YYYYMMDD_snake_case.ext convention.\n"
             "e.g. 20260330_standup_notes.json"),
            ("P3", "Metadata Completeness",
             "All output files must contain metadata fields: created_at, author, status.\n"
             "Format differs by file type: JSON uses top-level 'meta' object; "
             "Markdown uses YAML frontmatter; Python uses module docstring Meta section."),
            ("P4", "Modify vs. Create Boundary",
             "Modify existing files in-place; do not recreate them. "
             "Only create new files for genuinely new output."),
            ("P5", "Completion Log",
             "After finishing each task, append one entry to done.log:\n"
             "[DONE] <ISO8601-time> | <task_id> | <summary (≤80 chars)>"),
        ],
        col_widths=[0.4, 1.4, 4.5],
    )
    doc.add_paragraph()

    # ── Section 3 ────────────────────────────────────────────────────────────
    set_heading(doc, "3. The 30-Day Progression: Arcs and Accumulation", 1)

    set_heading(doc, "3.1  The Arc Mechanism", 2)

    add_para(doc,
        "Each day in all_tests.json carries two fields: arc (letter A–F) and "
        "preference_tags (list of active rules). Together they define the cumulative "
        "difficulty curve:")

    doc.add_paragraph()
    add_table(doc,
        headers=["Arc", "Days (small)", "Active Rules", "Design Intent"],
        rows=[
            ("A", "Day 01–02", "P1",          "Introduce P1; cross-domain transfer test"),
            ("B", "Day 03–04", "P1 + P2",     "Introduce P2; joint P1+P2 validation"),
            ("C", "Day 05–06", "P1 + P2 + P3","Introduce P3; three-rule comprehensive"),
            ("D", "Day 07–08", "P1–P4",       "Introduce P4; four-rule comprehensive"),
            ("E", "Day 09–10", "P1–P5",       "Introduce P5; full five-rule test"),
            ("F", "Day 11–12", "P1–P5",       "Zero-hint final exam; hardest scenarios"),
        ],
        col_widths=[0.5, 1.3, 1.6, 3.3],
    )
    doc.add_paragraph()

    set_heading(doc, "3.2  The Two-Day Pattern Inside Each Arc", 2)

    add_para(doc, "Every arc follows a consistent two-day pattern:")

    add_bullet(doc, "Day N  (arc opening) — \"First Encounter\": A new rule is introduced "
               "implicitly. The task does not mention the rule. The agent fails and receives "
               "corrective feedback that reveals the rule for the first time.")
    add_bullet(doc, "Day N+1 (arc closing) — \"Comprehensive Test\": The same domain or a "
               "different domain is used, but now the agent is expected to follow all rules "
               "introduced so far without any hints.")

    add_para(doc, "Example — Arc A (Day 01):", bold=True)
    add_bullet(doc, "r1: Ask the agent to organize meeting notes into JSON. "
               "No mention of date format.", level=1)
    add_bullet(doc, "r1 incorrect feedback: 'All time fields must use full ISO 8601 with "
               "+08:00 offset — e.g. 2026-03-16T09:30:00+08:00'", level=1)
    add_bullet(doc, "r2: Multiple-choice question testing conceptual understanding of P1.", level=1)
    add_bullet(doc, "r3: Another hands-on task — still testing P1 but in a different context.", level=1)
    add_para(doc, "Day 02 (Arc A): Switches to a completely different work domain (data processing) "
             "and tests P1 again — but this time the feedback does not explain P1. "
             "The agent must remember it on its own.")

    # ── Section 4 ────────────────────────────────────────────────────────────
    set_heading(doc, "4. Two Question Types: Complementary by Design", 1)

    set_heading(doc, "4.1  file_check — Behavioral Verification", 2)

    add_para(doc,
        "The agent performs real file operations in an isolated workspace. "
        "An eval command is executed as a subprocess to verify the output programmatically.")

    add_code_block(doc,
        '{\n'
        '  "type": "file_check",\n'
        '  "question": "Organize standup_raw.txt into standup.json with fields: '
        'meeting_time, attendees, action_items",\n'
        '  "eval": {\n'
        '    "command": "python scripts/check_iso8601.py day01/standup.json '
        'meeting_time action_items[].due_date",\n'
        '    "expect_exit": 0\n'
        '  }\n'
        '}')

    add_bullet(doc, "Scoring: binary pass/fail (1.0 or 0.0) — no model judge, zero noise.")
    add_bullet(doc, "Supports chaining multiple validators with && for multi-rule verification.")
    add_bullet(doc, "Supports extra flags: expect_stdout, expect_stdout_regex, timeout.")

    set_heading(doc, "4.2  multi_choice — Conceptual Probing", 2)

    add_para(doc,
        "The agent selects from multiple options representing typical correct and incorrect "
        "interpretations of a rule. Answer extraction uses \\bbox{X,Y} or \\boxed{X,Y} regex.")

    add_code_block(doc,
        '{\n'
        '  "type": "multi_choice",\n'
        '  "question": "Which time strings conform to the standard? '
        'Select all that apply.\\n'
        'A. 2026-03-16T09:30:00+08:00\\n'
        'B. 2026-03-16 09:30:00\\n'
        'E. 2026-03-16T09:30:00.000+08:00\\n'
        '... Answer using \\\\bbox{X,Y}",\n'
        '  "eval": { "answer": ["A", "E"] }\n'
        '}')

    add_para(doc, "Scoring formula (IoU F1, not exact match):", bold=True)
    add_code_block(doc,
        "score = 1 - (false_positives + false_negatives) / total_options\n"
        "\n"
        "# Metrics also computed: exact_match, IoU, precision, recall, F1")

    add_para(doc,
        "Each distractor option is hand-crafted to represent one specific "
        "misunderstanding — e.g. using Z (UTC) instead of +08:00, or using slash "
        "date separators, or missing the T separator.")

    set_heading(doc, "4.3  Why Two Types?", 2)
    add_para(doc,
        "The two types are complementary, not redundant. An agent may produce correct "
        "output without being able to articulate the rule (behavioral without conceptual), "
        "or may correctly identify valid formats without producing them under a realistic "
        "task prompt (conceptual without behavioral). Both dimensions together provide a "
        "complete picture of preference acquisition.")

    # ── Section 5 ────────────────────────────────────────────────────────────
    set_heading(doc, "5. Workspace Design: Isolation + Memory Continuity", 1)

    set_heading(doc, "5.1  Per-Day Directory Isolation", 2)

    add_para(doc,
        "Each day has its own workspace subdirectory containing that day's raw materials "
        "(standup notes, JSON stubs, code files, etc.). When a test runs, the agent's "
        "file access is restricted to the matching dayXX/ folder — it cannot see other "
        "days' files.")

    add_code_block(doc,
        "workspaces/shared/\n"
        "├── IDENTITY.md     ← Agent role definition (constant across all days)\n"
        "├── USER.md         ← User profile (constant across all days)\n"
        "├── day01/\n"
        "│   ├── README.md           ← Narrative context for today\n"
        "│   ├── standup_raw.txt     ← Raw input material\n"
        "│   └── sprint_tasks.json   ← Structured data to process\n"
        "├── day05/\n"
        "│   ├── sprint_review_notes_raw.txt\n"
        "│   └── ...\n"
        "└── day12/   ← hardest: 6 rounds, all 5 rules, no hints")

    set_heading(doc, "5.2  Session Architecture: One Session Per Day", 2)

    add_para(doc,
        "Each day has a unique session ID (e.g. day03_e6a9e1bf-...). "
        "The openclaw_state directory ships with pre-written session JSONL files for "
        "every day. Each file contains only a brief two-message opening: "
        "a user message setting the date and context, and a short agent acknowledgment.")

    add_para(doc,
        "Cross-day memory is not embedded inside session files. Instead, three "
        "independent mechanisms handle it:")

    add_bullet(doc, "Session Visibility (visibility: 'agent'): The agent can read all "
               "past session transcripts belonging to the same agent identity. "
               "When processing day N's questions, the agent has direct access to the "
               "real interaction logs of days 1 through N-1.")
    add_bullet(doc, "Memory Injection: After each day completes, POST /v1/memory/ingest "
               "extracts semantically structured memories from the session and injects "
               "them into subsequent days' context. This is a compressed, curated version "
               "of history — not raw transcripts.")
    add_bullet(doc, "RL Weight Updates: metaclaw train-step runs GRPO updates using "
               "the day's trajectories, burning the learned conventions into model "
               "weights. The next day runs with an updated model that 'knows' the rules "
               "without any explicit memory retrieval.")

    add_para(doc,
        "By toggling each mechanism independently, the benchmark can isolate and "
        "measure each mechanism's contribution to behavioral adaptation:")

    add_code_block(doc,
        "baseline:       session visibility off → agent remembers nothing\n"
        "memory only:    curated memory injected → explicit recall\n"
        "RL only:        weights updated → implicit retention\n"
        "RL + memory:    both combined → full MetaClaw mode")

    # ── Section 6 ────────────────────────────────────────────────────────────
    set_heading(doc, "6. Feedback Injection Mechanism", 1)

    add_para(doc,
        "After each round, the system immediately scores the agent's output (inline "
        "scoring) and prepends the result to the next round's message:")

    add_code_block(doc,
        '# prompts.py\n'
        'def with_feedback(feedback_text, question_text):\n'
        '    return f"[Previous Feedback] {feedback_text}\\n\\n{question_text}"\n'
        '\n'
        '# infer_cmd.py — _run_group()\n'
        'if prev_inline_score is not None:\n'
        '    feedback_text = _build_feedback_text(prev_round_record, prev_inline_score)\n'
        '    query = with_feedback(feedback_text, next_question)\n'
        'else:\n'
        '    query = next_question')

    add_para(doc, "Three categories of feedback content:", bold=True)

    add_bullet(doc, "Format error: \"Your response did not include a \\bbox{X} answer...\" "
               "— triggered when the agent fails to use the required answer format.")
    add_bullet(doc, "Option-level explanation: \"You missed option E: milliseconds are optional; "
               "the +08:00 timezone offset is present.\" / \"You incorrectly selected option B: "
               "missing the T separator...\" — one line per incorrect selection.")
    add_bullet(doc, "File check feedback: the 'correct' or 'incorrect' branch from the round "
               "record. The incorrect branch explicitly names which rule was violated and gives "
               "the correct form.")

    add_para(doc,
        "After the final round of each day, an additional standalone feedback message "
        "is sent to the agent with no attached question. This allows the agent's "
        "memory/RL system to internalize the day's lessons as a standalone signal, "
        "not just as a prefix to the next task.")

    # ── Section 7 ────────────────────────────────────────────────────────────
    set_heading(doc, "7. Eval Scripts: Engineering-Grade Verification", 1)

    add_para(doc,
        "Five Python validator scripts correspond to the five preferences. "
        "They are designed for precision, composability, and zero false positives.")

    add_table(doc,
        headers=["Script", "Checks", "Key Feature"],
        rows=[
            ("check_iso8601.py",
             "P1 — datetime format",
             "Regex: ^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(\\.\\d+)?\\+08:00$\n"
             "Supports dot-notation paths (meta.created_at) and array paths (items[].due_date)"),
            ("check_filename.py",
             "P2 — file naming",
             "Regex: ^\\d{8}_[a-z][a-z0-9_]*\\.[a-z0-9]+$\n"
             "Also supports --dir / --ext / --min-count for directory-level checks"),
            ("check_metadata.py",
             "P3 — metadata fields",
             "Dispatches by file type: JSON → meta object, MD → YAML frontmatter, "
             "PY → module docstring Meta section, CSV → first-line # meta: comment.\n"
             "Validates created_at ISO format + status enum {pending, in_progress, done}"),
            ("check_backup.py",
             "P4 — modify vs. create",
             "Verifies that specific existing files were modified in-place "
             "rather than recreated"),
            ("check_done_log.py",
             "P5 — completion log",
             "Line regex with ISO timestamp, task_id, summary (≤80 chars).\n"
             "Supports --min-entries and --task-prefix for progressive validation"),
        ],
        col_widths=[1.5, 1.4, 3.8],
    )
    doc.add_paragraph()

    add_para(doc, "Multi-rule eval commands chain validators with &&:", bold=True)
    add_code_block(doc,
        "# day12/r1 eval command (4 validators in sequence)\n"
        "python -c \"import glob,sys; files=sorted(glob.glob('day12/20260424_*.json')); "
        "sys.exit(0 if files else 1)\"\n"
        "&& python scripts/check_iso8601.py <file> meta.created_at dashboard_generated_at\n"
        "&& python scripts/check_metadata.py <file>\n"
        "&& python scripts/check_done_log.py done.log --min-entries 1 "
        "--task-prefix sprint10_dashboard")

    # ── Section 8 ────────────────────────────────────────────────────────────
    set_heading(doc, "8. Core Design Patterns", 1)

    set_heading(doc, "Pattern 1: Programmatically Verifiable Behavioral Rules", 2)
    add_para(doc,
        "The benchmark's power comes not from difficulty but from measurement precision. "
        "Every rule is expressible as a deterministic predicate executable on the "
        "agent's file output. This eliminates all subjectivity from the reward signal — "
        "a critical property for RL training data generation.")

    set_heading(doc, "Pattern 2: First-Encounter → Failure → Feedback → Consolidation", 2)
    add_para(doc,
        "Every rule follows a strict introduction lifecycle across the arc structure:")
    add_code_block(doc,
        "First encounter → agent fails → feedback reveals rule\n"
        "    → same-arc day 2 → test same rule (no hint)\n"
        "    → every subsequent comprehensive day → cumulative test of all rules so far\n"
        "    → Arc F final exam → zero hints, all rules, hardest tasks")
    add_para(doc,
        "This is a Spaced Repetition testing structure. Rules are not tested once; "
        "they recur across domains and time intervals, measuring genuine retention "
        "rather than session-level recall.")

    set_heading(doc, "Pattern 3: Behavior Layer + Concept Layer", 2)
    add_para(doc,
        "Each rule is tested at two levels: file_check tests whether the agent "
        "can produce compliant output (behavioral), while multi_choice tests whether "
        "the agent can distinguish correct from incorrect forms (conceptual). "
        "Distractors in multi_choice questions are precisely crafted — each wrong option "
        "represents one specific, commonly occurring misunderstanding of the rule.")

    set_heading(doc, "Pattern 4: Narrative-Driven Task Framing", 2)
    add_para(doc,
        "Tasks are framed as realistic workplace activities (\"organize today's standup "
        "notes\", \"write the sprint review\", \"prepare deployment notes\"), not as "
        "format-compliance exercises. The rules are embedded requirements that the agent "
        "must recognize and apply on its own — not instructions. This is what makes "
        "the benchmark measure genuine preference internalization rather than "
        "instruction-following capability.")

    set_heading(doc, "Pattern 5: Zero-Hint Final Exam", 2)
    add_para(doc,
        "Arc F (days 11–12 in the small version, days 29–30 in the full version) removes "
        "all hint-bearing feedback. The incorrect branch of feedback only identifies "
        "what went wrong, not how to fix it. This validates true internalization: "
        "the agent must know the rules from memory or weights, not from in-context clues.")

    # ── Section 9 ────────────────────────────────────────────────────────────
    set_heading(doc, "9. Implications for Training Data Generation", 1)

    add_para(doc,
        "The benchmark's design philosophy translates directly into a set of principles "
        "for constructing high-quality RL training data:")

    add_para(doc, "1. Design rules as executable validators, not LLM judgments.", bold=True)
    add_para(doc,
        "Build a reward function that is a deterministic program, not a model call. "
        "This gives clean, reproducible binary or partial-credit reward signals. "
        "The five eval scripts here are the canonical example.",
        indent=0.3)

    add_para(doc, "2. Generate trajectories across the full failure-correction-success arc.", bold=True)
    add_para(doc,
        "For each rule, you need at least two trajectory types: (a) first-encounter "
        "failure with corrective feedback signal, and (b) later-task success without "
        "hints. GRPO needs reward variance within a group — groups where the agent "
        "sometimes complies and sometimes does not produce the best learning signal.",
        indent=0.3)

    add_para(doc, "3. Wrap rules inside realistic tasks, not explicit instructions.", bold=True)
    add_para(doc,
        "A task prompt that says 'use ISO 8601' trains instruction-following, not "
        "preference internalization. The prompt should describe a realistic activity; "
        "the rule should be an implicit requirement the agent infers and applies. "
        "This creates trajectories with better generalization value.",
        indent=0.3)

    add_para(doc, "4. Apply Spaced Repetition across domains.", bold=True)
    add_para(doc,
        "The same rule appearing in project management (day01), data processing "
        "(day02), code engineering (day03), and deployment notes (day12) forces the "
        "model to learn a domain-agnostic representation of the rule. "
        "Single-domain training data produces brittle behavior.",
        indent=0.3)

    add_para(doc, "5. Chain validators for multi-rule tasks.", bold=True)
    add_para(doc,
        "Late-stage training tasks should require satisfying multiple rules simultaneously. "
        "The && chaining pattern in eval commands is the right model: pass all validators "
        "to earn reward = 1.0, fail any one to earn 0.0 or a partial score. "
        "This creates natural curriculum progression.",
        indent=0.3)

    # ── Footer ───────────────────────────────────────────────────────────────
    doc.add_paragraph()
    doc.add_paragraph()
    hr = doc.add_paragraph("─" * 80)
    hr.alignment = WD_ALIGN_PARAGRAPH.CENTER
    note = doc.add_paragraph(
        "Source: Analysis of MetaClaw repository at github.com/TinaZhang66/MetaClaw. "
        "All code references are from benchmark/src/ and benchmark/data/metaclaw-bench-small/."
    )
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    note.runs[0].font.size = Pt(9)
    note.runs[0].font.color.rgb = RGBColor(0x80, 0x80, 0x80)

    doc.save(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_report()
