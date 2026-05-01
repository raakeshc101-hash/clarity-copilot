from __future__ import annotations

import json
import os
import sqlite3
import uuid
import html as html_lib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from flask import Flask, render_template_string, request, session
from openai import OpenAI

load_dotenv()

app = Flask(__name__)
app.secret_key = "clarity-copilot-local-dev-secret-key"
app.permanent_session_lifetime = timedelta(days=30)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
DB_NAME = "clarity.db"


def init_db() -> None:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS submissions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_uuid TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        role TEXT,
        role_cluster TEXT,
        week TEXT,
        confidence TEXT,
        interaction TEXT,
        situation TEXT,
        primary_issue TEXT,
        secondary_issue TEXT,
        sub_issue TEXT,
        problem_type TEXT,
        behavioral_barrier TEXT,
        adjustment_target TEXT,
        retention_risk TEXT,
        best_first_stop TEXT,
        final_decision_owner TEXT,
        script TEXT
    )
    """)

    conn.commit()
    conn.close()


def get_user_uuid() -> str:
    session.permanent = True
    if "user_uuid" not in session:
        session["user_uuid"] = str(uuid.uuid4())
    return session["user_uuid"]


def save_submission(
    user_uuid: str,
    role: str,
    role_cluster: str,
    week: str,
    confidence: str,
    interaction: str,
    situation: str,
    result: Dict[str, Any],
) -> None:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO submissions (
        user_uuid, timestamp, role, role_cluster, week, confidence, interaction, situation,
        primary_issue, secondary_issue, sub_issue, problem_type, behavioral_barrier,
        adjustment_target, retention_risk, best_first_stop, final_decision_owner, script
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_uuid,
        datetime.now().isoformat(timespec="seconds"),
        role,
        role_cluster,
        week,
        confidence,
        interaction,
        situation,
        result.get("primary_issue", ""),
        result.get("secondary_issue", ""),
        result.get("sub_issue", ""),
        result.get("problem_type", ""),
        result.get("behavioral_barrier", ""),
        result.get("adjustment_target", ""),
        result.get("retention_risk", ""),
        result.get("best_first_stop", ""),
        result.get("final_decision_owner", ""),
        result.get("script", ""),
    ))

    conn.commit()
    conn.close()


def fetch_user_history(user_uuid: str) -> List[Tuple[str, str, str, str, str]]:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT timestamp, role, primary_issue, retention_risk, script
        FROM submissions
        WHERE user_uuid = ?
        ORDER BY timestamp DESC
    """, (user_uuid,))

    rows = cursor.fetchall()
    conn.close()
    return rows


def fetch_admin_aggregates() -> Dict[str, Any]:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM submissions")
    total = cursor.fetchone()[0]

    cursor.execute("""
        SELECT primary_issue, COUNT(*) 
        FROM submissions 
        GROUP BY primary_issue 
        ORDER BY COUNT(*) DESC
    """)
    primary_issues = cursor.fetchall()

    cursor.execute("""
        SELECT behavioral_barrier, COUNT(*) 
        FROM submissions 
        GROUP BY behavioral_barrier 
        ORDER BY COUNT(*) DESC
    """)
    barriers = cursor.fetchall()

    cursor.execute("""
        SELECT retention_risk, COUNT(*) 
        FROM submissions 
        GROUP BY retention_risk 
        ORDER BY CASE retention_risk
            WHEN 'Low' THEN 1
            WHEN 'Medium' THEN 2
            WHEN 'High' THEN 3
            ELSE 4
        END
    """)
    risks = cursor.fetchall()

    cursor.execute("""
        SELECT role_cluster, COUNT(*) 
        FROM submissions 
        GROUP BY role_cluster 
        ORDER BY COUNT(*) DESC
    """)
    role_clusters = cursor.fetchall()

    cursor.execute("""
        SELECT week, COUNT(*) 
        FROM submissions 
        GROUP BY week 
        ORDER BY COUNT(*) DESC
    """)
    weeks = cursor.fetchall()

    cursor.execute("""
        SELECT role_cluster,
        AVG(CASE retention_risk
            WHEN 'Low' THEN 1
            WHEN 'Medium' THEN 2
            WHEN 'High' THEN 3
            ELSE NULL
        END)
        FROM submissions
        GROUP BY role_cluster
        ORDER BY role_cluster
    """)
    avg_risk = cursor.fetchall()

    conn.close()

    return {
        "total": total,
        "primary_issues": primary_issues,
        "barriers": barriers,
        "risks": risks,
        "role_clusters": role_clusters,
        "weeks": weeks,
        "avg_risk": avg_risk,
    }


def escape(value: Any) -> str:
    return html_lib.escape(str(value if value is not None else ""))


def map_role_to_cluster(role: str) -> str:
    role = (role or "").lower().strip()

    if any(x in role for x in ["analyst", "data", "finance", "strategy", "reporting", "insights"]):
        return "analytical"
    if any(x in role for x in ["hr", "recruit", "people", "l&d", "learning", "talent", "od", "development"]):
        return "people"
    if any(x in role for x in ["operations", "ops", "logistics", "coordinator", "program", "project"]):
        return "operations"
    if any(x in role for x in ["engineer", "developer", "it", "technical", "software", "systems"]):
        return "technical"
    if any(x in role for x in ["sales", "customer", "account", "success", "support", "service"]):
        return "customer"
    if any(x in role for x in ["manager", "lead", "supervisor"]):
        return "leadership"
    return "general"


SYSTEM_PROMPT = """
You are Clarity Copilot, a human-in-the-loop onboarding decision-support system for early-career employees.

Your job is to convert unclear onboarding situations into:
- correct diagnosis
- correct routing
- useful questions
- low-friction communication
- clear next steps

You are not a general chatbot.
You are not a therapist.
You are not a performance evaluator.
You are not a policy authority.

Use these clarity buckets:
- Role Clarity
- Expectation Clarity
- Task / Priority Clarity
- Process Clarity
- Social Clarity
- Norm Clarity
- Confidence / Hesitation

You must identify:
- primary_issue
- secondary_issue
- sub_issue
- problem_type from one of:
  - Information
  - Decision
  - Execution
- adjustment_target from one of:
  - Role Clarity
  - Task Mastery
  - Social Integration
  - Confidence to Seek Help
- behavioral_barrier from one of:
  - Hesitation
  - Avoidance
  - Overthinking
  - Passive Waiting
  - Fear of judgment
  - Fear of interrupting
  - None
- retention_risk from one of:
  - Low
  - Medium
  - High

Decision logic:
- Primary issue = what most directly blocks action right now.
- Secondary issue = what makes resolution harder.
- Do NOT default to confidence.
- If the user cannot proceed because the work, output, ownership, expectations, priority, or level of detail is unclear, primary issue should usually be Task / Priority Clarity, Expectation Clarity, or Role Clarity.
- If the user knows what to do but is avoiding asking, speaking, escalating, or acting, Confidence / Hesitation may be primary.
- If the user knows the task but does not know who to go to, Social Clarity may be primary.
- If both confusion and hesitation exist, confusion is usually primary and hesitation is secondary unless avoidance is the main blocker.
- Before finalizing, check: "What most directly prevents this person from moving forward in the next hour?"

Problem type rules:
- Information = user mainly needs clarification, context, examples, norms, process understanding, or interpretation.
- Decision = user needs priority resolution, expectation alignment, role boundary resolution, ownership clarification, tradeoff judgment, or conflict resolution.
- Execution = user understands the direction but is blocked on doing, sequencing, skill confidence, carrying it out, or getting started.

Routing logic:
You must determine:
- best_first_stop = who the user should approach first for the fastest safe clarity
- final_decision_owner = who has actual authority to settle the issue

Do NOT map buckets rigidly to people.
Instead decide based on:
- ambiguity type
- problem_type
- authority needed
- speed of resolution
- psychological safety
- whether the issue is interpretive or decision-critical

Routing rules:
- Use peer / buddy / teammate first when the issue is mainly interpretive, procedural, local-norm, or "how things usually work here."
- Use documentation first when the issue is likely already codified and low-risk.
- Use manager / owner first when the issue requires role expectation, prioritization, ownership, evaluation, tradeoff, or final decision authority.
- A peer may be the best first stop even when the manager is the final decision owner.
- If the user signals low safety with manager (e.g., "I don't want to bother them", "manager is always busy", "I don't want to ask basic questions"), do not force manager-first unless the issue is clearly decision-critical.
- Always distinguish between:
  - fastest safe clarity path
  - final authority

Role adaptation rules:
- Adapt wording, examples, and suggested actions based on the user's role cluster.
- Do not change diagnosis logic based only on title.
- Use role context to make the response feel relevant, not generic.

Role cluster guidance:
- analytical = metrics, analysis, dashboards, insights, business questions
- people = stakeholders, communication, learning, employee support, alignment
- operations = timelines, coordination, execution, handoffs
- technical = systems, dependencies, implementation, debugging
- customer = client clarity, responsiveness, communication, service outcomes
- leadership = role scope, delegation, decision rights, team outcomes, performance expectations
- general = broad workplace language

Behavioral barrier rules:
- Hesitation = mild holding back
- Avoidance = deliberately delaying or staying silent despite needing clarity
- Overthinking = mentally looping without acting
- Passive Waiting = waiting for clarity without initiating
- Fear of judgment = concern about looking incompetent
- Fear of interrupting = concern about bothering or burdening others
- None = no clear barrier signal

Retention risk guidance:
- Low = ambiguity looks contained and action is still realistic.
- Medium = ambiguity is affecting confidence, help-seeking, routing, or quality of action.
- High = ambiguity plus hesitation plus weak support path may lead to silent struggle, slower ramp-up, or disengagement.

What to Do Next rules:
- Return exactly 3 concrete steps.
- Each step must be actionable within the next 15 to 30 minutes or the next obvious work interaction.
- Avoid generic advice.
- The steps must align with best_first_stop and final_decision_owner.

Question design rules:
- Generate questions by target, not just by topic.
- All questions must be in first-person voice because the user is reading and using them.
- Exactly 3 sections, each with exactly 3 questions:
  - questions_for_best_first_stop
  - questions_for_final_decision_owner
  - questions_for_myself_before_i_ask
- No repetition across sections.
- The "myself" questions should help the user prepare and reduce vague asking.
- Questions for best_first_stop should fit that target.
- Questions for final_decision_owner should fit that target.
- If best_first_stop and final_decision_owner are the same person, still keep the section names but make them distinguishable:
  - best_first_stop = quickest clarifiers
  - final_decision_owner = decision-setting questions
- If the situation includes unclear performance, success, targets, ownership, or evaluation, at least 2 questions across all sections must explicitly address success criteria, metrics, targets, or how performance is judged.

How to Phrase It rules:
- Return one ready-to-use message only.
- It must be for the final_decision_owner.
- Maximum 2 to 3 sentences.
- Copy-paste ready.
- No coaching.
- No explanations.
- No placeholders.
- Sound natural and professional.
- Reduce burden on the other person.

Tone refinement:
- The message must sound like a real early-career employee, not a formal email.
- Use simple, conversational workplace language.
- Avoid phrases like:
  - "I wanted to clarify"
  - "Could you please confirm"
  - "align my work accordingly"
- Prefer:
  - "quick check"
  - "just wanted to check"
  - "want to make sure I’m aligned"
- Keep sentences short and direct.
- One core ask per sentence.
- Avoid over-explaining context.
- The message should feel safe and low-pressure to send.

What’s Happening rules:
- Reflect the user's exact tension in plain language.
- Sound human, not robotic or clinical.

Output rules:
Return valid JSON only with these fields:
- primary_issue
- secondary_issue
- sub_issue
- problem_type
- adjustment_target
- behavioral_barrier
- retention_risk
- best_first_stop
- final_decision_owner
- why_this_routing
- whats_happening
- what_to_do_next
- what_success_looks_like
- why_this_works
- why_diagnosis
- missing
- retention_reason
- questions
- script
"""


SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "primary_issue": {"type": "string"},
        "secondary_issue": {"type": "string"},
        "sub_issue": {"type": "string"},
        "problem_type": {
            "type": "string",
            "enum": ["Information", "Decision", "Execution"],
        },
        "adjustment_target": {"type": "string"},
        "behavioral_barrier": {"type": "string"},
        "retention_risk": {"type": "string", "enum": ["Low", "Medium", "High"]},
        "best_first_stop": {"type": "string"},
        "final_decision_owner": {"type": "string"},
        "why_this_routing": {"type": "string"},
        "whats_happening": {"type": "string"},
        "what_to_do_next": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
        },
        "what_success_looks_like": {"type": "string"},
        "why_this_works": {"type": "string"},
        "why_diagnosis": {"type": "string"},
        "missing": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 4,
        },
        "retention_reason": {"type": "string"},
        "questions": {
            "type": "object",
            "properties": {
                "questions_for_best_first_stop": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "questions_for_final_decision_owner": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "questions_for_myself_before_i_ask": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 3,
                },
            },
            "required": [
                "questions_for_best_first_stop",
                "questions_for_final_decision_owner",
                "questions_for_myself_before_i_ask",
            ],
            "additionalProperties": False,
        },
        "script": {"type": "string"},
    },
    "required": [
        "primary_issue",
        "secondary_issue",
        "sub_issue",
        "problem_type",
        "adjustment_target",
        "behavioral_barrier",
        "retention_risk",
        "best_first_stop",
        "final_decision_owner",
        "why_this_routing",
        "whats_happening",
        "what_to_do_next",
        "what_success_looks_like",
        "why_this_works",
        "why_diagnosis",
        "missing",
        "retention_reason",
        "questions",
        "script",
    ],
    "additionalProperties": False,
}


def normalize_result(data: Dict[str, Any]) -> Dict[str, Any]:
    def safe_str(val: Any, default: str = "") -> str:
        return val if isinstance(val, str) else default

    def safe_list(val: Any) -> list:
        return val if isinstance(val, list) else []

    result = {
        "primary_issue": safe_str(data.get("primary_issue")),
        "secondary_issue": safe_str(data.get("secondary_issue")),
        "sub_issue": safe_str(data.get("sub_issue")),
        "problem_type": safe_str(data.get("problem_type"), "Information"),
        "adjustment_target": safe_str(data.get("adjustment_target")),
        "behavioral_barrier": safe_str(data.get("behavioral_barrier"), "None"),
        "retention_risk": safe_str(data.get("retention_risk"), "Medium"),
        "best_first_stop": safe_str(data.get("best_first_stop")),
        "final_decision_owner": safe_str(data.get("final_decision_owner")),
        "why_this_routing": safe_str(data.get("why_this_routing")),
        "whats_happening": safe_str(data.get("whats_happening")),
        "what_to_do_next": safe_list(data.get("what_to_do_next")),
        "what_success_looks_like": safe_str(data.get("what_success_looks_like")),
        "why_this_works": safe_str(data.get("why_this_works")),
        "why_diagnosis": safe_str(data.get("why_diagnosis")),
        "missing": safe_list(data.get("missing")),
        "retention_reason": safe_str(data.get("retention_reason")),
        "questions": {
            "questions_for_best_first_stop": safe_list(data.get("questions", {}).get("questions_for_best_first_stop")),
            "questions_for_final_decision_owner": safe_list(data.get("questions", {}).get("questions_for_final_decision_owner")),
            "questions_for_myself_before_i_ask": safe_list(data.get("questions", {}).get("questions_for_myself_before_i_ask")),
        },
        "script": safe_str(data.get("script")),
    }

    return result


def call_model(role: str, week: str, confidence: str, interaction: str, situation: str) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")

    client = OpenAI(api_key=api_key)
    cluster = map_role_to_cluster(role)

    prompt = f"""
Role: {role}
Role cluster: {cluster}
Week in role: {week}
Confidence level: {confidence}
Upcoming interaction: {interaction}

Situation:
{situation}
"""

    response = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "clarity",
                "schema": SCHEMA,
                "strict": True,
            }
        },
    )

    parsed = json.loads(response.output_text)
    return normalize_result(parsed)


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Clarity Copilot</title>
  <style>
    :root{
      --bg:#f4f7fb;
      --card:#ffffff;
      --line:#dbe4f0;
      --text:#1f2937;
      --muted:#667085;
      --blue:#174f8a;
      --blue-soft:#eaf2fb;
      --green:#0f8b4c;
      --green-soft:#e9f7ef;
      --amber:#a15c00;
      --amber-soft:#fff6e5;
      --red:#b42318;
      --red-soft:#fdecec;
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      font-family:Arial,sans-serif;
      background:var(--bg);
      color:var(--text);
    }
    .wrap{
      max-width:1240px;
      margin:28px auto;
      padding:0 16px 40px;
    }
    .hero,.card{
      background:var(--card);
      border:1px solid var(--line);
      border-radius:18px;
      box-shadow:0 8px 24px rgba(16,24,40,.05);
    }
    .hero{
      padding:28px;
      margin-bottom:20px;
    }
    h1{
      margin:0 0 10px;
      color:#123f6d;
      font-size:32px;
    }
    .subtitle{
      margin:0;
      color:var(--muted);
      line-height:1.6;
      max-width:940px;
    }
    .grid{
      display:grid;
      grid-template-columns:0.95fr 1.05fr;
      gap:20px;
      align-items:start;
    }
    .card{padding:22px}
    label{
      display:block;
      margin-bottom:8px;
      color:#143d6b;
      font-weight:700;
    }
    input,select,textarea{
      width:100%;
      padding:12px 14px;
      border:1px solid #ced8e6;
      border-radius:12px;
      background:#fff;
      color:var(--text);
      font-size:15px;
      margin-bottom:16px;
    }
    textarea{
      min-height:210px;
      resize:vertical;
      line-height:1.55;
    }
    .btn{
      width:100%;
      border:none;
      border-radius:12px;
      background:var(--blue);
      color:#fff;
      font-weight:700;
      font-size:16px;
      padding:14px 16px;
      cursor:pointer;
    }
    .btn:hover{filter:brightness(.98)}
    .note{
      margin-top:12px;
      padding:12px 14px;
      border:1px dashed #c9d8eb;
      border-radius:12px;
      background:#f8fbff;
      color:var(--muted);
      font-size:13px;
      line-height:1.55;
    }
    .error{
      background:var(--red-soft);
      color:var(--red);
      border:1px solid #f0c7c7;
      border-radius:12px;
      padding:12px 14px;
      margin-bottom:12px;
      line-height:1.55;
    }
    .badge-row{
      display:flex;
      flex-wrap:wrap;
      gap:10px;
      margin-bottom:16px;
    }
    .badge{
      display:inline-flex;
      align-items:center;
      padding:8px 12px;
      border-radius:999px;
      background:var(--blue-soft);
      color:var(--blue);
      font-size:14px;
      font-weight:700;
    }
    .risk{
      display:inline-flex;
      align-items:center;
      padding:8px 12px;
      border-radius:999px;
      font-size:14px;
      font-weight:700;
    }
    .risk.low{background:var(--green-soft);color:var(--green)}
    .risk.medium{background:var(--amber-soft);color:var(--amber)}
    .risk.high{background:var(--red-soft);color:var(--red)}
    .section{
      border-top:1px solid var(--line);
      padding-top:16px;
      margin-top:16px;
    }
    .section h3{
      margin:0 0 8px;
      color:#123f6d;
      font-size:18px;
    }
    .section p{
      margin:0;
      line-height:1.65;
    }
    ul,ol{
      margin:8px 0 0 20px;
      padding:0;
      line-height:1.7;
    }
    .two-col{
      display:grid;
      grid-template-columns:1fr 1fr;
      gap:14px;
      margin-top:8px;
    }
    .mini-card{
      border:1px solid var(--line);
      border-radius:12px;
      background:#fbfdff;
      padding:12px 14px;
    }
    .mini-card h4{
      margin:0 0 6px;
      color:var(--blue);
      font-size:15px;
    }
    .qgrid{
      display:grid;
      gap:12px;
      margin-top:6px;
    }
    .qbox{
      border:1px solid var(--line);
      border-radius:12px;
      padding:12px 14px;
      background:#fbfdff;
    }
    .qbox strong{
      display:block;
      margin-bottom:6px;
      color:var(--blue);
    }
    .script{
      white-space:pre-line;
      background:#f8fafc;
      border:1px solid var(--line);
      border-radius:12px;
      padding:14px;
      line-height:1.7;
    }
    .empty{
      color:var(--muted);
      line-height:1.7;
    }
    @media (max-width:980px){
      .grid{grid-template-columns:1fr}
      .two-col{grid-template-columns:1fr}
      h1{font-size:28px}
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>Clarity Copilot</h1>
      <p class="subtitle">
        A decision-support prototype for early-career onboarding ambiguity. It diagnoses the issue, identifies the safest-fastest routing path, and converts confusion into clearer action.
      </p>
    </div>

    <div class="grid">
      <div class="card">
        <form method="POST" action="/analyze">
          <label for="role">Role</label>
          <input id="role" name="role" value="{{ form.role }}" placeholder="Entry-Level Business Analyst">

          <label for="week">Week in Role</label>
          <select id="week" name="week">
            {% for w in ["Week 1","Week 2","Week 3","Week 4","Week 5+","Not sure"] %}
              <option value="{{ w }}" {% if form.week == w %}selected{% endif %}>{{ w }}</option>
            {% endfor %}
          </select>

          <label for="confidence">Current Confidence Level</label>
          <select id="confidence" name="confidence">
            {% for c in ["Low","Medium","High","Not sure"] %}
              <option value="{{ c }}" {% if form.confidence == c %}selected{% endif %}>{{ c }}</option>
            {% endfor %}
          </select>

          <label for="interaction">Upcoming Interaction</label>
          <select id="interaction" name="interaction">
            {% for i in ["None","1:1 with manager","Team meeting","Stakeholder meeting","Slack update","Review discussion","Not sure"] %}
              <option value="{{ i }}" {% if form.interaction == i %}selected{% endif %}>{{ i }}</option>
            {% endfor %}
          </select>

          <label for="situation">Describe the Situation</label>
          <textarea id="situation" name="situation" placeholder="Example: I was asked to put together insights from onboarding feedback, but I don’t know what exactly they expect or what format to use.">{{ form.situation }}</textarea>

          <button class="btn" type="submit">Analyze Situation</button>

          <div class="note">
            This prototype decides not just what the issue is, but who to approach first, who owns the final answer, what to ask, and how to ask it.
          </div>
        </form>
      </div>

      <div class="card">
        {% if error %}
          <div class="error">{{ error }}</div>
        {% endif %}

        {% if result %}
          <div class="badge-row">
            <span class="badge">Primary: {{ result.primary_issue }}</span>
            <span class="badge">Secondary: {{ result.secondary_issue }}</span>
            <span class="badge">Sub-Issue: {{ result.sub_issue }}</span>
            <span class="badge">Problem Type: {{ result.problem_type }}</span>
            <span class="badge">Adjustment Target: {{ result.adjustment_target }}</span>
            <span class="badge">Barrier: {{ result.behavioral_barrier }}</span>
            <span class="risk {{ result.retention_risk|lower }}">{{ result.retention_risk }} retention risk</span>
          </div>

          <div class="section">
            <h3>What’s Happening</h3>
            <p>{{ result.whats_happening }}</p>
          </div>

          <div class="section">
            <h3>Routing</h3>
            <div class="two-col">
              <div class="mini-card">
                <h4>Best First Stop</h4>
                <p>{{ result.best_first_stop }}</p>
              </div>
              <div class="mini-card">
                <h4>Final Decision Owner</h4>
                <p>{{ result.final_decision_owner }}</p>
              </div>
            </div>
            <p style="margin-top:10px;"><strong>Why this routing:</strong> {{ result.why_this_routing }}</p>
          </div>

          <div class="section">
            <h3>What to Do Next</h3>
            <ol>
              {% for step in result.what_to_do_next %}
                <li>{{ step }}</li>
              {% endfor %}
            </ol>
          </div>

          <div class="section">
            <h3>What Success Looks Like</h3>
            <p>{{ result.what_success_looks_like }}</p>
          </div>

          <div class="section">
            <h3>Why This Works</h3>
            <p>{{ result.why_this_works }}</p>
          </div>

          <div class="section">
            <h3>Why This Diagnosis Fits</h3>
            <p>{{ result.why_diagnosis }}</p>
          </div>

          <div class="section">
            <h3>What You Might Be Missing</h3>
            <ul>
              {% for item in result.missing %}
                <li>{{ item }}</li>
              {% endfor %}
            </ul>
          </div>

          <div class="section">
            <h3>Retention Risk Reason</h3>
            <p>{{ result.retention_reason }}</p>
          </div>

          <div class="section">
            <h3>Questions You Can Ask</h3>
            <div class="qgrid">
              <div class="qbox">
                <strong>Questions for Best First Stop</strong>
                <ul>
                  {% for q in result.questions.questions_for_best_first_stop %}
                    <li>{{ q }}</li>
                  {% endfor %}
                </ul>
              </div>

              <div class="qbox">
                <strong>Questions for Final Decision Owner</strong>
                <ul>
                  {% for q in result.questions.questions_for_final_decision_owner %}
                    <li>{{ q }}</li>
                  {% endfor %}
                </ul>
              </div>

              <div class="qbox">
                <strong>Questions for Myself Before I Ask</strong>
                <ul>
                  {% for q in result.questions.questions_for_myself_before_i_ask %}
                    <li>{{ q }}</li>
                  {% endfor %}
                </ul>
              </div>
            </div>
          </div>

          <div class="section">
            <h3>How to Phrase It</h3>
            <div class="script">{{ result.script }}</div>
            <p style="margin-top:10px;">
              <a href="/history" style="font-size:14px;color:#174f8a;">View your history</a>
            </p>
          </div>
        {% else %}
          <div class="empty">
            <p><strong>No analysis yet.</strong></p>
            <p>Enter a real work situation on the left to see diagnosis, routing, action steps, targeted questions, and a ready-to-use message.</p>
          </div>
        {% endif %}
      </div>
    </div>
  </div>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    get_user_uuid()
    return render_template_string(
        HTML,
        result=None,
        error=None,
        form={
            "role": "Entry-Level Business Analyst",
            "week": "Week 2",
            "confidence": "Low",
            "interaction": "None",
            "situation": "",
        },
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    role = request.form.get("role", "Entry-Level Business Analyst").strip() or "Entry-Level Business Analyst"
    week = request.form.get("week", "Week 2")
    confidence = request.form.get("confidence", "Low")
    interaction = request.form.get("interaction", "None")
    situation = request.form.get("situation", "").strip()

    form = {
        "role": role,
        "week": week,
        "confidence": confidence,
        "interaction": interaction,
        "situation": situation,
    }

    if not situation:
        return render_template_string(
            HTML,
            result=None,
            error="Please describe the situation first.",
            form=form,
        )

    try:
        result = call_model(role, week, confidence, interaction, situation)
        user_uuid = get_user_uuid()
        role_cluster = map_role_to_cluster(role)
        save_submission(
            user_uuid=user_uuid,
            role=role,
            role_cluster=role_cluster,
            week=week,
            confidence=confidence,
            interaction=interaction,
            situation=situation,
            result=result,
        )
        return render_template_string(HTML, result=result, error=None, form=form)
    except Exception as e:
        return render_template_string(HTML, result=None, error=f"Model call failed: {str(e)}", form=form)

@app.route("/history")
def history():
    user_uuid = get_user_uuid()
    rows = fetch_user_history(user_uuid)

    cards = ""
    for timestamp, role, primary_issue, retention_risk, script in rows:
        cards += f"""
        <div class="history-card">
            <p class="timestamp">{escape(timestamp)}</p>
            <p><strong>Role:</strong> {escape(role)}</p>
            <p><strong>Primary Issue:</strong> {escape(primary_issue)}</p>
            <p><strong>Retention Risk:</strong> {escape(retention_risk)}</p>
            <p><strong>Generated Script:</strong></p>
            <div class="script-box">{escape(script)}</div>
        </div>
        """

    if not cards:
        cards = """
        <div class="empty">
            <p><strong>No history yet.</strong></p>
            <p>Once you analyze a situation, your past submissions will appear here.</p>
        </div>
        """

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Your Clarity History</title>
      <style>
        body {{
          margin:0;
          font-family:Arial,sans-serif;
          background:#f4f7fb;
          color:#1f2937;
        }}
        .wrap {{
          max-width:900px;
          margin:28px auto;
          padding:0 16px 40px;
        }}
        .hero,.history-card,.empty {{
          background:#ffffff;
          border:1px solid #dbe4f0;
          border-radius:18px;
          box-shadow:0 8px 24px rgba(16,24,40,.05);
        }}
        .hero {{
          padding:24px;
          margin-bottom:18px;
        }}
        h1 {{
          margin:0 0 8px;
          color:#123f6d;
        }}
        .history-card {{
          padding:18px;
          margin-bottom:14px;
        }}
        .timestamp {{
          color:#667085;
          font-size:13px;
          margin-top:0;
        }}
        .script-box {{
          background:#f8fafc;
          border:1px solid #dbe4f0;
          border-radius:12px;
          padding:12px;
          line-height:1.6;
        }}
        .empty {{
          padding:18px;
          color:#667085;
        }}
        a {{
          color:#174f8a;
          text-decoration:none;
          font-weight:700;
        }}
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="hero">
          <h1>Your History</h1>
          <p>Only submissions linked to your anonymous browser ID are shown here.</p>
          <p><a href="/">Back to Clarity Copilot</a></p>
        </div>
        {cards}
      </div>
    </body>
    </html>
    """


@app.route("/admin")
def admin():
    password = request.args.get("password")

    if password != "admin123":
        return "Unauthorized", 401

    data = fetch_admin_aggregates()

    def make_list(rows: List[Tuple[Any, Any]]) -> str:
        if not rows:
            return "<li>No data yet.</li>"
        return "".join([f"<li><span>{escape(label or 'Unknown')}</span><strong>{escape(value)}</strong></li>" for label, value in rows])

    avg_risk_rows = []
    for role_cluster, avg in data["avg_risk"]:
        avg_risk_rows.append((role_cluster or "Unknown", round(avg, 2) if avg is not None else "N/A"))

    return f"""
<html>
<body style="font-family:Arial;background:#f4f7fb;padding:24px;">
    <h2>Admin Dashboard</h2>
    <div style="background:white;padding:16px;border-radius:14px;margin-bottom:16px;">
        <h3>Total Submissions</h3>
        <p>{data["total"]}</p>
    </div>
    <div style="background:white;padding:16px;border-radius:14px;margin-bottom:16px;">
        <h3>Primary Issues</h3>
        <ul>{make_list(data["primary_issues"])}</ul>
    </div>
    <div style="background:white;padding:16px;border-radius:14px;margin-bottom:16px;">
        <h3>Behavioral Barriers</h3>
        <ul>{make_list(data["barriers"])}</ul>
    </div>
    <div style="background:white;padding:16px;border-radius:14px;margin-bottom:16px;">
        <h3>Retention Risk</h3>
        <ul>{make_list(data["risks"])}</ul>
    </div>
    <div style="background:white;padding:16px;border-radius:14px;margin-bottom:16px;">
        <h3>Role Clusters</h3>
        <ul>{make_list(data["role_clusters"])}</ul>
    </div>
    <div style="background:white;padding:16px;border-radius:14px;margin-bottom:16px;">
        <h3>Week in Role</h3>
        <ul>{make_list(data["weeks"])}</ul>
    </div>
    <div style="background:white;padding:16px;border-radius:14px;margin-bottom:16px;">
        <h3>Average Retention Risk by Role Cluster</h3>
        <ul>{make_list([(a, round(b, 2)) for a, b in data["avg_risk"]])}</ul>
    </div>
    <a href="/">Back to Clarity Copilot</a>
</body>
</html>
"""

if __name__ == "__main__":
    init_db()
    app.run(host="127.0.0.1", port=5001, debug=True)