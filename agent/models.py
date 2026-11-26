"""
The model queue: several free LLMs, tried in order of quality, switched
automatically when one is busy or out of quota (log 3.52).

WHY
    Free tiers are small (about 20 requests a day per Gemini model) and busy
    (Gemini answered 503 four times in a row on case-14, 3.50). One model is
    not enough to run the test cases; several providers' free tiers together
    are.

THE QUEUE (SH_MODEL_QUEUE, or DEFAULT_QUEUE below), best first
    provider:model[@tier]      tier "fix" (default) or "diagnose"
    A model is only in the queue when its provider's key is set, so the agent
    still works with Gemini alone.

WHAT HAPPENS ON A FAILED CALL
    busy (503, 500, 502, 504, timeout, connection)  rest the model a few minutes
    rate limit per minute (429)                     rest it for the time the API says
    quota per day (429 "per day", 402)              skip it until the reset
    bad key / unknown model / request rejected      skip it for this run
    The NEXT model then continues the SAME investigation: nothing read so far
    is lost (known issue 34). When every model is resting, the queue waits
    for the first one to come back if that is soon; otherwise it gives up and
    the agent hands the failure to a person, as before.

CHANGING MODEL MID-INVESTIGATION
    Providers store tool calls differently (Gemini 3 also attaches signatures
    to them), so one provider's history can be rejected by another. When the
    model changes, the earlier steps are given to the new model as plain text
    - what was called and what came back - in one message. Every provider
    accepts that, and it is shorter.

WHERE THE MEMORY LIVES
    Rest periods are saved to a small JSON file (SH_QUEUE_STATE), so the next
    run does not waste calls on a model that is out of quota until tomorrow.
    The GitHub Action keeps that file between runs in the Actions cache.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import config

# ----------------------------------------------------------------- providers
PROVIDERS = {
    "gemini": {"keys": ("GOOGLE_API_KEY", "GEMINI_API_KEY"), "base_url": None,
               "reset": "America/Los_Angeles"},      # Google: midnight Pacific
    "groq": {"keys": ("GROQ_API_KEY",), "base_url": "https://api.groq.com/openai/v1",
             "reset": "UTC"},
    "openrouter": {"keys": ("OPENROUTER_API_KEY",), "base_url": "https://openrouter.ai/api/v1",
                   "reset": "UTC"},
}

# Best first, by expected quality for a multi-step tool-using agent (26 Nov
# 2026). Positions 1 and 4 are the Gemini models this project has measured
# (3.43-3.50); the others are ranked from their providers' descriptions and
# should be re-ranked with the multi-language test results.
#   gemini-4-hydrogen      healed the 4-fault gauntlet twice in one attempt
#   gpt-oss-120b (Groq)    Groq's production reasoning model; all Groq models
#                          support tool use; very fast
#   nemotron-3-ultra       550B mixture-of-experts reasoning model "for
#                          long-running agent workflows"; free on OpenRouter,
#                          50 requests a day, prompts may be logged
#   gemini-3.5-flash       measured: weaker (missed a fault, dropped pins)
#   llama-3.3-70b (Groq)   older (2024) model: diagnose only
#   gemma-4-26b            small model with native function calling: diagnose only
DEFAULT_QUEUE = ", ".join([
    "gemini:{main}",
    "groq:openai/gpt-oss-120b",
    "openrouter:nvidia/nemotron-3-ultra-550b-a55b:free",
    "gemini:{fallback}",
    "groq:llama-3.3-70b-versatile@diagnose",
    "openrouter:google/gemma-4-26b-a4b-it:free@diagnose",
])

TIERS = ("fix", "diagnose")
BUSY_REST = 120.0          # seconds a busy model rests if the API gives no time
MINUTE_REST = 60.0         # seconds after a per-minute rate limit with no time given
RESULT_CHARS = 6000        # a tool result's length when replayed to a new model


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str
    tier: str = "fix"

    @property
    def id(self) -> str:
        return f"{self.provider}:{self.model}"

    def key(self, env=None) -> str | None:
        env = os.environ if env is None else env
        for name in PROVIDERS[self.provider]["keys"]:
            if env.get(name):
                return env[name]
        return None


def parse_queue(text: str) -> list[ModelSpec]:
    """'gemini:gemini-4-hydrogen, openrouter:x/y:free@diagnose' -> specs."""
    specs = []
    for item in (text or "").split(","):
        item = item.strip()
        if not item:
            continue
        tier = "fix"
        if "@" in item:
            item, tier = item.rsplit("@", 1)
            tier = tier.strip().lower()
        provider, _, model = item.partition(":")
        provider, model = provider.strip().lower(), model.strip()
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r} in the model queue "
                             f"(known: {', '.join(PROVIDERS)})")
        if not model:
            raise ValueError(f"no model named after {provider!r} in the model queue")
        if tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, not {tier!r}")
        specs.append(ModelSpec(provider, model, tier))
    return specs


def configured_queue(main: str | None = None, fallback: str | None = None) -> list[ModelSpec]:
    """The queue to use: SH_MODEL_QUEUE if set, else DEFAULT_QUEUE with the
    Gemini models from SH_GEMINI_MODEL / SH_GEMINI_FALLBACK_MODEL (so the
    older settings keep working; an empty fallback removes that slot)."""
    text = os.environ.get("SH_MODEL_QUEUE", "").strip()
    if not text:
        main = main or config.GEMINI_MODEL
        fallback = config.GEMINI_FALLBACK_MODEL if fallback is None else fallback
        text = DEFAULT_QUEUE.replace("{main}", main)
        text = (text.replace("{fallback}", fallback) if fallback
                else ", ".join(p for p in text.split(", ") if "{fallback}" not in p))
    specs, seen = [], set()
    for s in parse_queue(text):
        if s.id not in seen:                     # e.g. main == fallback
            seen.add(s.id)
            specs.append(s)
    return specs


def make_chat(spec: ModelSpec):
    """The LangChain chat model for one queue entry. Retries are left to the
    queue (it switches instead of waiting), so each client retries little."""
    if spec.provider == "gemini":
        from .llm import make_llm
        return make_llm(spec.model, max_retries=0)
    from langchain_openai import ChatOpenAI       # Groq and OpenRouter speak the OpenAI API
    return ChatOpenAI(model=spec.model, base_url=PROVIDERS[spec.provider]["base_url"],
                      api_key=spec.key(), temperature=0, max_retries=0, timeout=180,
                      default_headers={"X-Title": "self-healing-cicd"})


# ------------------------------------------------------------ error sorting
def _status(error) -> int | None:
    for obj in (error, getattr(error, "response", None)):
        for attr in ("status_code", "code", "status"):
            value = getattr(obj, attr, None)
            if isinstance(value, int) and 100 <= value < 600:
                return value
    m = re.match(r"\D{0,40}?\b([45]\d\d)\b", str(error))
    return int(m.group(1)) if m else None


def retry_after(text: str) -> float | None:
    """Seconds the API asks us to wait, if it says: 'retryDelay': '41s',
    'try again in 7m12.5s', 'retry in 11h42m', 'Retry-After: 30'."""
    m = re.search(r"retry_?delay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", text, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"(?:try again|retry) in (?:about )?((?:\d+h)?(?:\d+m)?(?:\d+(?:\.\d+)?s)?)", text, re.I)
    if m and m.group(1):
        h = re.search(r"(\d+)h", m.group(1))
        mi = re.search(r"(\d+)m(?!s)", m.group(1))
        s = re.search(r"(\d+(?:\.\d+)?)s", m.group(1))
        total = (int(h.group(1)) * 3600 if h else 0) + (int(mi.group(1)) * 60 if mi else 0) \
            + (float(s.group(1)) if s else 0)
        return total or None
    m = re.search(r"retry-after['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)", text, re.I)
    return float(m.group(1)) if m else None


def classify(error: Exception) -> tuple[str, float | None]:
    """(kind, seconds): kind is busy | rate | quota | auth | missing | rejected | error."""
    text = f"{type(error).__name__}: {error}"
    low = text.lower()
    status = _status(error)
    wait = retry_after(text)
    if status in (401, 403) or any(w in low for w in (
            "invalid api key", "api key not valid", "unauthorized", "permission denied",
            "authenticationerror", "permissiondeniederror")):
        return "auth", None
    if status == 402 or "insufficient credits" in low or "negative balance" in low:
        return "quota", wait
    if status == 429 or any(w in low for w in ("resource_exhausted", "rate limit", "ratelimit",
                                               "too many requests", "quota")):
        daily = any(w in low for w in ("per day", "perday", "per_day", "daily",
                                       "free-models-per-day", "(rpd)", "(tpd)", "requests per day",
                                       "tokens per day"))
        return ("quota", wait) if daily else ("rate", wait or MINUTE_REST)
    if status in (500, 502, 503, 504) or any(w in low for w in (
            "overloaded", "unavailable", "high demand", "timed out", "timeout",
            "connection", "temporarily")):
        return "busy", wait or BUSY_REST
    if status == 404 or any(w in low for w in ("model not found", "does not exist",
                                               "no endpoints found", "notfounderror",
                                               "decommissioned")):
        return "missing", None
    if status in (400, 413, 422) or "badrequesterror" in low:
        return "rejected", None
    return "error", None


def next_reset(provider: str, now: float) -> float:
    """When a provider's daily quota resets after `now` (epoch seconds)."""
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(PROVIDERS[provider]["reset"])
    local = datetime.fromtimestamp(now, tz)
    midnight = (local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight.timestamp()


class QueueUnavailable(RuntimeError):
    """Every model in the queue is resting or unusable."""


# ------------------------------------------------------------------- queue
@dataclass
class _Rest:
    until: float               # epoch seconds; float("inf") = for this run
    kind: str
    why: str


@dataclass
class ModelQueue:
    """Duck-types a LangChain chat model for the agent: bind_tools() ->
    something with invoke(messages). See the module text."""
    specs: list
    factory: object = make_chat
    state_path: str | None = None
    max_wait: float = 180.0          # wait at most this long for a resting model ...
    max_total_wait: float = 600.0    # ... and this long in total per run
    clock: object = time.time
    sleep: object = time.sleep
    say: object = print
    rest: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    used: list = field(default_factory=list)      # model id per successful call
    waited: float = 0.0
    _chats: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.specs:
            raise RuntimeError(
                "No model can be used: set GOOGLE_API_KEY (Gemini), GROQ_API_KEY or "
                "OPENROUTER_API_KEY for at least one model in the queue.")
        self._load()

    # ----- describing it
    @property
    def ids(self) -> list[str]:
        return [s.id for s in self.specs]

    def tier_of(self, model_id: str | None) -> str:
        for s in self.specs:
            if s.id == model_id:
                return s.tier
        return "fix"

    @property
    def last(self) -> str | None:
        return self.used[-1] if self.used else None

    def describe(self) -> str:
        now = self.clock()
        parts = []
        for i, s in enumerate(self.specs, 1):
            r = self.rest.get(s.id)
            state = ""
            if r and r.until > now:
                state = (f" - {r.kind}, skipped this run" if r.until == float("inf")
                         else f" - {r.kind}, back in {_human(r.until - now)}")
            parts.append(f"{i}. {s.id}{' (diagnose only)' if s.tier == 'diagnose' else ''}{state}")
        return "; ".join(parts)

    # ----- the agent's interface
    def bind_tools(self, tools, tool_choice=None):
        return _Bound(self, list(tools), tool_choice)

    # ----- state
    def _available(self, exclude) -> ModelSpec | None:
        now = self.clock()
        for s in self.specs:
            r = self.rest.get(s.id)
            if s.id not in exclude and not (r and r.until > now):
                return s
        return None

    def _mark(self, spec: ModelSpec, error: Exception) -> None:
        kind, seconds = classify(error)
        now = self.clock()
        if kind in ("auth", "missing", "rejected", "error"):
            until = float("inf")
        elif kind == "quota":
            until = now + seconds if seconds else next_reset(spec.provider, now)
        else:
            until = now + (seconds or BUSY_REST)
        why = " ".join(str(error).split())[:240]
        self.rest[spec.id] = _Rest(until, kind, why)
        self.events.append({"model": spec.id, "kind": kind, "at": now,
                            "until": None if until == float("inf") else until, "why": why})
        self.say(f"[queue] {spec.id}: {kind}"
                 + ("" if until == float("inf") else f", resting {_human(until - now)}")
                 + f" ({why[:120]})")
        self._save()

    def _load(self) -> None:
        if not self.state_path or not Path(self.state_path).exists():
            return
        try:
            data = json.loads(Path(self.state_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return                               # a damaged file must not stop the agent
        now = self.clock()
        for model_id, r in (data or {}).items():
            if isinstance(r, dict) and float(r.get("until") or 0) > now:
                self.rest[model_id] = _Rest(float(r["until"]), r.get("kind", "?"), r.get("why", ""))

    def _save(self) -> None:
        if not self.state_path:
            return
        now = self.clock()
        data = {k: {"until": r.until, "kind": r.kind, "why": r.why}
                for k, r in self.rest.items() if r.until != float("inf") and r.until > now}
        try:
            Path(self.state_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.state_path).write_text(json.dumps(data, indent=1), encoding="utf-8")
        except OSError:
            pass

    def _soonest(self) -> float | None:
        now = self.clock()
        waits = [r.until - now for r in self.rest.values()
                 if r.until != float("inf") and r.until > now]
        return min(waits) if waits else None

    def _chat(self, spec: ModelSpec, tools, tool_choice):
        key = (spec.id, id(tools), tool_choice)
        if key not in self._chats:
            base = self.factory(spec)
            if tool_choice is None:
                self._chats[key] = base.bind_tools(tools)
            else:
                try:
                    self._chats[key] = base.bind_tools(tools, tool_choice=tool_choice)
                except TypeError:
                    self._chats[key] = base.bind_tools(tools)
        return self._chats[key]


class _Bound:
    def __init__(self, queue: ModelQueue, tools, tool_choice):
        self.queue, self.tools, self.tool_choice = queue, tools, tool_choice

    def invoke(self, messages):
        q = self.queue
        tried = set()
        while True:
            spec = q._available(tried)
            if spec is None:
                wait = q._soonest()
                if (wait is not None and wait <= q.max_wait
                        and q.waited + wait <= q.max_total_wait):
                    q.say(f"[queue] every model is resting; waiting {_human(wait)} for the first one")
                    q.sleep(wait)
                    q.waited += wait
                    tried.clear()
                    continue
                busy = any(r.kind in ("busy", "rate") and r.until != float("inf")
                           for r in q.rest.values())
                raise QueueUnavailable(
                    "every model in the queue is "
                    + ("overloaded, rate-limited or out of quota" if busy else "out of quota or unusable")
                    + ": " + q.describe())
            try:
                chat = q._chat(spec, self.tools, self.tool_choice)
                previous = q.last
                ai = chat.invoke(portable(messages, spec.id))
            except Exception as e:                       # noqa: BLE001
                q._mark(spec, e)
                tried.add(spec.id)
                continue
            if previous and previous != spec.id:
                q.say(f"[queue] {spec.id} continues the investigation (was {previous})")
            try:
                ai.response_metadata["sh_model"] = spec.id
            except (AttributeError, TypeError):
                pass
            q.used.append(spec.id)
            return ai


# ------------------------------------------------------ portable history
def portable(messages: list, model_id: str) -> list:
    """The conversation as `model_id` can read it. Unchanged while every
    earlier answer came from this model; otherwise the steps after the first
    request become one plain-text summary (see the module text)."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    others = {(m.response_metadata or {}).get("sh_model") for m in messages
              if isinstance(m, AIMessage)} - {None, model_id}
    if not others:
        return messages
    system = [m for m in messages[:1] if isinstance(m, SystemMessage)]
    rest = messages[len(system):]
    first = next((i for i, m in enumerate(rest) if isinstance(m, HumanMessage)), None)
    if first is None:
        return messages
    request, steps = rest[first], rest[first + 1:]
    lines = ["", "---", "EARLIER STEPS of this investigation, done by another model "
             "(it became unavailable). What it called and what came back:"]
    final = None
    for i, m in enumerate(steps):
        if isinstance(m, AIMessage):
            text = _plain(m.content).strip()
            if text:
                lines.append(f"[model said] {text[:1500]}")
            for c in m.tool_calls or []:
                lines.append(f"[called] {c['name']}({json.dumps(c.get('args', {}), ensure_ascii=False)[:600]})")
        elif isinstance(m, ToolMessage):
            content = str(m.content)
            if len(content) > RESULT_CHARS:
                content = content[:RESULT_CHARS] + f"\n... [{len(content) - RESULT_CHARS} more characters; read it again if needed]"
            lines.append(f"[result of {m.name or 'tool'}]\n{content}")
        elif isinstance(m, HumanMessage):
            if i == len(steps) - 1:
                final = _plain(m.content)
            else:
                lines.append(f"[note] {_plain(m.content)}")
    lines += ["---", final or ("Continue the investigation from here. Files read above count "
                               "as read. Use the tools if you need more, then call submit_diagnosis.")]
    return system + [HumanMessage(_plain(request.content) + "\n".join(lines))]


def _plain(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for p in content or []:
        if isinstance(p, str):
            parts.append(p)
        elif isinstance(p, dict) and p.get("type") == "text":
            parts.append(p.get("text", ""))
    return "\n".join(parts)


def _human(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds}s"


def build_queue(env=None, say=print, **kw) -> ModelQueue:
    """The queue of models whose keys are set, in order."""
    env = os.environ if env is None else env
    specs = configured_queue()
    usable = [s for s in specs if s.key(env)]
    skipped = [s.id for s in specs if not s.key(env)]
    if skipped:
        say("[queue] no key for: " + ", ".join(skipped))
    return ModelQueue(usable, say=say, state_path=kw.pop("state_path", config.QUEUE_STATE), **kw)
