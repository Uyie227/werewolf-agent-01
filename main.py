#!/usr/bin/env python3
"""werewolf-agent-01 entry.

Heuristic Bayesian Werewolf agent with house-model debate parsing and an
optional Claude gateway call when a dummy key is present in the trial.

Standard library only. Deterministic given the observe views and model
replies: no wall-clock values are sent anywhere and no unseeded randomness
is used.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request

CONCURRENCY = 10
RETRY_WAIT_S = 5
RETRY_AFTER_MAX_S = 20
MAX_REQUEST_BYTES = 131_072
MODEL = "claude-haiku-5-5"

LOG_RE = re.compile(r"seat\s*(\d+)\s*([A-Za-z]{1,30})?\s*(claimed|claims|accused|accuses|defended|defends|voted|votes|bluff)", re.I)
ACCUSE_RE = re.compile(r"(seat\s*\d+|\b(?:they|he|she|player)\b|i think|i suspect|seems|probably|definitely|likely|wolf|werewolf|suspicious|cleared|trust|innocent|guilty|claimed (?:the )?seer|claimed (?:the )?doctor|as (?:the )?seer|as (?:the )?doctor|i am (?:the )?seer|i am (?:the )?doctor)\b", re.I)
SEAT_RE = re.compile(r"seat\s*(\d+)", re.I)


def post_json(url: str, payload: dict, timeout: int = 30):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"content-type": "application/json", "user-agent": "werewolf-agent"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # network only to env / gateway
            return resp.status, json.loads(resp.read().decode("utf-8", "ignore"))
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8", "ignore") or "{}")
        except Exception:
            body = {}
        return e.code, body
    except Exception:
        return 0, {}


def tools_for_brief(brief: str):
    return [brief] if brief else []


def rng_for_seed(seed_text: str, game: int, decision: str) -> int:
    h = hashlib.sha256(f"{seed_text}|{game}|{decision}".encode()).digest()
    return int.from_bytes(h, "big") % (10 ** 12)


def fold_statement(text: str) -> str:
    text = re.sub(r"[\t\n\r\f\v\u0085\u2028\u2029]+", " ", text)
    text = "".join(c for c in text if c >= " " or c.isspace())
    return " ".join(text.split()).strip()


def view_state(view):
    players = view.get("players") or []
    log = view.get("log") or []
    you = view.get("you") or {}
    my_seat = you.get("seat")

    alive = [p["seat"] for p in players if p.get("alive")]
    revealed = {p["seat"]: p.get("role") for p in players if p.get("role")}
    names = {p["seat"]: p.get("name") for p in players}

    statements = []
    votes = {}
    for entry in log:
        kind = entry.get("kind")
        if kind == "speech" and entry.get("seat") is not None:
            statements.append({"seat": entry["seat"], "text": fold_statement(str(entry.get("text") or "")), "day": entry.get("day")})
        elif kind == "vote" and entry.get("seat") is not None and entry.get("target") is not None:
            votes[entry["seat"]] = entry["target"]

    return {"alive": alive, "revealed": revealed, "names": names, "statements": statements, "votes": votes, "my_seat": my_seat}


def suspicion_scores(view, state):
    """Return suspicion scores for alive non-self seats."""
    role = (view.get("you") or {}).get("role")
    alive = [s for s in state["alive"] if s != state["my_seat"]]
    scores = {s: 0.0 for s in alive}

    # Revealed wolves are the strongest signal.
    for seat, r in state["revealed"].items():
        if seat in scores and r == "werewolf":
            scores[seat] += 10.0

    # Seer results, if we are the seer.
    for r in view.get("seerResults") or []:
        seat = r.get("seat")
        if seat in scores:
            scores[seat] += 10.0 if r.get("werewolf") else -3.0

    # Werewolf's own knowledge.
    if role == "werewolf":
        partner = (view.get("you") or {}).get("partner")
        if partner and partner.get("seat") in scores:
            scores[partner["seat"]] -= 5.0

    # Statement parsing.
    for st in state["statements"]:
        speaker = st["seat"]
        text = st["text"]
        if speaker not in state["alive"] or speaker == state["my_seat"]:
            continue
        m = SEAT_RE.search(text)
        if m:
            t = int(m.group(1))
            if t in scores:
                low = text.lower()
                if any(w in low for w in ("wolf", "werewolf", "suspicious", "guilty", "accus", "suspect")):
                    scores[t] += 0.8
                if any(w in low for w in ("innocent", "clear", "trust", "safe", "village")):
                    scores[t] -= 0.6
        low = text.lower()
        if "seer" in low or "doctor" in low:
            # Role claims add urgency to verify or protect the seer.
            # Keep the score small so it supports but does not override facts.
            if "seer" in low and role in ("werewolf", "doctor"):
                scores[speaker] = scores.get(speaker, 0.0) + 0.4
            if "doctor" in low and role == "seer":
                scores[speaker] = scores.get(speaker, 0.0) + 0.2

    # Vote pressure: seats that just received public votes are suspects.
    for target, count in _vote_counts(state["votes"]).items():
        if target in scores:
            scores[target] += 0.6 * count

    # If we're a werewolf, avoid teammates in kill decisions via the partner note above.
    return scores


def _vote_counts(votes):
    out = {}
    for t in votes.values():
        if t is not None:
            out[t] = out.get(t, 0) + 1
    return out


def choose_target(scores, options):
    if not options:
        return None
    valid = [s for s in options if s in scores]
    if not valid:
        return options[0]
    # Highest suspicion, deterministic tie break by seat.
    best = sorted(valid, key=lambda s: (-scores[s], s))[0]
    return best


def pick_victim(view, options):
    state = view_state(view)
    scores = suspicion_scores(view, state)
    return choose_target(scores, options)


def heuristic_action(pending, trial_seed: str):
    game = int(pending["game"])
    decision = pending["decision"]
    view = pending.get("view") or {}
    options = list(pending.get("options") or [])
    max_chars = int(pending.get("max_chars") or 500)
    role = (view.get("you") or {}).get("role")

    if decision == "speak":
        state = view_state(view)
        scores = suspicion_scores(view, state)
        alive_others = [s for s in state["alive"] if s != state["my_seat"]]
        # We only discuss real seats so the house model can follow the argument.
        if view.get("phase") == "over":
            return {"game": game, "decision": "speak", "pass": True}
        if not alive_others:
            return {"game": game, "decision": "speak", "pass": True}

        # If we are a werewolf, blend in and pressure a lead suspect but do not overclaim a role.
        if role == "werewolf":
            target = choose_target(scores, alive_others)
            if target is not None and target != state["my_seat"]:
                txt = f"I think seat {target} looks off today; their story keeps shifting. We should hear them out, but I lean seat {target}."
            else:
                txt = "I am watching the early votes closely; let us see who jumps on whom."
        else:
            target = choose_target(scores, alive_others)
            if view.get("seerResults"):
                findings = []
                for r in view.get("seerResults") or []:
                    findings.append(f"seat {r['seat']} is {'a werewolf' if r.get('werewolf') else 'not a werewolf'}")
                txt = "My checks: " + "; ".join(findings) + "."
            elif role == "seer":
                txt = "I am the seer. I will share my check when I have one; do not waste a vote on me."
            elif role == "doctor":
                txt = "I will keep someone protected tonight. If you have a check, speak up."
            elif target is not None:
                txt = f"Seat {target} has drawn a few looks and I want answers from seat {target} before the vote."
            else:
                txt = "I have nothing certain yet, so I will listen before choosing a vote."
        txt = fold_statement(txt)
        if len(txt) > max_chars:
            txt = txt[:max_chars].rstrip()
        if not txt:
            return {"game": game, "decision": "speak", "pass": True}
        return {"game": game, "decision": "speak", "text": txt}

    if decision == "vote":
        return {"game": game, "decision": "vote", "target": pick_victim(view, options)}

    if decision == "kill":
        # Werewolves: take the most clear suspicion player, avoid seer-looking speakers too bluntly.
        state = view_state(view)
        scores = suspicion_scores(view, state)
        # If public reasoning named a seer claim, killing them is acceptable there, but normally trim the noisy houses.
        return {"game": game, "decision": "kill", "target": choose_target(scores, options)}

    if decision == "protect":
        state = view_state(view)
        scores = suspicion_scores(view, state)
        # Protect the least suspicious living ally-ish seat; if the seer is alive and claimed, prefer them.
        last = view.get("lastProtect") or {}
        last_seat = last.get("seat")
        candidates = [s for s in options if s != last_seat]
        if not candidates:
            return {"game": game, "decision": "protect", "target": None}
        # Seer claim support.
        for st in state["statements"]:
            if "seer" in st["text"].lower() and st["seat"] in candidates:
                return {"game": game, "decision": "protect", "target": st["seat"]}
        return {"game": game, "decision": "protect", "target": choose_target({s: -scores.get(s, 0.0) for s in candidates}, candidates)}

    if decision == "inspect":
        state = view_state(view)
        scores = suspicion_scores(view, state)
        return {"game": game, "decision": "inspect", "target": choose_target(scores, options)}

    return {"game": game, "decision": decision, "target": None}


def run_trial(env_url: str, prompt_file: str, trial_seed: str):
    brief = ""
    try:
        brief = open(prompt_file, encoding="utf-8", errors="ignore").read()
    except Exception:
        brief = ""

    while True:
        st, obs = post_json(f"{env_url}/observe", {}, timeout=25)
        if st != 200:
            # No pending decision: wait briefly by re-observing.
            continue
        if obs.get("done"):
            break
        pending_list = obs.get("pending") or []
        for pending in pending_list:
            game = int(pending["game"])
            decision = pending["decision"]
            # Sequential acting is fine: at most one decision per live game waits.
            body = heuristic_action(pending, trial_seed)
            # A failed post returns a status; retry once with the same deterministic body.
            st, _ = post_json(f"{env_url}/act", body, timeout=30)
            if st not in (200,):
                amt = post_json(f"{env_url}/act", body, timeout=30)
                if amt[0] == 409 and amt[1].get("error") == "not_pending":
                    continue
                # If the act was refused, fall back to a pass / abstention rather than stalling forever.
                first = amt[0]
                if first == 400:
                    fallback = {"game": game, "decision": decision}
                    if decision == "speak":
                        fallback.update({"pass": True})
                    else:
                        fallback.update({"target": None})
                    post_json(f"{env_url}/act", fallback, timeout=30)


def main():
    env_url = (os.environ.get("MAKEFUN_ENV_URL") or "").rstrip("/")
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE") or ""
    trial_seed = os.environ.get("MAKEFUN_TRIAL_SEED") or "0"
    if not env_url:
        return
    run_trial(env_url, prompt_file, trial_seed)


if __name__ == "__main__":
    main()
