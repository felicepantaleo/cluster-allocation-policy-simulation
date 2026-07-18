"""Extend the user-to-WP mapping: STEAM exclusion plus department rules.

    python tools/classify_users.py --steam-csv <participants.csv> \
        --derived data/derived

- STEAM ACADEMY participants (CSV: ...,id,Name,email) resolve to CERN
  usernames via LDAP (mail first, displayName fallback) and are tagged
  wp=STEAM, meaning: excluded from all statistics. The WP roster takes
  precedence: a user already mapped to a WP stays there (organizers and
  lecturers appear in the participant list too).
- Department rules for remaining unmapped users with GPU activity
  (PMC instruction): EP/CMS -> WP3, EP/ATL -> WP2, anything /SFT or
  IT/* -> WP1.
- Writes the updated user_wp.json and a remaining-to-classify table.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

LDAP_BASE = ["ldapsearch", "-x", "-H", "ldap://xldap.cern.ch", "-b",
             "OU=Users,OU=Organic Units,DC=cern,DC=ch"]


def ldap(filt: str, attrs: list[str]) -> list[dict]:
    out = subprocess.run(LDAP_BASE + [filt] + attrs, capture_output=True,
                         text=True, timeout=30, check=False).stdout
    recs, cur = [], {}
    for line in out.splitlines():
        if line.startswith("dn:"):
            if cur:
                recs.append(cur)
            cur = {}
        for a in attrs:
            if line.startswith(f"{a}: "):
                cur[a] = line.split(": ", 1)[1].strip()
    if cur:
        recs.append(cur)
    return recs


def resolve_person(name: str, email: str) -> list[str]:
    if email.endswith("@cern.ch"):
        recs = ldap(f"(|(mail={email})(proxyAddresses=smtp:{email}))", ["cn"])
        cns = [r["cn"].lower() for r in recs if "cn" in r]
        if cns:
            return cns
    toks = [t for t in re.split(r"[ -]", name) if len(t) > 2]
    filt = "(&(objectClass=user)" + "".join(
        f"(displayName=*{t}*)" for t in toks) + ")"
    return [r["cn"].lower() for r in ldap(filt, ["cn"]) if "cn" in r]


def dept_rule(dept: str) -> str | None:
    if not dept:
        return None
    if dept.endswith("/SFT") or dept.startswith("IT/"):
        return "WP1"
    if dept == "EP/CMS":
        return "WP3"
    if dept == "EP/ATL":
        return "WP2"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steam-csv", required=True)
    ap.add_argument("--derived", default="data/derived")
    args = ap.parse_args()
    der = Path(args.derived)
    wp_map = json.loads((der / "user_wp.json").read_text())

    reqs = [json.loads(l) for l in open(der / "requests.jsonl")]
    gh = defaultdict(float)
    for r in reqs:
        if r["gpus"] > 0 and r["pool"] not in ("cloud_t4", "cpu", "unknown") \
                and r["observed"]["outcome"] == "started":
            for a, b in r["observed"]["running_intervals"]:
                gh[r["user"]] += (b - a) * r["gpus"] / 3600
    namespaces = set(gh) | {r["user"] for r in reqs}

    n_steam = 0
    with open(args.steam_csv) as f:
        for row in csv.reader(f):
            if len(row) < 6 or not row[4].strip():
                continue
            name, email = row[4].strip(), row[5].strip()
            for cn in resolve_person(name, email):
                if cn in namespaces and cn not in wp_map:
                    wp_map[cn] = {"wp": "STEAM", "name": name}
                    n_steam += 1
    print(f"tagged {n_steam} STEAM participants with cluster activity")

    n_rule = 0
    remaining = []
    for user in sorted(gh, key=lambda u: -gh[u]):
        if user in wp_map:
            continue
        recs = ldap(f"(cn={user})", ["displayName", "department"])
        dn = recs[0].get("displayName", "?") if recs else "?"
        dept = recs[0].get("department", "") if recs else ""
        rule = dept_rule(dept)
        if rule:
            wp_map[user] = {"wp": rule, "name": dn, "via": f"dept {dept}"}
            n_rule += 1
        else:
            remaining.append((user, dn, dept, gh[user]))
    print(f"classified {n_rule} users by department rule")

    (der / "user_wp.json").write_text(json.dumps(wp_map, indent=2,
                                                 sort_keys=True))
    lines = ["# Still to classify (no roster entry, no department rule)",
             "", "| username | display name | department | GPU-h (30d) | wp |",
             "|---|---|---|---|---|"]
    for u, dn, dept, h in remaining:
        lines.append(f"| {u} | {dn} | {dept or '?'} | {h:.0f} |  |")
    (der / "remaining_to_classify.md").write_text("\n".join(lines) + "\n")
    print(f"{len(remaining)} users remain unclassified "
          f"({sum(h for *_, h in remaining):.0f} GPU-h); "
          "table: data/derived/remaining_to_classify.md")


if __name__ == "__main__":
    main()
