# ARCHIVED (KILL disposition)

Per the 2026-07-05 portfolio audit (SYNERGY-PLAN.md), the **equity-trading facade**
in this repo is **KILL-listed**: a non-booting entry point, ~300 dead SPEK files, and
fail-open real-money auth. It is kept as reversible git history (an anti-demo of what
a fail-open safety-critical system looks like), not as an active asset.

**Do not run the equity trader.** The B+ Mieza prediction-market spine is the part
worth keeping and is being consolidated toward `mieza-quant`.

Security/safety hardening applied before archival (so the archived artifact is honest):
- Card 3: `/api/trading/execute` auth now fails **closed** — the middleware raises 503
  instead of passing requests through when the token verifier is unavailable
  (`src/security/auth_middleware.py`, `src/dashboard/run_server_simple.py`).
- Card 5: the phantom `library.components` import is replaced with guarded no-op stubs
  so the entry point boots (`src/universal_components.py`).

Known remaining (intentionally not fixed — KILL): the `/ws` WebSocket path is
unauthenticated; take the Railway dashboard down rather than harden a facade.

This marker is reversible (`git rm ARCHIVED.md`).
