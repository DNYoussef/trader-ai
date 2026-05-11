# Mieza Production Spine

The v1 production spine is intentionally narrow:

```text
mieza-quant approved signals
  -> signed mieza.signal.v1 JSON envelope in Trader-AI inbox
  -> trader-ai inbox processor
  -> committee review
  -> MOO sizing recommendation
  -> prediction-market risk gate
  -> dry-run execution journal
```

## Authority Boundaries

- `mieza-quant` is the prediction-market alpha producer. It scans, models, solves, and exports signed signal envelopes.
- `trader-ai` is the execution and risk authority. It verifies signatures, rejects replay, persists evidence, applies committee/MOO/risk gates, and records execution results.
- MOO sizing is advisory. It can shrink or reject a candidate size, but it cannot grant execution permission.
- Live prediction-market execution remains opt-in only and is outside this v1 spine.

## Out Of Path For V1

- `quant-lab` is a research and falsification workbench.
- `atlas` is an experiment harness.
- MiroFish is quarantined until an adapter implements its full ontology/graph/simulation lifecycle and proves service health.
- Hummingbot is reference/future execution infrastructure.
- Railway is a deployment verification gate after local dry-run acceptance, not a dependency of the local spine.

## Atomic Inbox Protocol

- `mieza-quant` writes envelopes through a temp file and atomically renames the final `*.json` into the inbox.
- `trader-ai` claims each ready `*.json` by atomically moving it into `processing/` before ingesting.
- Accepted envelopes move to `processed/`.
- Exact replay files move to `processed/duplicates/` and do not execute again.
- Invalid, tampered, stale, or nonce-collision files move to `rejected/` with an `.error.json` sidecar.
- Every processed file gets a `.manifest.json` sidecar with source path, final path, SHA-256, status, and counts.

## Local Acceptance

1. Export a signed envelope from `mieza-quant` using `MIEZA_SIGNAL_SIGNING_KEY`:

```bash
MIEZA_SIGNAL_SIGNING_KEY=... ./run.sh export-signals signals.edn
```

Set `TRADER_AI_MIEZA_INBOX` to override the default inbox path.

2. Process the inbox locally:

```bash
python -m src.integration.mieza_inbox_cli --inbox data/inbox/mieza-quant --db data/mieza_signals.db
```

3. Inspect the spine before trusting the handoff:

```bash
python -m src.integration.mieza_spine_status_cli --inbox data/inbox/mieza-quant --db data/mieza_signals.db
```

The status command is read-only by default. It returns `ok`, `degraded`, or `blocked`.

Use explicit recovery only for stale claimed files:

```bash
python -m src.integration.mieza_spine_status_cli --inbox data/inbox/mieza-quant --db data/mieza_signals.db --recover-stale-processing
```

Recovery moves a stale `processing/*.json` file back to the ready inbox only when no processed, duplicate, or rejected archive exists for the same SHA-256.

4. Confirm the SQLite journal contains the nonce, alpha event, committee decision, MOO decision, risk decision, and `dry_run_accepted` execution result.

## Producer Outbox Check

Before exporting files from `mieza-quant`, verify the producer side:

```bash
MIEZA_SIGNAL_SIGNING_KEY=... ./run.sh export-signals --check-outbox
```

The check verifies the signing key is present, resolves the configured outbox, probes writability, and blocks if stranded temp files are present.

## Deployment Gate

The local spine can be accepted without Railway, but production deployment is blocked until the deployment reality audit is clean. See `docs/DEPLOYMENT-REALITY-AUDIT.md`.
