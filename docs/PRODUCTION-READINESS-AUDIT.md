# Production Readiness Audit

Date: 2026-05-11

Scope: `trader-ai`, `mieza-ai/quant`, `quant-lab`, Mieza, GlobalMOO, MiroFish, Railway deployment state, and the signed signal-detection spine.

## Verdict

This is not production-ready as a hedge fund system.

It is closest to a governed dry-run prediction-market signal pipeline. The signed Mieza handoff and trader-ai ingestion path are the best-built pieces. The wider platform story around GitHub controls, Railway release hygiene, GlobalMOO, and MiroFish is still too loose for live capital.

## Quantified Readiness

| Area | Readiness | Reason |
| --- | ---: | --- |
| Signed Mieza signal export | 80% | `mieza-quant` now exports `mieza.signal.v1` envelopes with required signing key checks and outbox checks. Needs CI and repeated dry-run evidence. |
| Trader-AI signed inbox ingestion | 75% | Atomic claim/process/archive flow, duplicate handling, rejected sidecars, manifests, and dry-run journal exist locally. Needs deployed proof and scheduled operation. |
| Signal detection spine overall | 65% | Good local mechanics from signed signal to dry-run result. Still lacks long-running production telemetry, daily acceptance reports, and outcome calibration. |
| Mieza as alpha producer | 55% | Mieza can be a producer of approved signal envelopes. It is not wired as a live execution authority and should not become one. |
| GlobalMOO | 35% | Code exists, but production use falls back to mock/local behavior when keys or API are absent. Current prediction-market MOO path uses `deterministic_local_v1`, not GlobalMOO. |
| MiroFish | 25% | Railway service is up in `quant-lab`, but route contracts and lifecycle integration remain wrong or incomplete. Keep it out of the execution path. |
| Railway deployment readiness | 45% | `trader-ai` public health is currently 200 from an older active deployment, but the latest deployment is failed/stopped. That is a broken release pipeline. |
| GitHub/release governance | 25% | No GitHub Actions workflows were found for `trader-ai` or `mieza-ai/quant`; `trader-ai` main is unprotected; `mieza-ai/quant` protection was not proved. |
| Live trading readiness | 15% | The controls needed for real money are not present: protected releases, CI gates, deployment health, secrets audit, broker kill switch evidence, and dry-run burn-in. |

Overall: **40% production-ready for dry-run signal operations; 15% production-ready for live capital.**

## Current Proven State

- GitHub CLI is authenticated as `DNYoussef`.
- Railway CLI is authenticated as `David Youssef <dydavidyoussef@gmail.com>`.
- `DNYoussef/trader-ai` is public, default branch `main`, and branch protection API returns `Branch not protected`.
- `mieza-ai/quant` is private, default branch `main`, and branch protection was not proved by the GitHub API.
- `gh workflow list` returned no workflows for both core repos.
- `trader-ai` Railway project `e211a4c5-bc03-48f2-ab2f-bc67b27b3a9d` has latest app deployment `d490519d-0fd5-48d0-969e-7f13b5f01d57` in `FAILED` status, stopped.
- `trader-ai` public health endpoint `https://trader-ai-production.up.railway.app/health` currently returns 200 from an older successful deployment.
- `trader-ai` Railway app service has no explicit `HOST` variable. The code now defaults to `0.0.0.0` when Railway runtime markers are present.
- `quant-lab` Railway services `web`, `worker`, `mirofish`, and `Postgres` are currently `SUCCESS`.
- `guardspine` and `guardspine-ai-ops` Railway service groups are currently `SUCCESS`.

## Component Audit

### Mieza

Mieza should stay a producer. It should emit signed candidate signals and evidence. Trader-AI should own verification, committee review, risk gates, sizing, and execution journaling.

The new producer contract is correct in shape: `mieza.signal.v1`, signing key required, atomic outbox, explicit check mode. That is a reasonable spine. It still needs CI, a scheduled run, and a 7-day dry-run acceptance report before it deserves trust.

### Trader-AI Signal Detection

The local spine is stronger than the deployment story. The inbox processor claims files atomically, verifies signatures through the existing bridge, rejects stale/tampered/replayed input, archives results, and writes manifests. The status CLI is read-only by default and blocks on stale processing state.

What is missing is operational evidence: a scheduled producer, a scheduled consumer, daily manifests, outcome reflection, alerting, and a hard guarantee that live execution cannot be enabled by accident.

### GlobalMOO

GlobalMOO is not a production dependency today.

Evidence:

- `trader-ai` has GlobalMOO and WovenMOO code under `src/optimization/`.
- `trader-ai` feature flag `use_globalmoo` is false.
- Current prediction-market MOO allocation uses `optimizer_source = "deterministic_local_v1"`.
- `quant-lab` has a GlobalMOO adapter, but it returns empty/failure without `GMOO_API_KEY` and otherwise can fall back to mock behavior.

Production rule: GlobalMOO can be advisory only until API keys, response schemas, failure handling, result persistence, and no-mock production enforcement are proved.

### MiroFish

MiroFish is not production-integrated.

Evidence:

- `quant-lab` has a successful Railway `mirofish` service.
- `quant-lab` adapter documentation says the lifecycle is incomplete and scheduler usage remains disabled.
- The required MiroFish lifecycle is not a single `/simulate` call. It is graph/ontology setup, simulation create, prepare, start, and run-status polling.
- `guardspine-decision-engine-railway` still calls `POST {MIROFISH_SIM_URL}/simulate` and builds poll URLs under `/simulate/{job_id}`.
- The MiroFish backend exposes `/api/simulation/create`, `/api/simulation/prepare`, `/api/simulation/start`, and `/api/simulation/<id>/run-status`.

Production rule: MiroFish stays quarantined until the adapter implements the real lifecycle, persists simulation artifacts, proves a live round trip, and stops returning fake confidence when unavailable.

### Railway

Railway is not clean. The visible app can be healthy while the newest deployment is failed. That means a user can see green while the release train is red.

Minimum acceptable state:

1. Latest `trader-ai` deployment must be `SUCCESS`.
2. `/health` must return 200 on the latest deployment.
3. Runtime logs must show the app binding to `0.0.0.0:$PORT` or equivalent Railway-compatible host binding.
4. The deployment must reference the pushed commit containing the host-binding fix and spine docs.

## Hard Gates

1. No live trading until `trader-ai` and `mieza-ai/quant` have CI, branch protection or rulesets, and required checks on `main`.
2. No live trading until the latest `trader-ai` Railway deployment is successful and verified through `/health`.
3. No live prediction-market execution until the signed Mieza spine has 7 consecutive dry-run days with zero blocked status from the spine doctor.
4. No GlobalMOO production authority until production disables mock fallback and persists every optimizer request/response hash.
5. No MiroFish production authority until the lifecycle adapter is implemented and live round-trip tested against Railway.
6. No service can grant execution authority except trader-ai risk/execution code. Mieza, MiroFish, and GlobalMOO are inputs only.

## Next Slice

The next useful slice is deployment and governance, not more model code:

1. Commit and push the local `trader-ai` and `mieza-quant` spine changes.
2. Redeploy `trader-ai` and verify Railway latest deployment, runtime bind host, and `/health`.
3. Add minimal GitHub Actions for the focused Python and Clojure spine tests.
4. Enable branch protection or repository rulesets requiring those checks.
5. Add a daily dry-run job and a machine-readable spine report.
6. Keep MiroFish and GlobalMOO out of execution until their adapters have production no-mock contracts.
