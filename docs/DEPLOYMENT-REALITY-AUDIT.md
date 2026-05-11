# Deployment Reality Audit

Date: 2026-05-11

This audit records what GitHub and the local CLIs could prove about the AI hedgefund / financial spine. It is a production gate, not a deploy script.

## CLI State

- GitHub CLI is authenticated as `DNYoussef`.
- Railway CLI is installed as `railway 4.30.5`.
- Railway CLI is authenticated as `David Youssef <dydavidyoussef@gmail.com>`.
- Railway workspace: `David Youssef's Projects` (`8e743f4d-7fb0-4f26-a7de-e40fb02a6383`).

Read-only Railway verification is now available. Do not mutate Railway projects, services, variables, or deployments from this audit without an explicit deploy/change step.

## GitHub To Railway Evidence

| Repo | Railway evidence from GitHub deployments | Latest status |
| --- | --- | --- |
| `DNYoussef/trader-ai` | Project `e211a4c5-bc03-48f2-ab2f-bc67b27b3a9d`, environment `be86f6a3-6c00-42c9-a360-7990decba85b` | `failure` |
| `DNYoussef/quant-lab` | Project `a5cf8542-2be3-4241-8ca1-e57bec93f600`, environment `8a9f3260-3f94-49e9-9540-39754c8a76d8` | GitHub deployment record showed `failure`; live Railway status is now `success` |
| `DNYoussef/GuardSpine` | Project `41e26c70-bb11-4274-90dd-0118ec5c8871`, environment `90782fe6-7771-46c1-a7aa-714230a0ee9e` | `success` |
| `DNYoussef/guardspine-litellm-railway` | Project `bf7584bc-0ee4-4271-955a-2076c133c43f`, environment `276f19c3-4309-405e-bb3a-83131a44e6fc` | `success` |
| `DNYoussef/guardspine-n8n-railway` | Project `bf7584bc-0ee4-4271-955a-2076c133c43f`, environment `276f19c3-4309-405e-bb3a-83131a44e6fc` | `success` |

## Live Railway Status

| Project | Services |
| --- | --- |
| `trader-ai` | latest `trader-ai` deployment failed/stopped; an older active deployment still serves `/health` 200; `Postgres` success |
| `quant-lab` | `web`, `worker`, `mirofish`, and `Postgres` success |
| `guardspine` | `frontend`, `backend`, and `Postgres` success |
| `guardspine-ai-ops` | `telemetry-api`, `paperclip`, `memory-mcp`, `Postgres`, `openclaw`, `n8n`, `guardspine-internal`, and `litellm` success |

Local repo linkage is not trustworthy:

- `D:\Projects\trader-ai` is not locally linked to Railway.
- `D:\Projects\quant-lab` is not locally linked to Railway.
- `D:\Projects\mirofish` is not locally linked to Railway.
- `D:\Projects\guardspine-litellm-railway` is not locally linked to Railway.
- `D:\Projects\guardspine-n8n-railway` is not locally linked to Railway.
- `D:\Projects\guardspine-decision-engine-railway` is not locally linked to Railway.
- `D:\Projects\GuardSpine` is locally linked to `guardspine-ai-ops`, not the `guardspine` project that GitHub deployment records use for `DNYoussef/GuardSpine`.

No GitHub Railway deployment records were found for:

- `mieza-ai/quant`
- `DNYoussef/atlas`
- `DNYoussef/mirofish`
- `DNYoussef/guardspine-decision-engine-railway`
- `mieza-ai/guardspine-negotiation`

Those services may be undeployed, manually deployed, deployed from a bundle, or linked to Railway outside their GitHub deployment records. Do not assume.

## Blocking Findings

- `trader-ai` latest Railway deployment is recorded as failed.
- `trader-ai` public `/health` currently returns 200 from an older successful active deployment, so public health alone is insufficient release proof.
- `trader-ai` latest deployment completed build, then Railway's `/health` check failed after the 30 second retry window.
- `trader-ai` has no explicit Railway `HOST` variable. The local app now defaults to `0.0.0.0` when Railway runtime markers are present.
- `quant-lab` is currently healthy on Railway despite the older GitHub deployment failure record.
- `guardspine-decision-engine-railway` has no deployment evidence from its own GitHub repo.
- `mirofish` exists as a successful service inside the `quant-lab` Railway project, not as an evidenced deployment from the `DNYoussef/mirofish` repo.
- Decision engine expects MiroFish `POST /simulate`, while MiroFish exposes a lifecycle under `/api/simulation/create`, `/api/simulation/prepare`, and `/api/simulation/start`.
- Core production-adjacent repos lack branch protection.
- CI is missing or weak across the core financial spine.
- Vulnerability alerts are disabled or inaccessible on the checked core repos.
- Local `trader-ai` and `mieza-quant` contain uncommitted bridge work, so GitHub and Railway are not running the current local spine changes.

## Gate Criteria

Production changes remain blocked until:

1. Push and redeploy the `trader-ai` Railway host-binding fix, then prove the latest deployment is `SUCCESS`.
2. Prove whether `guardspine-decision-engine-railway` is deployed anywhere, or remove it from the live path.
3. Prove the exact `quant-lab` `mirofish` service source and route contract.
4. The MiroFish route contract is fixed or the decision engine keeps MiroFish quarantined.
5. Branch protection or rulesets are enabled on financial-spine repos before live execution changes.
6. The local Mieza signal spine is committed, pushed, and redeployed only after local tests pass.

## Commands To Resume The Live Audit

Inspect the known projects from a linked context:

```powershell
railway whoami --json
railway status --json
railway logs --lines 200 --json
railway service status --all --json
```

Because most local repos are unlinked, use a temporary directory and `railway link --project <id> --environment <id>` for read-only status checks.
