# Trader AI Closed-Loop Architecture Plan

## Executive Summary

This document outlines the complete architecture for closed-loop AI training in the Trader AI system, addressing model persistence on Railway, automated retraining, and feedback-driven learning.

---

## Phase 3 Analysis: Current TRM Architecture

### Model Architecture
- **Model**: TinyRecursiveModel (7M parameters)
- **Structure**: T=3 cycles, n=6 steps, hidden_dim=512
- **Effective depth**: 42 layers equivalent (T x (n+1) x 2)
- **Checkpoints**: 31 MB each (.pt format)

### Training Pipeline
| Component | File | Status |
|-----------|------|--------|
| Data Loader | `src/training/trm_data_loader.py` | Operational |
| Model | `src/models/trm_model.py` | Operational |
| Trainer | `src/training/trm_trainer.py` | Operational |
| Loss Functions | `src/training/trm_loss_functions.py` | Operational |

### Inference Pipeline
| Component | File | Status |
|-----------|------|--------|
| Streaming Predictor | `src/intelligence/trm_streaming_integration.py` | Operational |
| WebSocket Broadcaster | `src/dashboard/server/trm_websocket_integration.py` | Operational |

### Feedback System (Partial)
| Component | Status |
|-----------|--------|
| `PerformanceFeedback.record_feedback()` | Implemented |
| `PerformanceFeedback._process_recent_feedback()` | Implemented |
| `PerformanceFeedback._generate_feedback_signals()` | Implemented |
| Signal action handlers | NOT IMPLEMENTED |
| Automated retraining | NOT IMPLEMENTED |

---

## Phase 4: Closed-Loop Model Storage Design

### Problem Statement
1. Railway deployments are ephemeral - checkpoints deleted on pod restart
2. Model checkpoints (31 MB) too large for git commit
3. No versioning of trained models
4. No mechanism to update production model from training results

### Solution Architecture

```
                    TRAINING ENVIRONMENT
                    (Local / GitHub Actions)
                           |
                           v
+----------------------------------------------------------+
|                    MODEL REGISTRY                         |
|                                                          |
|  S3 Bucket: trader-ai-models/                            |
|  +-------------------+  +-------------------+            |
|  | v1.0/             |  | v1.1/             |            |
|  | - best_model.pt   |  | - best_model.pt   |            |
|  | - metrics.json    |  | - metrics.json    |            |
|  | - config.json     |  | - config.json     |            |
|  +-------------------+  +-------------------+            |
|                                                          |
|  MANIFEST.json                                           |
|  {                                                       |
|    "current": "v1.1",                                    |
|    "versions": {                                         |
|      "v1.0": {"accuracy": 0.5645, "date": "2025-12-10"}, |
|      "v1.1": {"accuracy": 0.5698, "date": "2025-12-16"}  |
|    }                                                     |
|  }                                                       |
+----------------------------------------------------------+
                           |
                           v
+----------------------------------------------------------+
|                  RAILWAY DEPLOYMENT                       |
|                                                          |
|  1. On startup:                                          |
|     - Check /mnt/cache/checkpoints/                      |
|     - If missing, fetch from S3 using MODEL_VERSION env  |
|     - Load into TRMStreamingPredictor                    |
|                                                          |
|  2. During operation:                                    |
|     - Predictions via WebSocket (60s interval)           |
|     - Trade results stored in PostgreSQL                 |
|     - Feedback system writes to performance_feedback.db  |
|                                                          |
|  3. On signal 'retrain':                                 |
|     - Trigger GitHub Actions workflow                    |
|     - Wait for new model version                         |
|     - Hot-reload or rolling restart                      |
+----------------------------------------------------------+
                           |
                           v
+----------------------------------------------------------+
|                  FEEDBACK DATABASE                        |
|                                                          |
|  PostgreSQL (Railway)                                    |
|  - trades: symbol, entry, exit, pnl, strategy_id        |
|  - predictions: timestamp, strategy_id, confidence       |
|  - feedback_metrics: error, direction_accuracy           |
|  - feedback_signals: signal_type, urgency, metadata      |
+----------------------------------------------------------+
```

### Storage Implementation

```python
# src/models/model_registry.py

import os
import boto3
import json
from pathlib import Path

class ModelRegistry:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        self.bucket = os.getenv('MODEL_S3_BUCKET', 'trader-ai-models')
        self.cache_dir = Path(os.getenv('CACHE_DIR', '/tmp/checkpoints'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_current_version(self) -> str:
        """Get current production model version from S3 manifest"""
        manifest = self._fetch_manifest()
        return manifest.get('current', 'v1.0')

    def download_model(self, version: str = None) -> Path:
        """Download model checkpoint to local cache"""
        version = version or os.getenv('MODEL_VERSION') or self.get_current_version()
        local_path = self.cache_dir / f'{version}_best_model.pt'

        if not local_path.exists():
            s3_key = f'{version}/best_model.pt'
            self.s3.download_file(self.bucket, s3_key, str(local_path))

        return local_path

    def upload_model(self, local_path: Path, version: str, metrics: dict) -> str:
        """Upload new model version to S3"""
        # Upload checkpoint
        s3_key = f'{version}/best_model.pt'
        self.s3.upload_file(str(local_path), self.bucket, s3_key)

        # Upload metrics
        metrics_key = f'{version}/metrics.json'
        self.s3.put_object(
            Bucket=self.bucket,
            Key=metrics_key,
            Body=json.dumps(metrics).encode()
        )

        # Update manifest
        self._update_manifest(version, metrics)

        return s3_key

    def _fetch_manifest(self) -> dict:
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key='MANIFEST.json')
            return json.loads(response['Body'].read().decode())
        except self.s3.exceptions.NoSuchKey:
            return {'current': 'v1.0', 'versions': {}}

    def _update_manifest(self, version: str, metrics: dict):
        manifest = self._fetch_manifest()
        manifest['versions'][version] = {
            'accuracy': metrics.get('accuracy', 0),
            'date': metrics.get('date', ''),
            'commit': metrics.get('commit', '')
        }
        manifest['current'] = version

        self.s3.put_object(
            Bucket=self.bucket,
            Key='MANIFEST.json',
            Body=json.dumps(manifest, indent=2).encode()
        )
```

### Environment Variables for Railway

```env
# Railway Service Variables
MODEL_VERSION=v1.1
MODEL_S3_BUCKET=trader-ai-models
AWS_ACCESS_KEY_ID=<secret>
AWS_SECRET_ACCESS_KEY=<secret>
AWS_REGION=us-east-1
CACHE_DIR=/mnt/cache/checkpoints

# Database
DATABASE_URL=postgresql://...

# Feature Flags
ENABLE_CLOSED_LOOP=true
RETRAIN_THRESHOLD_ACCURACY=0.40
RETRAIN_THRESHOLD_DRAWDOWN=0.10
```

---

## Phase 5: Implementation Plan

### Step 1: Create S3 Bucket and Upload Current Model

```bash
# Create bucket
aws s3 mb s3://trader-ai-models --region us-east-1

# Upload current best model
aws s3 cp checkpoints/best_model_optimized.pt s3://trader-ai-models/v1.0/best_model.pt
aws s3 cp checkpoints/final_metrics.json s3://trader-ai-models/v1.0/metrics.json

# Create manifest
echo '{"current": "v1.0", "versions": {"v1.0": {"accuracy": 0.5698, "date": "2025-12-16"}}}' \
  | aws s3 cp - s3://trader-ai-models/MANIFEST.json
```

### Step 2: Integrate Model Registry into TRMStreamingPredictor

Modify `src/intelligence/trm_streaming_integration.py`:

```python
from src.models.model_registry import ModelRegistry

class TRMStreamingPredictor:
    def __init__(self, model_path: str = None):
        self.registry = ModelRegistry()

        if model_path is None:
            # Fetch from S3 if not provided
            model_path = str(self.registry.download_model())

        self._load_model(model_path)
```

### Step 3: Add Signal Action Handlers

Create `src/learning/signal_handlers.py`:

```python
class SignalHandler:
    def __init__(self):
        self.feedback = PerformanceFeedback()
        self.registry = ModelRegistry()
        self.trainer = TRMTrainer()

    def handle_signal(self, signal: FeedbackSignal):
        if signal.signal_type == 'retrain':
            return self._trigger_retrain(signal)
        elif signal.signal_type == 'rollback':
            return self._rollback_model(signal)
        elif signal.signal_type == 'adjust_parameters':
            return self._adjust_parameters(signal)

    def _trigger_retrain(self, signal):
        # Option 1: Trigger GitHub Actions via API
        # Option 2: Run training in background worker
        pass

    def _rollback_model(self, signal):
        # Get previous version from manifest
        # Update MODEL_VERSION env
        # Restart service or hot-reload
        pass
```

### Step 4: GitHub Actions Workflow for Retraining

Create `.github/workflows/retrain-model.yml`:

```yaml
name: Retrain TRM Model

on:
  workflow_dispatch:
    inputs:
      trigger_reason:
        description: 'Reason for retraining'
        required: true
        default: 'scheduled'
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday

jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Download training data from S3
        run: |
          aws s3 sync s3://trader-ai-training-data/ data/training/
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

      - name: Run training
        run: python scripts/training/train_regime_aware_trm.py --epochs 30

      - name: Evaluate model
        id: evaluate
        run: python scripts/evaluate_model.py --compare-current

      - name: Upload if improved
        if: steps.evaluate.outputs.improved == 'true'
        run: |
          VERSION="v$(date +%Y%m%d)"
          aws s3 cp checkpoints/best_model.pt s3://trader-ai-models/$VERSION/best_model.pt
          # Update Railway env via API
```

### Step 5: Hot-Reload Mechanism

```python
# src/dashboard/server/model_updater.py

import asyncio
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ModelReloader:
    def __init__(self, predictor: TRMStreamingPredictor):
        self.predictor = predictor
        self.observer = Observer()

    async def watch_for_updates(self):
        """Watch S3 manifest for version changes"""
        current_version = self.predictor.registry.get_current_version()

        while True:
            await asyncio.sleep(60)  # Check every minute
            new_version = self.predictor.registry.get_current_version()

            if new_version != current_version:
                logger.info(f"New model version detected: {new_version}")
                await self._reload_model(new_version)
                current_version = new_version

    async def _reload_model(self, version: str):
        """Hot-reload model without service restart"""
        model_path = self.predictor.registry.download_model(version)
        self.predictor._load_model(str(model_path))
        logger.info(f"Model reloaded to version {version}")
```

---

## Timeline and Priorities

| Priority | Task | Effort | Status |
|----------|------|--------|--------|
| P0 | Create S3 bucket, upload v1.0 | 1h | Pending |
| P0 | Implement ModelRegistry class | 2h | Pending |
| P1 | Integrate registry into predictor | 1h | Pending |
| P1 | Add Railway env variables | 30m | Pending |
| P2 | Implement signal handlers | 4h | Pending |
| P2 | Create GitHub Actions workflow | 3h | Pending |
| P3 | Add hot-reload mechanism | 2h | Pending |
| P3 | Set up monitoring dashboard | 4h | Pending |

---

## Success Criteria

1. **Model Persistence**: Checkpoints survive Railway deployment restarts
2. **Version Control**: Every model version tagged and tracked in S3
3. **Closed-Loop**: Trading results automatically improve model over time
4. **Zero Downtime**: Model updates don't interrupt predictions
5. **Rollback**: Can revert to any previous model version in <1 minute

---

## Next Steps

1. Create AWS S3 bucket for model storage
2. Upload current best model as v1.0
3. Implement ModelRegistry class
4. Update Railway environment variables
5. Test full cycle: train -> upload -> deploy -> predict

