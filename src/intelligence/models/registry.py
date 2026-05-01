"""
Model registry for managing trained models
Production-ready model versioning and storage
"""

import pickle
import base64
import hashlib
import hmac
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch

logger = logging.getLogger(__name__)

_PICKLE_SIGNING_KEY_ENV = "TRADER_AI_PICKLE_SIGNING_KEY"
_SIGNED_PICKLE_MAGIC = "TRADER_AI_SIGNED_PICKLE_V1"


def _pickle_signing_key() -> bytes:
    key = os.environ.get(_PICKLE_SIGNING_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{_PICKLE_SIGNING_KEY_ENV} must be set to load or save signed pickle artifacts"
        )
    return key.encode("utf-8")


def _save_signed_pickle(path: Path, data: Any) -> None:
    payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    signature = hmac.new(_pickle_signing_key(), payload, hashlib.sha256).hexdigest()
    envelope = {
        "magic": _SIGNED_PICKLE_MAGIC,
        "algorithm": "HMAC-SHA256",
        "payload": base64.b64encode(payload).decode("ascii"),
        "signature": signature,
    }
    path.write_bytes(json.dumps(envelope, separators=(",", ":")).encode("utf-8"))


def _load_signed_pickle(path: Path) -> Any:
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Refusing unsigned pickle artifact: {path}") from exc

    if not isinstance(envelope, dict) or envelope.get("magic") != _SIGNED_PICKLE_MAGIC:
        raise ValueError(f"Refusing unsigned pickle artifact: {path}")
    if envelope.get("algorithm") != "HMAC-SHA256":
        raise ValueError(f"Unsupported pickle signature algorithm: {envelope.get('algorithm')}")

    payload = base64.b64decode(envelope["payload"])
    expected_signature = hmac.new(_pickle_signing_key(), payload, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(envelope.get("signature", ""), expected_signature):
        raise ValueError(f"Invalid pickle artifact signature: {path}")

    return pickle.loads(payload)  # nosec B301 - HMAC signature verified before deserialization

class ModelRegistry:
    """
    Production model registry with versioning and metadata
    """

    def __init__(self, registry_path: str = "/c/Users/17175/Desktop/trader-ai/trained_models"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        # Initialize registry metadata
        self.metadata_file = self.registry_path / "registry_metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load registry metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "models": {},
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }

    def _save_metadata(self):
        """Save registry metadata"""
        self.metadata["last_updated"] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def register_model(self, model_name: str, model: Any,
                      metrics: Optional[Dict[str, float]] = None,
                      parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Register a trained model

        Args:
            model_name: Name of the model
            model: Trained model object
            metrics: Performance metrics
            parameters: Training parameters

        Returns:
            Model version string
        """
        # Generate version
        if model_name not in self.metadata["models"]:
            self.metadata["models"][model_name] = {"versions": []}

        version = len(self.metadata["models"][model_name]["versions"]) + 1
        version_str = f"v{version}"

        # Create version directory
        version_dir = self.registry_path / model_name / version_str
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = version_dir / "model"
        if hasattr(model, 'state_dict'):  # PyTorch model
            torch.save(model.state_dict(), f"{model_path}.pth")
        else:  # Scikit-learn or other models
            _save_signed_pickle(Path(f"{model_path}.pkl"), model)

        # Save model metadata
        model_metadata = {
            "model_name": model_name,
            "version": version_str,
            "created_at": datetime.now().isoformat(),
            "model_type": type(model).__name__,
            "metrics": metrics or {},
            "parameters": parameters or {},
            "file_path": str(model_path)
        }

        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)

        # Update registry metadata
        self.metadata["models"][model_name]["versions"].append(model_metadata)
        self.metadata["models"][model_name]["latest_version"] = version_str
        self._save_metadata()

        logger.info(f"Model {model_name} {version_str} registered successfully")
        return version_str

    def register_models(self, models: Dict[str, Any]) -> Dict[str, str]:
        """Register multiple models"""
        versions = {}
        for name, model in models.items():
            if model is not None:
                version = self.register_model(name, model)
                versions[name] = version
        return versions

    def load_model(self, model_name: str, version: Optional[str] = None) -> Any:
        """
        Load a registered model

        Args:
            model_name: Name of the model
            version: Model version (latest if None)

        Returns:
            Loaded model object
        """
        if model_name not in self.metadata["models"]:
            raise ValueError(f"Model {model_name} not found in registry")

        if version is None:
            version = self.metadata["models"][model_name]["latest_version"]

        version_dir = self.registry_path / model_name / version

        # Find model file
        model_files = list(version_dir.glob("model.*"))
        if not model_files:
            raise FileNotFoundError(f"Model file not found for {model_name} {version}")

        model_file = model_files[0]

        # Load based on file extension
        if model_file.suffix == '.pth':
            # Load state dict only (requires model architecture)
            torch.load(model_file, map_location='cpu', weights_only=True)
            # Note: This requires the model architecture to be available
            raise NotImplementedError("Loading state dict requires model architecture")
        elif model_file.suffix == '.pkl':
            # Scikit-learn or other pickle-able models
            model = _load_signed_pickle(model_file)
        else:
            raise ValueError(f"Unsupported model file format: {model_file.suffix}")

        logger.info(f"Model {model_name} {version} loaded successfully")
        return model

    def list_models(self) -> Dict[str, Any]:
        """List all registered models"""
        return self.metadata["models"]

    def get_model_info(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Get model information"""
        if model_name not in self.metadata["models"]:
            raise ValueError(f"Model {model_name} not found in registry")

        if version is None:
            version = self.metadata["models"][model_name]["latest_version"]

        for version_info in self.metadata["models"][model_name]["versions"]:
            if version_info["version"] == version:
                return version_info

        raise ValueError(f"Version {version} not found for model {model_name}")

    def delete_model(self, model_name: str, version: Optional[str] = None):
        """Delete a model version"""
        if model_name not in self.metadata["models"]:
            raise ValueError(f"Model {model_name} not found in registry")

        if version is None:
            # Delete entire model
            import shutil
            model_dir = self.registry_path / model_name
            if model_dir.exists():
                shutil.rmtree(model_dir)
            del self.metadata["models"][model_name]
        else:
            # Delete specific version
            version_dir = self.registry_path / model_name / version
            if version_dir.exists():
                import shutil
                shutil.rmtree(version_dir)

            # Update metadata
            versions = self.metadata["models"][model_name]["versions"]
            self.metadata["models"][model_name]["versions"] = [
                v for v in versions if v["version"] != version
            ]

            # Update latest version
            remaining_versions = self.metadata["models"][model_name]["versions"]
            if remaining_versions:
                latest = max(remaining_versions, key=lambda x: x["created_at"])
                self.metadata["models"][model_name]["latest_version"] = latest["version"]
            else:
                del self.metadata["models"][model_name]

        self._save_metadata()
        logger.info(f"Model {model_name} {version or 'all versions'} deleted")

    def compare_models(self, model_name: str, metric: str = "mse") -> List[Dict[str, Any]]:
        """Compare different versions of a model"""
        if model_name not in self.metadata["models"]:
            raise ValueError(f"Model {model_name} not found in registry")

        versions = self.metadata["models"][model_name]["versions"]
        comparisons = []

        for version_info in versions:
            if metric in version_info.get("metrics", {}):
                comparisons.append({
                    "version": version_info["version"],
                    "metric_value": version_info["metrics"][metric],
                    "created_at": version_info["created_at"]
                })

        # Sort by metric value (ascending for error metrics)
        comparisons.sort(key=lambda x: x["metric_value"])
        return comparisons

    def export_model(self, model_name: str, version: Optional[str] = None,
                    export_path: Optional[str] = None) -> str:
        """Export model for deployment"""
        model = self.load_model(model_name, version)
        model_info = self.get_model_info(model_name, version)

        if export_path is None:
            export_path = f"{model_name}_{model_info['version']}_export"

        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)

        # Export model
        if hasattr(model, 'state_dict'):  # PyTorch
            torch.save(model, export_dir / "model.pth")
        else:  # Scikit-learn
            _save_signed_pickle(export_dir / "model.pkl", model)

        # Export metadata
        with open(export_dir / "model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)

        logger.info(f"Model exported to {export_path}")
        return str(export_dir)

    def get_production_model(self, model_name: str) -> Any:
        """Get the production-ready version of a model"""
        # For now, return the latest version
        # In a real system, this would be based on production tags
        return self.load_model(model_name)
