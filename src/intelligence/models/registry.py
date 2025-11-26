"""
Model registry for managing trained models
Production-ready model versioning and storage
"""

import pickle
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch

logger = logging.getLogger(__name__)

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
            # Also save the model architecture
            torch.save(model, f"{model_path}_full.pth")
        else:  # Scikit-learn or other models
            with open(f"{model_path}.pkl", 'wb') as f:
                pickle.dump(model, f)

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
            # PyTorch model - try to load full model first
            full_model_file = version_dir / "model_full.pth"
            if full_model_file.exists():
                model = torch.load(full_model_file, map_location='cpu')
            else:
                # Load state dict only (requires model architecture)
                torch.load(model_file, map_location='cpu')
                # Note: This requires the model architecture to be available
                raise NotImplementedError("Loading state dict requires model architecture")
        elif model_file.suffix == '.pkl':
            # Scikit-learn or other pickle-able models
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
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
            with open(export_dir / "model.pkl", 'wb') as f:
                pickle.dump(model, f)

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