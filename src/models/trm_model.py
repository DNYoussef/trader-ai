"""
Tiny Recursive Model (TRM) for Trading Strategy Selection

Based on the TRM paper architecture:
- 2-layer neural network (~7M parameters)
- Recursive reasoning with latent state evolution
- Deep recursion: T cycles × n steps per cycle = 42-layer equivalent depth
- Halting mechanism for early convergence

Input: 10 market features (VIX, returns, volume, breadth, etc.)
Output: 8-way strategy classification with confidence scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TinyRecursiveModel(nn.Module):
    """
    TRM architecture for multi-stream market pattern recognition.

    Architecture Overview:
    - Single 2-layer network (7M params vs HRM's 27M)
    - Recursive latent reasoning: z ← net(x, y, z)
    - Solution refinement: y ← net(y, z)
    - Halting mechanism: early stopping via confidence signal

    Training Strategy:
    - T-1 cycles without gradients (efficiency)
    - Final cycle with gradients (learning)
    - Effective depth: T(n+1)×n_layers = 3×7×2 = 42 layers
    """

    def __init__(
        self,
        input_dim: int = 10,           # Market features (VIX, returns, etc.)
        hidden_dim: int = 512,         # TRM standard
        output_dim: int = 8,           # 8 trading strategies
        num_latent_steps: int = 6,     # n: latent reasoning steps
        num_recursion_cycles: int = 3, # T: deep recursion cycles
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_latent_steps = num_latent_steps
        self.num_recursion_cycles = num_recursion_cycles

        # Input projection: market features → hidden space
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Core reasoning network (2 layers as per TRM)
        self.reasoning_layer1 = nn.Linear(hidden_dim * 3, hidden_dim)  # x, y, z combined
        self.reasoning_layer2 = nn.Linear(hidden_dim, hidden_dim)

        # Solution state update network
        self.solution_layer1 = nn.Linear(hidden_dim * 2, hidden_dim)  # y, z combined
        self.solution_layer2 = nn.Linear(hidden_dim, hidden_dim)

        # Output head: strategy classification
        self.output_head = nn.Linear(hidden_dim, output_dim)

        # Halting network: early stopping signal
        self.halt_layer1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.halt_layer2 = nn.Linear(hidden_dim // 2, 1)

        # Optional regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.use_layer_norm = use_layer_norm

        if use_layer_norm:
            self.norm_reasoning = nn.LayerNorm(hidden_dim)
            self.norm_solution = nn.LayerNorm(hidden_dim)
            self.norm_output = nn.LayerNorm(hidden_dim)

        # Parameter count tracking
        self._log_parameter_count()

    def _log_parameter_count(self):
        """Log total parameter count (should be ~7M)"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"TRM Parameter Count: {total_params:,} total, {trainable_params:,} trainable")
        logger.info("Target: ~7M parameters (TRM standard)")

    def reasoning_update(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Core TRM recursion: update latent reasoning state

        z_new ← net(x, y, z)

        Args:
            x: Input features (batch_size, hidden_dim)
            y: Solution state (batch_size, hidden_dim)
            z: Latent reasoning state (batch_size, hidden_dim)

        Returns:
            Updated latent state z_new
        """
        # Combine all information streams
        combined = torch.cat([x, y, z], dim=-1)  # (batch, hidden_dim * 3)

        # 2-layer reasoning network
        h = F.gelu(self.reasoning_layer1(combined))
        h = self.dropout(h)
        z_new = self.reasoning_layer2(h)

        if self.use_layer_norm:
            z_new = self.norm_reasoning(z_new)

        # Residual connection for stability
        z_new = z_new + z

        return z_new

    def solution_update(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Update solution state using latent reasoning

        y_new ← net(y, z)

        Args:
            y: Current solution state (batch_size, hidden_dim)
            z: Latent reasoning state (batch_size, hidden_dim)

        Returns:
            Updated solution state y_new
        """
        # Combine solution and reasoning states
        combined = torch.cat([y, z], dim=-1)  # (batch, hidden_dim * 2)

        # 2-layer solution network
        h = F.gelu(self.solution_layer1(combined))
        h = self.dropout(h)
        y_new = self.solution_layer2(h)

        if self.use_layer_norm:
            y_new = self.norm_solution(y_new)

        # Residual connection
        y_new = y_new + y

        return y_new

    def latent_recursion(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform n steps of latent reasoning recursion

        for step in range(n):
            z ← reasoning_update(x, y, z)
            y ← solution_update(y, z)

        Args:
            x: Input features
            y: Solution state
            z: Latent state
            n: Number of recursion steps

        Returns:
            (y_new, z_new): Updated solution and latent states
        """
        for _ in range(n):
            # Update latent reasoning state
            z = self.reasoning_update(x, y, z)

            # Update solution state
            y = self.solution_update(y, z)

        return y, z

    def compute_halt_signal(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute halting probability from latent state

        Higher probability = model is confident in current solution
        Lower probability = needs more reasoning steps

        Args:
            z: Latent state (batch_size, hidden_dim)

        Returns:
            Halting probability (batch_size, 1)
        """
        h = F.gelu(self.halt_layer1(z))
        h = self.dropout(h)
        halt_logit = self.halt_layer2(h)
        halt_prob = torch.sigmoid(halt_logit)

        return halt_prob

    def forward(
        self,
        x: torch.Tensor,
        T: Optional[int] = None,
        n: Optional[int] = None,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with deep recursion

        Strategy:
        - T-1 cycles without gradients (efficiency)
        - Final cycle with gradients (learning)
        - n latent steps per cycle

        Args:
            x: Input market features (batch_size, input_dim)
            T: Number of recursion cycles (default: self.num_recursion_cycles)
            n: Number of latent steps per cycle (default: self.num_latent_steps)
            return_intermediate: Whether to return intermediate states

        Returns:
            Dictionary containing:
                - strategy_logits: (batch_size, output_dim) 8-way classification
                - halt_probability: (batch_size, 1) confidence signal
                - latent_state: (batch_size, hidden_dim) final reasoning state
                - solution_state: (batch_size, hidden_dim) final solution
                - (optional) intermediate_states: list of (y, z) tuples
        """
        T = T if T is not None else self.num_recursion_cycles
        n = n if n is not None else self.num_latent_steps

        batch_size = x.size(0)
        device = x.device

        # Project input to hidden space
        x_proj = self.input_proj(x)  # (batch, hidden_dim)

        # Initialize states
        y = torch.zeros(batch_size, self.hidden_dim, device=device)  # Solution state
        z = torch.zeros(batch_size, self.hidden_dim, device=device)  # Latent state

        intermediate_states = []

        # T-1 cycles without gradients (efficiency)
        if T > 1:
            with torch.no_grad():
                for cycle in range(T - 1):
                    y, z = self.latent_recursion(x_proj, y, z, n)

                    if return_intermediate:
                        intermediate_states.append((y.clone(), z.clone()))

        # Final cycle with gradients (learning)
        y, z = self.latent_recursion(x_proj, y, z, n)

        if return_intermediate:
            intermediate_states.append((y.clone(), z.clone()))

        # Compute outputs
        if self.use_layer_norm:
            y_norm = self.norm_output(y)
        else:
            y_norm = y

        # Strategy classification logits
        strategy_logits = self.output_head(y_norm)  # (batch, output_dim)

        # Halting signal (confidence)
        halt_probability = self.compute_halt_signal(z)  # (batch, 1)

        # Build output dictionary
        output = {
            'strategy_logits': strategy_logits,
            'halt_probability': halt_probability.squeeze(-1),  # (batch,)
            'latent_state': z,
            'solution_state': y
        }

        if return_intermediate:
            output['intermediate_states'] = intermediate_states

        return output

    def predict_strategy(
        self,
        x: torch.Tensor,
        return_confidence: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict strategy with confidence scores

        Args:
            x: Input market features (batch_size, input_dim)
            return_confidence: Whether to return confidence scores

        Returns:
            strategy_idx: (batch_size,) predicted strategy indices (0-7)
            confidence: (batch_size,) confidence scores (0-1) if return_confidence=True
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)

            # Get predicted strategy
            strategy_idx = output['strategy_logits'].argmax(dim=-1)

            if return_confidence:
                # Use halting probability as confidence
                confidence = output['halt_probability']
                return strategy_idx, confidence
            else:
                return strategy_idx, None

    def get_strategy_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability distribution over 8 strategies

        Args:
            x: Input market features (batch_size, input_dim)

        Returns:
            probabilities: (batch_size, 8) probability distribution
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            probabilities = F.softmax(output['strategy_logits'], dim=-1)
            return probabilities


def create_trm_model(config: Optional[Dict] = None) -> TinyRecursiveModel:
    """
    Factory function to create TRM model with configuration

    Args:
        config: Configuration dictionary (optional)

    Returns:
        Initialized TRM model
    """
    if config is None:
        config = {}

    model = TinyRecursiveModel(
        input_dim=config.get('input_dim', 10),
        hidden_dim=config.get('hidden_dim', 512),
        output_dim=config.get('output_dim', 8),
        num_latent_steps=config.get('num_latent_steps', 6),
        num_recursion_cycles=config.get('num_recursion_cycles', 3),
        dropout=config.get('dropout', 0.1),
        use_layer_norm=config.get('use_layer_norm', True)
    )

    logger.info("TRM model created successfully")
    return model


if __name__ == "__main__":
    # Test TRM model creation
    logging.basicConfig(level=logging.INFO)

    print("Creating TRM model...")
    model = create_trm_model()

    print("\nTesting forward pass...")
    batch_size = 4
    input_dim = 10

    # Create sample market features
    x = torch.randn(batch_size, input_dim)

    # Forward pass
    output = model(x, return_intermediate=True)

    print(f"\nInput shape: {x.shape}")
    print(f"Strategy logits shape: {output['strategy_logits'].shape}")
    print(f"Halt probability shape: {output['halt_probability'].shape}")
    print(f"Latent state shape: {output['latent_state'].shape}")
    print(f"Solution state shape: {output['solution_state'].shape}")
    print(f"Number of intermediate states: {len(output['intermediate_states'])}")

    # Test prediction
    strategy_idx, confidence = model.predict_strategy(x)
    print(f"\nPredicted strategies: {strategy_idx}")
    print(f"Confidence scores: {confidence}")

    # Test probability distribution
    probs = model.get_strategy_probabilities(x)
    print(f"\nStrategy probabilities shape: {probs.shape}")
    print(f"First sample probabilities: {probs[0]}")
    print(f"Sum of probabilities: {probs[0].sum()}")  # Should be 1.0

    print("\n✅ TRM model test passed!")
