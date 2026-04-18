# Muon Optimizer: Theoretical Foundations and Training Dynamics

**Research Date:** 2025-12-16
**Focus:** Newton-Schulz Orthogonalization, Training Signatures, and Optimizer Interactions

---

## Executive Summary

Muon (MomentUm Orthogonalized by Newton-Schulz) is a modern optimizer for neural network hidden layers that achieves 2x efficiency gains over AdamW through gradient orthogonalization. It combines SGD momentum with Newton-Schulz iterations to approximate polar decomposition, effectively implementing steepest descent under spectral norm constraints. This research synthesizes theoretical foundations, expected training behaviors, failure modes, and interaction patterns critical for understanding Muon's role in training dynamics.

**Key Finding for Grokking:** Muon significantly accelerates grokking, reducing mean grokking epoch from 153.09 to 102.89 (p = 6.33e-08), suggesting orthogonalization helps models transition from memorization to generalization faster.

---

## 1. Core Mechanism: Newton-Schulz Orthogonalization

### 1.1 Mathematical Foundation

The Newton-Schulz iteration computes the orthogonal polar factor Q of a matrix A through iterative refinement:

**X_{k+1} = (1/2) X_k (3I - X_k^T X_k)**

Starting with **X_0 = A / ||A||** (normalized by spectral or Frobenius norm).

**Convergence Properties:**
- **Quadratic convergence:** Number of correct digits doubles each iteration
- **Convergence condition:** Singular values of X_0 must lie in (0, sqrt(3))
- **Typical iterations:** 5 steps suffice for neural network training (N=5)
- **Precision:** Stable in bfloat16, enabling efficient GPU computation

**Optimized Coefficients:**
Recent research identified optimal Newton-Schulz coefficients for stability:
- Standard: (0.5, 3.0) in X_{k+1} = a * X_k * (b*I - X_k^T X_k)
- Optimized: (3.4445, -4.7750, 2.0315) for improved convergence
- These provide 1-2% efficiency gains with better stability

### 1.2 Muon's Application to Gradient Updates

**Algorithm:**
1. Compute momentum update: **m_t = beta * m_{t-1} + (1-beta) * grad**
2. Normalize: **M = m_t / ||m_t||_rms**
3. Apply Newton-Schulz orthogonalization (5 iterations): **Q = NS(M)**
4. Update parameters: **W_{t+1} = W_t - lr * Q - wd * W_t**

**Key Innovation:** Muon moves momentum BEFORE orthogonalization (unlike Orthogonal-SGDM which applies it after), yielding better empirical performance.

### 1.3 Why Orthogonalize Gradients?

**Theoretical Benefits:**
1. **Spectral homogenization:** Makes singular values more uniform, improving conditioning
2. **Rare direction amplification:** Small-magnitude but important gradient components gain influence
3. **Gradient explosion prevention:** Spectral norm constraint bounds update magnitudes
4. **Implicit regularization:** Enforces updates to lie on spectral norm-ball

**Geometric Interpretation:**
- Preserves gradient directions (via U and V matrices in SVD)
- Standardizes all singular values to 1
- Creates unit-norm updates in spectral domain
- Emphasizes geometric structure over raw gradient magnitudes

---

## 2. Theoretical Connections

### 2.1 Relation to Natural Gradient Descent

Muon exhibits a **connection to natural gradient descent on the Stiefel manifold**:
- Stiefel manifold: Space of orthonormal matrices
- Muon enforces spectral normalization, similar to Riemannian optimization
- Equivalence to **steepest gradient descent under spectral norm**
- Recent work ("Riemannion") generalizes Muon to fixed-rank matrix manifolds

### 2.2 Second-Order Optimization Properties

**Connections:**
- Muon approximates curvature information through orthogonalization
- Under isotropic curvature model, orthogonalized gradients become optimal when curvature exhibits specific asymptotic behavior
- Unlike full second-order methods (BFGS), Muon uses only first-order gradients but implicitly captures curvature via spectral structure
- Computational complexity: O(matrix multiplication) vs O(D^3) for Hessian inversion

**Implicit Curvature Adaptation:**
- Orthogonalization makes spectrum more homogeneous
- Under general curvature growth conditions, this improves conditioning of update matrix
- Enhances updates along directions of negative curvature, facilitating escape from sharp minima

### 2.3 Loss Landscape Effects

**Observed Behaviors:**
1. **Faster convergence:** More efficient loss landscape traversal than AdamW
2. **Sharper minima escape:** Better projection along negative curvature directions
3. **Spectral norm constraints:** Keeps parameters on bounded manifold, preventing divergence
4. **Non-linearized regime operation:** Unlike NTK-regime methods, Muon allows weights to move substantially from initialization

---

## 3. Expected Training Signatures

### 3.1 Convergence Behavior

**Typical Patterns:**
- **Early stage:** Faster initial convergence than AdamW (first 20-30% of training)
- **Mid-stage:** AdamW may catch up if weight decay is insufficient
- **Final stage:** Muon achieves lower final loss with proper configuration
- **Learning rate schedule:** Linear warmup + cosine decay to 0.1 of max LR works well

**Quantitative Benchmarks:**
- 2x efficiency improvement over AdamW at compute-optimal scales
- Example: 1.5B parameter transformer reaches GPT-2 XL level in 10.0 vs 13.3 hours (AdamW)
- Scales to 1 trillion parameters (Kimi K2, GLM4.5)

### 3.2 Gradient Norm Dynamics

**Expected Behaviors:**
1. **Bounded gradient norms:** Weight decay + orthogonalization prevent explosion
2. **Tighter bounds with weight decay:** Theory proves incorporating WD yields tighter gradient norm bounds
3. **Stable throughout training:** Unlike AdamW, gradient norms remain well-controlled
4. **Spectral norm correlation:** Lower spectral norms with higher weight decay values (e.g., 0.1)

**Observable Metrics:**
- Average expected gradient norm decreases faster with Nesterov momentum + weight decay
- Empirically: Muon+Nesterov+WD attains fastest loss and gradient norm reduction

### 3.3 Weight Matrix Conditioning

**Spectral Properties:**
- **Condition number improvement:** Orthogonalization moves toward condition number = 1
- **Singular value homogenization:** Ratio between largest/smallest singular values decreases
- **Spectral norm tracking:** Monitor ||W||_2 across layers
- **RMS growth patterns:** Both weight and layer output RMS grow but can exceed bf16 precision at large scales

**Layer-wise Observations:**
- LM head spectral norms decrease with higher weight decay
- Tensor parallel splits in MuonSBW lead to higher spectral norms across layers
- Attention logit maximums may exceed threshold (>100) in specific layers during early training

### 3.4 Learning Rate Characteristics

**Optimal Ranges:**
- **muP scaling:** Learning rates transfer across model widths (validated up to 3.7B parameters)
- **Direct AdamW transfer:** With RMS scaling adjustment, Muon can reuse AdamW-tuned LR and WD
- **Higher than AdamW:** Spectral norm constraints enable larger learning rates safely
- **Only 2 hyperparameters:** LR and WD require tuning; momentum (0.95) and ns_steps (5) use defaults

---

## 4. Observable Signatures of Correct Operation

### 4.1 Healthy Training Indicators

**Positive Signs:**
1. **Faster early convergence:** 30-50% faster loss reduction in first training phase
2. **Stable gradient norms:** No spikes or explosions throughout training
3. **Bounded spectral norms:** Weight matrix spectral norms remain controlled
4. **Lower final loss:** Achieves better validation loss than AdamW baseline
5. **Smooth loss curves:** No sudden instabilities or divergence

### 4.2 Numerical Stability Checks

**Monitor These Metrics:**
- **Newton-Schulz convergence:** 5 iterations should achieve <1% error
- **Singular value spread:** Condition number κ(moment matrix) should be reasonable
- **bf16 range compliance:** Weight/activation RMS should stay within bf16 high-precision range
- **Parameter norm growth:** Should grow gradually, not explosively
- **Attention logit maximums:** Should stay reasonable (<100 threshold recommended)

### 4.3 Muon-Specific Diagnostics

**Implementation Checks:**
1. **2D parameter filtering:** Only matrices/conv filters get Muon; scalars/vectors use AdamW
2. **Input/output exclusion:** Embeddings and classifier heads should use AdamW
3. **Weight decay active:** Muon requires WD to prevent instability
4. **RMS normalization:** Ensure proper RMS-to-RMS operator norm scaling
5. **Memory reduction:** Should see ~33% less optimizer state memory vs AdamW

---

## 5. Failure Modes and Instability Patterns

### 5.1 Weight Decay Omission

**Problem:** Original Muon lacked weight decay; implementations without it show instability.

**Symptoms:**
- Fast early convergence followed by AdamW catching up
- "Internal metrics" show instability signs
- Loss curves become erratic
- Parameter norms grow unbounded

**Solution:** Always include decoupled weight decay (typically 0.1 for large-scale training)

**Theoretical Basis:** Weight decay yields tighter bounds on parameter and gradient norms; Muon requires LR constraints: **lr < threshold dependent on WD**

### 5.2 Newton-Schulz Convergence Failure

**Causes:**
1. **Ill-conditioned moment matrices:** High condition number κ(M) increases NS error
2. **Insufficient iterations:** <5 steps may not achieve adequate orthogonalization
3. **Numerical precision issues:** Exceeding bf16 range causes instability
4. **Poor initialization:** Singular values outside convergence domain (0, sqrt(3))

**Symptoms:**
- High variance in resulting singular values (too much noise)
- Unstable training in later stages
- Performance degradation at scale
- Sensitivity to other hyperparameters

**Mitigation:**
- Use optimized NS coefficients (3.4445, -4.7750, 2.0315)
- Increase NS iterations for critical layers (though 5 typically suffices)
- Apply spectral scaling to initial guess
- Switch to SVD-based method for very ill-conditioned cases

### 5.3 Scaling Instabilities

**Large Model Issues:**
- Weight and layer output RMS grow to large scale, exceeding bf16 high-precision range
- Performance gains diminish when scaling to very large models (20B+ parameters, 1T+ tokens)
- Distribution challenges: Newton-Schulz iterations across large GPU clusters not yet solved

**Block-wise Orthogonalization Solution:**
- Split weight matrices into independent tiles
- Orthogonalize separately and recombine
- Allows up to 16x tensor parallel splits
- Retains validation loss while improving scalability

### 5.4 Optimizer Mismatch Problem

**Issue:** Models pretrained with AdamW perform suboptimally when fine-tuned with Muon, and vice versa.

**Impact:**
- Significant barrier to leveraging AdamW-pretrained checkpoints
- Prevents drop-in replacement in existing workflows
- Requires training from scratch or careful transition protocols

**Workaround:** Use consistent optimizer throughout pretraining and fine-tuning phases.

### 5.5 Hyperparameter Misconfiguration

**Common Mistakes:**
1. **Using Muon on scalar/vector parameters:** Only apply to 2D matrices
2. **Optimizing embeddings/heads with Muon:** These should use AdamW
3. **Too-low weight decay:** Use 0.1 for large-scale training, not 0.01
4. **Clipping WD too early:** Clip at ~80% of training, not earlier
5. **Wrong momentum value:** 0.95 is optimal; tuning often doesn't help

---

## 6. Interactions with Other Components

### 6.1 Weight Decay Interaction

**Critical Relationship:**
- **Muon requires weight decay** to remain stable
- **Decoupled implementation:** WD applied separately from optimization step
- **Scaling recommendations:**
  - Small models: 0.01-0.05
  - Large models (>1B): 0.1 for most of training
  - Clipping: Reduce WD at 80% of training to improve final validation loss
- **Theoretical justification:** WD yields tighter parameter/gradient norm bounds

**Coupled vs Decoupled:**
- Coupled: WD applied within momentum term (not recommended)
- Decoupled: WD applied directly to parameters (preferred, enables better regularization)

### 6.2 Bigeometric Gradient Transforms

**Potential Synergies:**
- Both methods modify gradient geometry before applying updates
- Bigeometric: Applies coordinate transformations (linear blending)
- Muon: Applies orthogonalization (spectral transformation)

**Interaction Considerations:**
1. **Sequential application:** Unclear whether bigeometric should precede or follow orthogonalization
2. **Coordinate system dependence:** Orthogonalization is coordinate-independent; bigeometric is coordinate-dependent
3. **Spectral effects:** Bigeometric may change singular value distribution, affecting NS convergence
4. **No published research:** Combination not yet studied in literature

**Recommendation:** Test bigeometric transforms independently before combining with Muon; monitor NS convergence quality if combining.

### 6.3 Momentum Variants

**Nesterov Momentum:**
- **Default:** Nesterov=True in Muon implementations
- **Performance:** Empirically achieves fastest gradient norm reduction
- **Theory:** Provides better gradient estimate through lookahead mechanism

**Momentum Coefficient:**
- **Standard:** beta=0.95 (default)
- **Tuning:** Research shows no consistent gain from tuning momentum
- **Stability:** Higher momentum (>0.95) may increase NS error accumulation

### 6.4 Adaptive Variants

**AdaMuon:**
- Augments Muon with second-moment estimator and RMS-based rescaling
- Enables adaptive downweighting of high-variance coordinates
- Lacks formal convergence guarantees
- Provides better stability across heterogeneous parameter structures

**COSMOS:**
- Integrates SOAP and Muon ideas
- Demonstrates stability and memory advantages
- Also lacks convergence proof
- Suitable for complex architectures

### 6.5 Precision and Quantization

**Mixed Precision:**
- NS iterations stable in bfloat16 (major advantage over SVD)
- Watch for RMS growth exceeding bf16 high-precision range at scale
- FP32 master weights recommended for very large models

**Optimizer State Quantization:**
- Muon reduces optimizer states (no second moment)
- Further quantization possible for memory-constrained settings
- Research: "Effective Quantization of Muon Optimizer States"

---

## 7. Grokking and Generalization Dynamics

### 7.1 Grokking Acceleration

**Key Research:** "Muon Optimizer Accelerates Grokking" (2025-04-22)

**Main Findings:**
- **Mean grokking epoch:** 153.09 (AdamW) → 102.89 (Muon)
- **Statistical significance:** t = 5.0175, p = 6.33e-08
- **Effect size:** ~33% reduction in epochs to generalization
- **Consistency:** Across 7 algorithmic tasks (modular arithmetic, parity classification)

### 7.2 Mechanistic Explanation

**Why Muon Helps Grokking:**
1. **Spectral norm constraints:** Steer away from simple memorization pathways
2. **Second-order cues:** More informative updates discover true patterns faster
3. **Broader exploration:** Orthogonalization prevents getting stuck in memorization modes
4. **Layer synchronization:** Updates coordinated across layers via magnitude control
5. **Stability against softmax collapse:** Maintains representation diversity

**Memorization vs Generalization:**
- **Early training:** Muon still memorizes but explores more broadly
- **Transition phase:** Spectral constraints enable faster discovery of generalizing solutions
- **Final stage:** Better generalization due to implicit spectral regularization

### 7.3 Implications for Trader-AI

**Potential Benefits:**
1. **Faster convergence:** Reduced training time to achieve target generalization
2. **Better sample efficiency:** Learn true patterns with fewer examples
3. **Reduced overfitting:** Spectral regularization discourages memorization
4. **Improved out-of-distribution performance:** Orthogonalization may help discover more robust features

**Cautions:**
1. **Financial data characteristics:** Grokking studied on algorithmic tasks; market data is non-stationary
2. **Scale differences:** Research used smaller models; trader-ai uses larger transformers
3. **Optimizer mismatch:** Cannot easily fine-tune AdamW checkpoints with Muon
4. **Hyperparameter sensitivity:** Requires careful tuning of LR and WD

---

## 8. Practical Implementation Guidelines

### 8.1 Which Parameters to Optimize

**Use Muon For:**
- Hidden layer weight matrices (2D)
- Convolutional filters (flattened to 2D)
- Transformer attention/FFN weight matrices

**Use AdamW For:**
- Scalar parameters (biases, LayerNorm gains)
- Vector parameters
- Embedding layers
- Classifier heads (even if 2D)
- Any non-matrix parameters

### 8.2 Hyperparameter Tuning Strategy

**Minimal Tuning (Recommended):**
1. Start with AdamW-tuned LR and WD (apply RMS scaling adjustment)
2. Use momentum=0.95, nesterov=True, ns_steps=5
3. Monitor early training (first 10-20% of steps)
4. Adjust only if instability observed

**Full Tuning (If Necessary):**
1. Run fine-grained sweep at small scale (100M-500M params)
2. Validate at medium scale (1B-2B params)
3. Use telescoping hyperparameter search: contract grid as width doubles
4. Focus on LR and WD; leave momentum/ns_steps at defaults

**muP Transfer:**
- Enables LR/WD transfer across widths (validated to 3.7B)
- Reduces tuning overhead significantly
- Particularly valuable for large-scale experiments

### 8.3 Learning Rate Schedules

**Recommended:**
- Linear warmup (first 5-10% of training)
- Cosine decay to 0.1 of max LR
- Max LR typically 1.5-3x higher than AdamW equivalent (due to spectral norm constraints)

**Weight Decay Scheduling:**
- Constant 0.1 for first 80% of training
- Clip/reduce at 80% mark for final validation loss improvement
- Clipping too early hurts performance

### 8.4 Monitoring and Debugging

**Key Metrics:**
- Training/validation loss curves
- Gradient norms (per layer)
- Weight spectral norms
- Parameter RMS values
- Attention logit maximums (transformer-specific)
- Newton-Schulz convergence error (if instrumented)

**Red Flags:**
- Sudden gradient norm spikes
- Loss divergence or NaN
- Spectral norms growing unbounded
- RMS values exceeding bf16 range
- Attention logits >100 in early training
- AdamW catching up after initial Muon advantage

---

## 9. Open Questions and Research Gaps

### 9.1 Theoretical Understanding

**Unresolved Issues:**
1. **Formal convergence proofs:** Most variants (AdaMuon, COSMOS) lack guarantees
2. **Optimal NS iteration count:** Why does 5 work? Is it task-dependent?
3. **Natural gradient connection:** Precise relationship to Riemannian optimization unclear
4. **Curvature approximation:** How well does spectral norm approximate true curvature?

### 9.2 Scalability Questions

**Open Problems:**
1. **20B+ parameter scaling:** Will Muon maintain advantages at massive scale?
2. **Distributed NS iterations:** How to parallelize efficiently across GPU clusters?
3. **Token scaling:** Performance at 1T+ tokens not fully characterized
4. **Reinforcement learning:** Will Muon work for RL workloads (e.g., RLHF)?

### 9.3 Practical Gaps

**Needs More Research:**
1. **Fine-tuning protocols:** How to transition from AdamW-pretrained checkpoints?
2. **Non-transformer architectures:** Does Muon help CNNs, RNNs, state-space models?
3. **Multi-modal models:** Interaction with vision encoders, audio encoders
4. **Sparse architectures:** Muon for MoE, sparse attention patterns

---

## 10. Recommendations for Trader-AI

### 10.1 Implementation Strategy

**Phased Approach:**
1. **Phase 1 - Baseline:** Establish AdamW baseline with current architecture
2. **Phase 2 - Muon Integration:**
   - Apply Muon to hidden layers only
   - Keep AdamW for embeddings/heads
   - Use defaults: momentum=0.95, ns_steps=5, nesterov=True
3. **Phase 3 - Hyperparameter Transfer:**
   - Start with AdamW LR * 1.5-2.0
   - Use weight decay = 0.1
   - Apply cosine decay schedule
4. **Phase 4 - Monitoring:**
   - Track all metrics from Section 8.4
   - Compare convergence speed vs AdamW
   - Validate grokking acceleration on held-out data

### 10.2 Expected Benefits

**Likely Improvements:**
1. **Training speed:** 1.5-2x faster to target validation loss
2. **Grokking:** Faster transition from memorization to generalization
3. **Memory:** ~33% reduction in optimizer states
4. **Stability:** Better gradient norm control with proper WD

**Potential Risks:**
1. **Hyperparameter sensitivity:** May require careful LR/WD tuning
2. **Numerical precision:** Watch RMS growth at scale
3. **Optimizer mismatch:** Cannot fine-tune existing AdamW checkpoints
4. **Implementation complexity:** Requires separate optimizers for different parameter types

### 10.3 Monitoring Priorities

**Critical Metrics:**
1. **Loss convergence:** Primary success indicator
2. **Gradient norms:** Early warning for instability
3. **Spectral norms:** Verify spectral regularization active
4. **Grokking onset:** Track when validation loss drops (memorization → generalization)
5. **bf16 range:** Ensure numerical stability

**Diagnostic Tools:**
- Log gradient norms per layer per step
- Track spectral norms every N steps
- Monitor NS convergence error (if possible to instrument)
- Visualize loss curves: train vs validation
- Plot attention logit distributions

### 10.4 Fallback Plan

**If Muon Underperforms:**
1. Verify weight decay is active and set correctly (0.1)
2. Check NS iterations (5 should suffice)
3. Try optimized NS coefficients (3.4445, -4.7750, 2.0315)
4. Reduce learning rate if instability persists
5. Consider AdaMuon or block-wise orthogonalization variants
6. Revert to AdamW if gains don't justify complexity

---

## 11. Key Formulas and Pseudocode

### 11.1 Newton-Schulz Iteration

```
def newton_schulz(A, num_iterations=5):
    """
    Compute orthogonal polar factor Q of matrix A.

    Args:
        A: Input matrix
        num_iterations: Number of NS iterations (default 5)

    Returns:
        Q: Orthogonal matrix (polar factor of A)
    """
    # Normalize
    X = A / torch.norm(A, p='fro')

    # Iterate
    for i in range(num_iterations):
        X = 0.5 * X @ (3 * I - X.T @ X)

    return X
```

### 11.2 Muon Update Step

```
def muon_step(params, grads, momentum_buffer, lr, wd, beta=0.95, ns_steps=5):
    """
    Single Muon optimization step.

    Args:
        params: Current parameters
        grads: Gradients
        momentum_buffer: Momentum state
        lr: Learning rate
        wd: Weight decay
        beta: Momentum coefficient
        ns_steps: Newton-Schulz iterations
    """
    # Update momentum
    momentum_buffer = beta * momentum_buffer + (1 - beta) * grads

    # RMS normalization
    rms = torch.sqrt(torch.mean(momentum_buffer ** 2))
    normalized = momentum_buffer / (rms + 1e-8)

    # Orthogonalize
    orthogonalized = newton_schulz(normalized, ns_steps)

    # Update parameters
    params = params - lr * orthogonalized - wd * params

    return params, momentum_buffer
```

### 11.3 Convergence Bound

**Theoretical Result:**

Under standard assumptions (Lipschitz gradients, bounded variance), Muon with weight decay achieves:

**E[||∇f(W_T)||^2] ≤ O(1/T) + error_ns**

Where:
- T: Number of training steps
- error_ns: Accumulated Newton-Schulz approximation error
- Learning rate constraint: **lr < c / (L * sqrt(T))** for some constant c
- Weight decay improves constant factors in bound

**Error Accumulation:**

Newton-Schulz error grows with condition number κ(M):

**||Q_ns - Q_exact||_F ≤ C * κ(M)^2 / num_iterations**

Where:
- Q_ns: Newton-Schulz approximation
- Q_exact: True polar factor (via SVD)
- κ(M): Condition number of moment matrix
- C: Constant depending on NS coefficients

---

## 12. Sources and References

### Primary Sources

1. [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) - Keller Jordan blog
2. [Deriving Muon](https://jeremybernste.in/writing/deriving-muon) - Jeremy Bernstein
3. [Towards Understanding Orthogonalization in Muon](https://openreview.net/forum?id=ppmyFtr9EW) - OpenReview
4. [Squeezing 1-2% Efficiency Gains Out of Muon by Optimizing the Newton-Schulz Coefficients](https://leloykun.github.io/ponder/muon-opt-coeffs/) - Franz Louis Cesista
5. [Building the Muon Optimizer in PyTorch: A Geometric Approach to Neural Network Optimization](https://medium.com/@kyeg/building-the-muon-optimizer-in-pytorch-a-geometric-approach-to-neural-network-optimization-17f4601be548) - Kye Gomez
6. [Muon Optimizer: The Power of Collective Momentum](https://huggingface.co/blog/onekq/muon-optimizer) - Hugging Face blog

### Grokking Research

7. [Muon Optimizer Accelerates Grokking](https://arxiv.org/abs/2504.16041) - arXiv 2504.16041
8. [Muon Optimizer Significantly Accelerates Grokking in Transformers](https://www.marktechpost.com/2025/04/22/muon-optimizer-significantly-accelerates-grokking-in-transformers-microsoft-researchers-explore-optimizer-influence-on-delayed-generalization/) - MarkTechPost

### Theoretical Analysis

9. [Convergence Bound and Critical Batch Size of Muon Optimizer](https://arxiv.org/html/2507.01598) - arXiv 2507.01598
10. [Beyond the Ideal: Analyzing the Inexact Muon Update](https://arxiv.org/html/2510.19933) - arXiv 2510.19933
11. [On the Convergence of Muon and Beyond](https://arxiv.org/html/2509.15816) - arXiv 2509.15816
12. [Isotropic Curvature Model for Understanding Deep Learning Optimization: Is Gradient Orthogonalization Optimal?](https://arxiv.org/html/2511.00674) - arXiv 2511.00674

### Scaling and Variants

13. [Muon is Scalable for LLM Training](https://arxiv.org/html/2502.16982v1) - arXiv 2502.16982
14. [Practical Efficiency of Muon for Pretraining](https://arxiv.org/html/2505.02222v1) - arXiv 2505.02222
15. [NorMuon: Making Muon more efficient and scalable](https://arxiv.org/html/2510.05491v1) - arXiv 2510.05491
16. [MuonBP: Faster Muon via Block-Periodic Orthogonalization](https://arxiv.org/html/2510.16981) - arXiv 2510.16981
17. [AdaMuon: Adaptive Muon Optimizer](https://arxiv.org/html/2507.11005v1) - arXiv 2507.11005

### Newton-Schulz Methods

18. [Accelerating Newton-Schulz Iteration for Orthogonalization via Chebyshev-type Polynomials](https://arxiv.org/abs/2506.10935) - arXiv 2506.10935
19. [A Stable Scaling of Newton-Schulz for Improving the Sign Function](https://faculty.cc.gatech.edu/~echow/pubs/chen-chow-2014.pdf) - Chen & Chow 2014

### Orthogonalization and Second-Order Methods

20. [Orthogonalising gradients to speed up neural network optimisation](https://arxiv.org/abs/2202.07052) - arXiv 2202.07052
21. [Orthonormalising gradients improves neural network optimisation](https://openreview.net/forum?id=ZfcmwokDvr) - OpenReview
22. [Comparing BFGS and OGR for Second-Order Optimization](https://arxiv.org/html/2512.06969v1) - arXiv 2512.06969

### Riemannian Optimization Connection

23. [LoRA meets Riemannion: Muon Optimizer for Parametrization-independent Low-Rank Adapters](https://arxiv.org/html/2507.12142) - arXiv 2507.12142

### Implementation Resources

24. [GitHub - KellerJordan/Muon](https://github.com/KellerJordan/Muon) - Official implementation
25. [NVIDIA Emerging Optimizers - Muon Documentation](https://docs.nvidia.com/nemo/emerging-optimizers/0.1.0/_modules/emerging_optimizers/orthogonalized_optimizers/muon.html)

### Industry Adoption

26. [Kimi.ai on X: "Why We Chose Muon: Our Chain of Thought"](https://x.com/Kimi_Moonshot/status/1897929976948965870) - Kimi K2 (1T params)
27. [Going Beyond AdamW: A Practical Guide to the Muon Optimizer](https://medium.com/@jenwei0312/going-beyond-adamw-a-practical-guide-to-the-muon-optimizer-93d90e91dbd3) - Jennifer Wei

---

## 13. Conclusion

Muon represents a significant advancement in neural network optimization, leveraging Newton-Schulz orthogonalization to achieve 2x efficiency gains over AdamW. Its theoretical foundations connect to natural gradient descent, second-order optimization, and Riemannian geometry, while providing practical benefits through spectral regularization and improved gradient conditioning.

**Key Takeaways:**

1. **Core Innovation:** Orthogonalizing momentum updates via efficient Newton-Schulz iterations
2. **Grokking Acceleration:** 33% faster transition from memorization to generalization
3. **Scaling Success:** Proven to 1T parameters (Kimi K2, GLM4.5)
4. **Critical Requirements:** Weight decay (0.1) and proper hyperparameter tuning essential
5. **Observable Signatures:** Faster convergence, stable gradient norms, bounded spectral norms
6. **Failure Modes:** Ill-conditioned matrices, scale instabilities, optimizer mismatch
7. **Practical Strategy:** Use Muon for hidden layers, AdamW for embeddings/heads

**For Trader-AI Implementation:**

Muon's grokking acceleration is particularly promising for financial forecasting, where discovering true patterns (generalization) vs memorizing noise (overfitting) is critical. The phased implementation approach (Section 10.1) provides a low-risk path to evaluate benefits while maintaining fallback options. Key success factors: proper weight decay (0.1), careful monitoring of gradient/spectral norms, and willingness to iterate on hyperparameters if initial defaults don't transfer cleanly from AdamW.

**Next Steps:**

1. Implement Muon with defaults (momentum=0.95, ns_steps=5, nesterov=True)
2. Transfer AdamW hyperparameters with RMS scaling adjustment
3. Monitor critical metrics: loss curves, gradient norms, spectral norms
4. Track grokking onset: when does validation loss drop?
5. Compare training efficiency: steps to target validation loss
6. Document observed behaviors: which failure modes appear?
7. Iterate on hyperparameters if needed, focusing on LR and WD

---

**Document Status:** Complete theoretical research synthesis
**Last Updated:** 2025-12-16
**Next Review:** After initial Muon implementation testing
