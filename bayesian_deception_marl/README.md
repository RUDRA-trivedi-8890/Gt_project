# Bayesian Deception Multi-Agent Reinforcement Learning (MARL)

An Asymmetric Information Markov Signaling Game on an 11x11 Gridworld.

---

## Defender Upgrade: Strong Sequential Bayesian Receiver

### 1. Problem Diagnosis & Motivation
In the asymmetric gridworld signaling game:
- **Attacker**: Receives a private type $\theta \in \{0, 1\}$ denoting Goal A $(10, 0)$ or Goal B $(10, 10)$.
- **Defender**: Observes physical coordinates only (cannot observe $\theta$) and must maintain a sequential Bayesian belief over the attacker's trajectory to prevent goal invasion.

Prior to this upgrade, the Defender suffered from several structural pathologies:
1. **Memoryless Policy**: The actor and critic operated on raw instantaneous spatial inputs (or static snapshot beliefs) rather than sequential hidden representations, preventing the Defender from differentiating deceptive maneuvers (such as feints or delayed reveals) from direct approaches.
2. **Artificial Softening Handicaps**: A softmax temperature of $1.5$ and BCE label smoothing of $0.1$ capped maximum posterior certainty below $0.90$, leaving the screening head perpetually diffuse.
3. **Temporal Scrambling in PPO**: Mini-batch updates randomly shuffled individual transitions across timesteps, destroying sequence continuity.
4. **Cell-Swap / Pass-Through Collisions**: Simultaneous moves enabled agents to cross adjacent cells without triggering post-step point distance checks ($d < 0.8$).
5. **Missing Horizon Awareness**: Defender observations lacked $t / T_{\max}$, hiding how much time remained to protect the goal line.

---

### 2. Upgraded Architecture & Components

```
Defender Observation: [att_x/10, att_y/10, def_x/10, def_y/10, t/T_max] (dim = 5)
                                 │
                                 ▼
                     Shared Recurrent Trunk
                      GRU(5, hidden_dim=64)
                                 │
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
            Belief Head     Policy Head     Value Head
         Linear(64 -> 2)   Linear(66 -> 9) Linear(64 -> 1)
         P(theta | h_t)      Actor pi_def       Critic V(s)
                 │               ▲
                 └───────────────┘
            (Optional stop_gradient)
```

1. **Recurrent Actor-Critic (`models/recurrent_defender.py`)**:
   - Shared 64-dim GRU trunk processing the temporal history of physical coordinates and normalized step $t / T_{\max}$.
   - Sequential belief head predicts $P(\theta \mid h_t)$ directly from the recurrent state.
   - Actor takes $[h_t, b_t]$ with configurable `belief_stop_gradient` to prevent policy variance from degrading belief calibration.
   - Temperature default set to $1.0$ and label smoothing to $0.0$.

2. **Continuous Swept-Sphere & Swap Collision Detection (`envs/grid_world.py`)**:
   - Evaluates discrete cell swaps and continuous-time swept-sphere trajectory segment intersections:
     $$d_{\min} = \min_{t \in [0, 1]} \lVert (A_0 - D_0) + t ((A_1 - A_0) - (D_1 - D_0)) \rVert$$
   - Guarantees immediate interception if agents cross paths.

3. **Sequence-Aware PPO (`algos/sequence_ppo.py`)**:
   - Batches full un-shuffled episodes with boolean sequence masks and $h_0$ hidden state tracking.

4. **Exact Bayes Teacher & Distillation (`algos/exact_bayes.py`)**:
   - Recursive posterior filter:
     $$b_t(\theta) \propto b_{t-1}(\theta) \cdot \pi_{\text{att}}(a_t \mid s_t, \theta)$$
   - Logs $D_{\text{KL}}(\text{GRU belief} \parallel \text{exact Bayes})$ per timestep and provides auxiliary distillation loss.

5. **Scripted Trick Attacker Suite (`envs/trick_attackers.py`)**:
   - 10 curriculum adversaries: `StraightTo3`, `StraightTo5`, `StraightTo7`, `StraightTo8`, `StraightTo9`, `FeintAndCut`, `Zigzag`, `MidPoolDash`, `IdleRush`, `NoiseJitter`.

---

### 3. Empirical Benchmark (Before vs. After across 5 Seeds)

| Metric | Old Memoryless Defender | Upgraded Recurrent Defender | Delta |
| :--- | :---: | :---: | :---: |
| **Trick Suite Interception Rate** | $26.5\% \pm 2.9\%$ | **$50.0\% \pm 1.0\%$** | **$+23.5\%$ (+88.7% rel)** |
| **False Alarm Rate** | $50.0\%$ | **$0.0\%$** | **$-50.0\%$ (Eliminated)** |
| **Attacker BR Win Rate** | $0.4\% \pm 0.9\%$ | **$0.8\% \pm 1.1\%$** | Robust against BR |
| **Calibration (ECE)** | Diffuse ($\sim 0.35$) | **Calibrated ($\le 0.12$)** | **$-65.7\%$ Error** |
| **Swap Collision Pass-Through** | Present (Buggy) | **Zero (Resolved)** | Fixed |

---

### 4. Dynamic Online RL Co-Evolution Engine (`live_demo.py`)

Rather than relying purely on static pre-trained weights or scripted dropdown tricks, `live_demo.py` features a **Live Online RL Adaptation Engine** where both agents visibly evolve across training epochs:
1. **Epoch 1 - 3 (Naive Direct Rush)**: The untrained Attacker takes direct diagonal lines to the goal, leaking private $\theta$ at step 1. The Defender immediately intercepts at row 5.
2. **Online Learning Step**: When caught, the Attacker receives a $-10.0$ penalty, updates the Caught Archive $C_\theta$, and performs a real-time policy backprop update ($\Delta w \sim 0.35$).
3. **Epoch 4 - 7 (Information Leakage Discovery)**: The Attacker discovers the neutral Column 5 corridor, holding Defender belief at $50/50$ ($H(b) \approx 1.0$ bits).
4. **Epoch 8 - 14 (Emergent Pooling Equilibrium)**: The Attacker pools down Column 5 all the way to Row 7 before executing a sharp cut, successfully capturing the goal!
5. **Epoch 15 - 25+ (High-Stakes Arms Race)**: The Defender camps the corridor exit $(8, 5)$, forcing the Attacker to evolve feints and evasive cuts.

**Interactive Presenter Controls:**
- `[E]`: Step forward to the Next Training Epoch.
- `[0]`: Reset learning back to Epoch 1 (Naive Direct Rush) to replay the evolutionary story from scratch.
- `[L]`: Toggle Live Auto-Learning (automatically trains after every match).
- `[D]`: Toggle Defender Engine (Upgraded Recurrent GRU vs Old Memoryless Guard).
- `[T]`: Switch between Dynamic RL Co-Evolution and Scripted Curriculum Tricks.
- `[C]`: Toggle Caught-Route Hazard Overlay ($C_\theta$).

---

### 5. Running the Tests & Benchmark

```powershell
# Run all unit tests (11 passing tests)
python -m pytest Gt_project/bayesian_deception_marl/tests -v

# Run 5-seed benchmark & exploitability check
python Gt_project/bayesian_deception_marl/run_defender_upgrade_experiment.py

# Launch interactive Minecraft-style Pygame demo
python Gt_project/bayesian_deception_marl/live_demo.py
```
