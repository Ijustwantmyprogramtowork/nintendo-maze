# Wallace the Gold Prospector – Executive Summary

## Objective
Train an agent to explore a random maze and collect gold. Initial approach used **Q-learning**, later shifted to **A* + Multi-Armed Bandit (MAB)**.

---

## Q-Learning Approach

- **State**: position, nearby walls, relative gold  
- **Actions**: up, down, left, right, gather  
- **Model**: 3-layer MLP `[64,128,64]`  
- **Training**: experience replay, epsilon-greedy exploration, AdamW, MSE loss  

**Observations**:  
- Agent learns (reward increases), but **fails to collect gold** in evaluation.  
- Highly unstable due to **random maze & gold placement**.  
- **FOCUSED local search** around gold wastes time → candidate for removal.

---

## A* + UCB Exploitation

- **BROAD exploration**: reach unvisited cells using A*; guide toward geometric center of gold.  
- **Exploitation**:
  - Map 4 gold sources (arbitrary)  
  - Treat each gold as a bandit arm  
  - Select using **UCB**: `(avg_reward + bonus)/path_cost`  
  - Plan A* path → `GATHER` gold  

**Behavior**: alternates between broad exploration and targeted exploitation.  

**Results**:  
- Effective and robust, but highly variable: 14 → 4700 gold/1200 steps.  
- Variance due to stochastic environment and partial exploration.


## Important Update (Design Revision)

After further analysis, I realized that the **FOCUSED exploration mode** (local A* search around gold) was inefficient and often caused the agent to lose time.  
This component is therefore a strong candidate for removal in future iterations.

Instead, I transitioned to a **Multi-Armed Bandit (MAB)** formulation:

### Phase 1: Pure Exploration
- Maze mapping (no gold collection)
- Detection and mapping of a fixed number of gold sources (**4**, chosen arbitrarily)

### Phase 2: Exploitation
- Each gold source is treated as a **bandit arm**
- Selection is done using **UCB**
- Rewards update arm statistics (mean & visit count)

**Results**:  
- Less variable but results are less impressive. ( mean of 5 golds in 1234 steps )

This cleanly separates:
- **Spatial planning** → A*
- **Decision-making under uncertainty** → MAB

**Future extension:** Replace UCB with **Thompson Sampling**.


---

## Key Takeaways

1. **Q-learning**: promising but cannot generalize in sparse, random mazes.  
2. **A* + UCB**: simple planning + MAB solves exploration/exploitation efficiently.  
3. MAB has shown good promise, better result than with A* and UCB only.
3. **Future improvement**: replace UCB with **Thompson Sampling** for probabilistic selection.

---
