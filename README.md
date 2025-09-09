# Report from Wallace the Gold Prospector

## Contents
- [Part 1: Q-Learning](#part-1--q-learning-deep-rl)
  - [Agent Architecture](#agent-architecture)
  - [Exploration and Learning](#exploration-and-learning)
  - [Limitations](#limitations)
- [Part 2: A* Technique](#part-2-a-technique)
  - [Exploration Modes](#1-exploration-with-a)
  - [Exploitation Strategy](#3-exploitation-and-gold-collection)
- [Conclusion](#conclusion)


## Part 1 – Q-Learning (Deep RL)

### Objective

The initial idea was to use a Q-learning based agent to learn how to explore a randomly generated maze and maximize gold collection.

I considered using a genetic algorithm but ultimately chose Q-learning, even though it was new to me.

---

### Agent Architecture

- **Input**: 7 features (current position, surrounding walls, relative position to the gold)
- **Output**: 5 Q-values (corresponding to the 5 possible actions)
- **Neural network**: 3-layer MLP `[64, 128, 64]`
- **Optimizer**: `AdamW`, `lr = 1e-3`
- **Loss**: Standard MSE
- **Gradient clipping** enabled
- **Target network** updated every 10 episodes
- **Experience replay buffer** storing transitions `(state, action, reward, next state, done)`
- **Batch training**: 64 randomly sampled transitions
- **Fitness**: average episode rewards (gold + exploration bonuses + penalties)
- **Memory size**: 10,000 transitions

---

_Note: I created a new function to generate a maze without associated video, fixed size (13x13) to stabilize training. I could also have added a `create_video=False` argument in `maze.py` and used it._

### Exploration and Learning

#### Initial Problems

Initially, the agent failed to explore and remained in its initial state.

![q_learning_gif](/pics/q_learning_demo.gif)

Solutions tested:

- **Simple Argmax** → caused pointless left/right loops
- **Softmax sampling** (temperature = 2) → more random
- Added **spatial bonuses/penalties** to force exploration:
  - `-0.5` if it stays within a radius of 5 cells around the initial position
  - `+0.2` if it moves 3 cells away, `+0.3` at 5 cells, `+0.6` at 10 cells
  - `+0.5` if it reaches at least 2 squares from a far corner (other than the starting one)

> Distances are measured using **Manhattan** distance (more logical on a grid than **Euclidean**).

#### Switching to Epsilon-Greedy

This worked best, although the improvement was limited:

- `epsilon = 1.0` initially (full exploration)
- Gradual decrement (epsilon decay)
- Ultimately better than softmax sampling

---

### Useful Parameters

- **Number of episodes / steps**:
  - `500×500` → 2.5 rewards
  - `100×1000` → 0 reward (too long, too unstable)
  - `1000×100` → **3.24 rewards** (best tradeoff)

- Unsuccessful attempts (due to lack of time):
  - Dyna-Q
  - Optimistic Start

---

### Limitations

> With a **random maze each episode** and **changing gold placement**, it is too unstable for simple Q-learning.

Even with reward heuristics (exploration bonuses, stagnation penalties), the agent fails to generalize.

---

This inspired a human-inspired strategy: explore until gold is found, then return and exploit the region.



How to use the code:
```bash
cd experiments
python q_learning_solution.py
```

And to see the agent in a new environment how he performs ( with video ):
```bash
python evaluate_q_learning.py
```



___________________________________



## Part 2: A* Technique

I had already seen the A* algorithm, and from what I understood, it is an AI algorithm used to find the best path on the first try.  
A* uses two components:

- The cost already traveled: g(n) – the distance from the start to point n.
- The estimated cost to the goal: h(n) – an estimate of the distance from n to the target.

It tries to minimize:  
**f(n) = g(n) + h(n)**  
That is: known path + estimate of the remainder.

### Summary of Wallace’s Functioning

#### 1. Exploration with A*

- Wallace uses the A* algorithm to plan his movements.
- In **BROAD** mode (wide exploration), he tries to reach unvisited cells.
- The A* heuristic targets the geometric center of known gold zones, or his current position if none are known.
- This enforces systematic exploration of the maze.

#### 2. FOCUSED Mode (local search around gold)

- When he finds gold, he switches to **FOCUSED** mode.
- He explores locally around this position using A* within a limited radius.
- The goal is to circle around the gold to find other nearby sources.

#### 3. Exploitation and Gold Collection

- Wallace evaluates gold locations according to a score combining:
  - Average value collected,
  - Number of visits,
  - Path cost (distance).
- He chooses the most profitable gold source based on this score (UCB).
- He plans an A* path to this source and collects the gold (`Action.GATHER`).

#### 4. Overall Behavior

- Wallace automatically alternates between:
  - Broad exploration (BROAD),
  - Local exploration (FOCUSED),
  - Optimized exploitation (gold collection),
- All relying on A* to guarantee optimal movement.

---

The results for this technique are impressive but __EXTREMLY VARIABLE__. For 1200 steps, the collected gold ranges from 14 to 4700.

![a_star_gif](/pics/video_1.gif)
#### _Why are the results so variable?_

Several factors may explain this wide variation:

- **Environment variability**  
  The maze or gold distribution can change between runs, affecting performance.

- **Stochastic behavior**  
  When Wallace has no plan, he chooses random actions, leading to very different paths.

- **Exploitation/exploration strategy**  
  Wallace can sometimes get stuck or poorly exploit rich gold zones, reducing his harvest.

---


How to use the code:
```bash
python test_wallace.py
```



# Conclusion

While the Q-learning approach may appear more sophisticated and promising, it proved challenging to optimize within the limited timeframe of this test. With more time, I would have liked to explore the **Dyna-Q** algorithm, which — from what I understand — introduces a more **planning-based strategy** compared to standard Q-learning.

From the training data, we can clearly observe that the agent **is learning**, even if the loss increases from **20 to 66**, which is expected during exploration.
- This rise reflects the gap between predicted Q-values and optimal Q-values as the agent explores new, unseen states — rather than just repeating known paths (like naive argmax strategies stuck in loops).
However, the average reward increases from 3 to 6 per episode.



However, in testing conditions, the agent **fails to collect any gold over 1000 steps** — likely due to insufficient generalization or convergence. Despite the increase in average reward during training, it's not enough to ensure robust performance.

In contrast, the **A\*** algorithm proves far more effective and **robust to the maze's randomness**, even if the amount of gold collected can vary widely. Its planning-based approach, coupled with systematic exploration and exploitation, leads to significantly better results in practice.

The results for A* are in results with the performance analysis.md and the results of q learning is in the experiments folder with the python files.