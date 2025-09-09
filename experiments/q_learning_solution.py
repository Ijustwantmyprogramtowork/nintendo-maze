"""
Deep Q-Learning implementation for maze exploration and gold collection.

This module implements a DQN (Deep Q-Network) agent that learns to navigate
through procedurally generated mazes to collect gold. The agent uses experience
replay, target networks, and various exploration bonuses to improve learning.
"""

import numpy as np
import pickle
import os
from collections import deque, defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import importlib.util
import os
import sys

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
wrappers_path = os.path.join(base_path, "wrappers.py")
maze_path = os.path.join(base_path, "maze.py")

if base_path not in sys.path:
    sys.path.insert(0, base_path)

spec_wrappers = importlib.util.spec_from_file_location("wrappers", wrappers_path)
spec_maze = importlib.util.spec_from_file_location("maze", maze_path)
wrappers = importlib.util.module_from_spec(spec_wrappers)
maze = importlib.util.module_from_spec(spec_maze)
spec_wrappers.loader.exec_module(wrappers)
spec_maze.loader.exec_module(maze)

procedural_maze = maze.procedural_maze
ActionGather = wrappers.ActionGather
StochasticGolds = wrappers.StochasticGolds
Monitor = wrappers.Monitor



class DQN(nn.Module):
    """
    Deep Q-Network for the maze environment.
    
    A feedforward neural network that takes maze observations as input
    and outputs Q-values for each possible action.
    
    Args:
        input_size (int): Size of input observation (default: 7)
        hidden_sizes (list): List of hidden layer sizes (default: [64, 128, 64])
        output_size (int): Number of actions (default: 5)
    """
    
    def __init__(self, input_size=7, hidden_sizes=[64, 128, 64], output_size=5):
        super(DQN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers with ReLU activation
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer (no activation - raw Q-values)
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)


class QLearningAgent:
    """
    Q-Learning agent with Deep Q-Network and experience replay.
    
    This agent learns to navigate mazes using DQN with the following features:
    - Experience replay buffer for stable learning
    - Target network for stable Q-value estimation
    - Exploration bonuses for spatial exploration
    - Episode-based reward processing with discounting
    
    Args:
        agent_id (int): Unique identifier for this agent
        input_size (int): Size of observation space
        hidden_sizes (list): Hidden layer sizes for DQN
        output_size (int): Number of actions
        learning_rate (float): Learning rate for optimizer
        gamma (float): Discount factor for future rewards
        epsilon (float): Initial exploration rate for epsilon-greedy
        epsilon_decay (float): Decay rate for epsilon
        epsilon_min (float): Minimum epsilon value
        memory_size (int): Size of experience replay buffer
    """
    
    def __init__(self, agent_id=0, input_size=7, hidden_sizes=[64, 128, 64], output_size=5, 
                 learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01, memory_size=10000):
        
        self.agent_id = agent_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks setup
        self.q_network = DQN(input_size, hidden_sizes, output_size).to(self.device)
        self.target_network = DQN(input_size, hidden_sizes, output_size).to(self.device)
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Experience replay setup
        self.memory = deque(maxlen=memory_size)
        self.episode_buffer = []  # Buffer for current episode transitions
        
        # Exploration tracking
        self.visited_positions = set()  # Positions visited in current episode
        self.maze_size = 13  # Default maze size for normalization
        self.corner_penalty = -0.1  # Penalty for each corner not visited
        
        # Distance-based exploration rewards
        self.starting_position = (3, 3)  # Starting position is always (3,3) by game rules
        self.moves_in_start_circle = 0  # Consecutive moves within 3 squares of start
        self.max_distance_reached = 0  # Maximum distance from start in current episode
        self.distance_bonuses_awarded = set()  # Track which distance bonuses were already given
        
        # Statistics tracking
        self.reset_stats()
        
        # Initialize target network with same weights as main network
        self.update_target_network()
        
    def reset_stats(self):
        """Reset all statistics for new evaluation."""
        self.total_gold = 0.0
        self.total_steps = 0
        self.episodes = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # UP, RIGHT, DOWN, LEFT, GATHER
        self.training_losses = []
        
        # Reset exploration tracking
        self.visited_positions = set()
        self.corner_visit_stats = []  # Track corners visited per episode
        self.proximity_bonus_stats = []  # Track proximity bonuses per episode
        
        # Reset distance-based exploration tracking
        self.starting_position = (3, 3)
        self.moves_in_start_circle = 0
        self.max_distance_reached = 0
        self.distance_bonuses_awarded = set()
        self.distance_bonus_stats = []  # Track distance bonuses per episode
        
    def normalize_observation(self, obs):
        """
        Normalize observation for neural network input.
        
        Args:
            obs: Raw observation [y, x, wall_up, wall_left, wall_right, wall_down, has_gold]
            
        Returns:
            np.ndarray: Normalized observation suitable for neural network
        """
        obs = np.array(obs, dtype=np.float32)
        # Normalize position coordinates for 13x13 maze (center at 0)
        obs[0] = (obs[0] - 6.5) / 6.5  # y position: -1 to 1
        obs[1] = (obs[1] - 6.5) / 6.5  # x position: -1 to 1
        # Wall indicators and has_gold are already 0/1, no normalization needed
        return obs
        
    def get_q_values(self, obs):
        """
        Get Q-values for an observation.
        
        Args:
            obs: Raw observation from environment
            
        Returns:
            np.ndarray: Q-values for each action
        """
        obs_normalized = self.normalize_observation(obs)
        obs_tensor = torch.FloatTensor(obs_normalized).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)
        return q_values.cpu().numpy()[0]
    
    def select_action(self, obs, training=True, temperature=2.0, use_epsilon_greedy=False):
        """
        Select action using either epsilon-greedy or softmax sampling.
        
        Args:
            obs: Current observation
            training (bool): Whether agent is in training mode
            temperature (float): Temperature for softmax sampling
            use_epsilon_greedy (bool): Whether to use epsilon-greedy instead of softmax
            
        Returns:
            int: Selected action (0-4)
        """
        if use_epsilon_greedy:
            # Classic epsilon-greedy: random action or argmax
            if training and np.random.random() < self.epsilon:
                return np.random.randint(5)
            else:
                q_values = self.get_q_values(obs)
                return np.argmax(q_values)  # Greedy action selection
        else:
            q_values = self.get_q_values(obs)
            # Use softmax sampling for exploration
            return self.softmax_action_selection(q_values, temperature=2.0)
    
    def softmax_action_selection(self, q_values, temperature=1.0):
        """
        Select action using softmax probability distribution.
        
        Args:
            q_values (np.ndarray): Q-values for each action
            temperature (float): Temperature parameter for softmax (higher = more random)
            
        Returns:
            int: Selected action
        """
        # Apply temperature scaling and softmax
        scaled_q_values = q_values / temperature
        # Subtract max for numerical stability to avoid exploding values
        exp_q_values = np.exp(scaled_q_values - np.max(scaled_q_values))
        probabilities = exp_q_values / np.sum(exp_q_values)
        
        # Sample action based on probabilities
        return np.random.choice(len(q_values), p=probabilities)
    
    def act(self, obs, gold_received_at_previous_step, done):
        """
        Main action function compatible with the maze environment interface.
        
        This function handles:
        - Episode termination and reward processing
        - Position tracking for exploration bonuses
        - Action selection with gold gathering override
        - Experience buffer management
        
        Args:
            obs: Current observation from environment
            gold_received_at_previous_step (float): Gold collected in previous step
            done (bool): Whether episode is finished
            
        Returns:
            int or None: Selected action (0-4) or None if episode ended
        """
        if done:
            # End of episode - process the episode buffer
            if gold_received_at_previous_step > 0:
                self.total_gold += gold_received_at_previous_step
            
            if self.episode_buffer:
                # Add final reward to last transition
                if self.episode_buffer:
                    last_transition = list(self.episode_buffer[-1])
                    last_transition[2] += gold_received_at_previous_step  # Add to reward
                    self.episode_buffer[-1] = tuple(last_transition)
                
                # Calculate discounted rewards and store episode
                self.process_episode()
                
            # Reset exploration tracking for new episode
            self.visited_positions = set()
            self.starting_position = (3,3)
            self.moves_in_start_circle = 0
            self.max_distance_reached = 0
            self.distance_bonuses_awarded = set()
            self.episodes += 1
            return None
        
        self.total_steps += 1
        
        # Track current position for exploration bonuses
        current_pos = (int(obs[0]), int(obs[1]))  # (y, x) position
        self.visited_positions.add(current_pos)
        
        # Initialize starting position for new episode
        if self.starting_position==(3,3):
            self.starting_position = current_pos
            self.moves_in_start_circle = 0
            self.max_distance_reached = 0
            self.distance_bonuses_awarded = set()
        
        # Calculate distance from starting position (Manhattan distance)
        distance_from_start = abs(current_pos[0] - self.starting_position[0]) + abs(current_pos[1] - self.starting_position[1])
        self.max_distance_reached = max(self.max_distance_reached, distance_from_start)
        
        # Check for gold at current position
        has_gold = obs[6]
        immediate_reward = gold_received_at_previous_step
        
        # Select action using neural network
        action = self.select_action(obs, training=True)
        
        # Override: if we're on gold, always gather it (regardless of NN decision)
        if has_gold:
            action = 4  # GATHER action
        
        # Store transition in episode buffer (we'll get the reward next step)
        if hasattr(self, 'last_obs') and hasattr(self, 'last_action'):
            transition = (
                self.last_obs,      # Previous state
                self.last_action,   # Action taken
                immediate_reward,   # Immediate reward (gold collected)
                obs,               # Current state
                False              # Episode not done
            )
            self.episode_buffer.append(transition)
        
        # Update action statistics
        self.action_counts[action] += 1
        
        # Store current observation and action for next transition
        self.last_obs = obs
        self.last_action = action
        
        return action
    
    def process_episode(self):
        """
        Process completed episode by calculating exploration bonuses and storing in memory.
        
        This function:
        1. Calculates various exploration bonuses/penalties
        2. Applies them to episode rewards
        3. Stores transitions in experience replay buffer
        4. Triggers training if enough experiences are available
        """
        if not self.episode_buffer:
            return
        
        episode_length = len(self.episode_buffer)
        self.episode_lengths.append(episode_length)
        
        # Calculate corner exploration bonus/penalty
        corners = [(1, 1), (1, self.maze_size-2), (self.maze_size-2, 1), (self.maze_size-2, self.maze_size-2)]
        corners_visited = sum(1 for corner in corners if corner in self.visited_positions)
        corners_missed = len(corners) - corners_visited
        corner_penalty = self.corner_penalty * corners_missed
        
        # Track corner visit statistics
        self.corner_visit_stats.append(corners_visited)
        
        # Calculate proximity bonus for being near corners
        proximity_bonus = 0
        for pos in self.visited_positions:
            for corner in corners:
                distance = abs(pos[0] - corner[0]) + abs(pos[1] - corner[1])
                if distance <= 2:  # Within 2 steps of corner
                    proximity_bonus += 0.5
                    break  # Only count once per position
        
        # Track proximity bonus statistics
        self.proximity_bonus_stats.append(proximity_bonus)
        
        # Calculate distance-based exploration bonus
        distance_bonus = 0
        if self.max_distance_reached >= 3:
            distance_bonus += 0.2  # Bonus for leaving starting area
        if self.max_distance_reached >= 5:
            distance_bonus += 0.3  # Additional bonus for exploring further
        if self.max_distance_reached >= 10:
            distance_bonus += 0.6  # Large bonus for exploring far distances
        
        # Calculate distance malus for staying too close to start
        moves_too_close = sum(1 for pos in self.visited_positions 
                             if abs(pos[0] - self.starting_position[0]) + abs(pos[1] - self.starting_position[1]) <= 2)
        distance_malus = -0.5 * moves_too_close / len(self.visited_positions) if self.visited_positions else 0
        
        # Track distance bonus statistics
        if hasattr(self, 'distance_bonus_stats'):
            self.distance_bonus_stats.append(distance_bonus)
        else:
            self.distance_bonus_stats = [distance_bonus]
        
        # Calculate total episode reward (sum of immediate rewards + bonuses)
        episode_immediate_reward = sum(transition[2] for transition in self.episode_buffer)
        total_episode_reward = episode_immediate_reward + corner_penalty + proximity_bonus + distance_bonus + distance_malus
        
        # Apply gold penalty if too much gold was collected (for balance)
        gold_count = sum(1 for transition in self.episode_buffer if transition[2] > 0)
        gold_penalty = -0.1 * gold_count  # Small penalty per gold piece
        total_episode_reward += gold_penalty
        
        # Store episode reward for statistics
        self.episode_rewards.append(total_episode_reward)
        
        # Apply episode reward to all transitions in the episode
        for i, transition in enumerate(self.episode_buffer):
            # Calculate discounted reward from episode end
            steps_from_end = len(self.episode_buffer) - i - 1
            discounted_episode_reward = total_episode_reward * (self.gamma ** steps_from_end)
            
            # Create new transition with combined reward
            enhanced_transition = (
                transition[0],  # state
                transition[1],  # action
                transition[2] + discounted_episode_reward,  # immediate + discounted episode reward
                transition[3],  # next_state
                transition[4]   # done
            )
            
            # Store in replay memory
            self.memory.append(enhanced_transition)
        
        # Clear episode buffer
        self.episode_buffer = []
        
        # Train if we have enough experience
        if len(self.memory) >= 64:  # Minimum batch size
            self.train()
    
    def train(self, batch_size=64):
        """
        Train the Q-network using experience replay.
        
        Args:
            batch_size (int): Number of transitions to sample for training
        """
        if len(self.memory) < batch_size:
            return
        
        # Sample random batch from memory
        batch_indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in batch_indices]
        
        # Convert batch to tensors
        states = torch.FloatTensor([self.normalize_observation(transition[0]) for transition in batch]).to(self.device)
        actions = torch.LongTensor([int(transition[1]) for transition in batch]).to(self.device)
        rewards = torch.FloatTensor([float(transition[2]) for transition in batch]).to(self.device)
        next_states = torch.FloatTensor([self.normalize_observation(transition[3]) for transition in batch]).to(self.device)
        dones = torch.BoolTensor([bool(transition[4]) for transition in batch]).to(self.device)
        
        # Current Q-values for taken actions
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)  # Gradient clipping just in case
        self.optimizer.step()
        
        # Update epsilon for exploration decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Store loss for monitoring
        self.training_losses.append(loss.item())
        
    def update_target_network(self):
        """Update target network with current network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_fitness(self):
        """
        Calculate fitness as average episode reward.
        
        Returns:
            float: Average reward per episode
        """
        if len(self.episode_rewards) == 0:
            return 0.0
        return np.mean(self.episode_rewards)
    
    def get_stats(self):
        """
        Get comprehensive agent performance statistics.
        
        Returns:
            dict: Dictionary containing various performance metrics
        """
        avg_episode_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        avg_episode_length = np.mean(self.episode_lengths) if self.episode_lengths else 0.0
        avg_loss = np.mean(self.training_losses[-100:]) if self.training_losses else 0.0
        avg_corners_visited = np.mean(self.corner_visit_stats) if self.corner_visit_stats else 0.0
        avg_proximity_bonus = np.mean(self.proximity_bonus_stats) if self.proximity_bonus_stats else 0.0
        avg_distance_bonus = np.mean(self.distance_bonus_stats) if hasattr(self, 'distance_bonus_stats') and self.distance_bonus_stats else 0.0
        
        return {
            'agent_id': self.agent_id,
            'fitness': self.get_fitness(),
            'total_gold': self.total_gold,
            'total_steps': self.total_steps,
            'episodes': self.episodes,
            'avg_episode_reward': avg_episode_reward,
            'avg_episode_length': avg_episode_length,
            'epsilon': self.epsilon,
            'avg_loss': avg_loss,
            'action_counts': self.action_counts.copy(),
            'memory_size': len(self.memory),
            'avg_corners_visited': avg_corners_visited,
            'avg_proximity_bonus': avg_proximity_bonus,
            'avg_distance_bonus': avg_distance_bonus,
            'corner_visit_stats': self.corner_visit_stats.copy(),
            'proximity_bonus_stats': self.proximity_bonus_stats.copy(),
            'distance_bonus_stats': getattr(self, 'distance_bonus_stats', []).copy()
        }
    
    def save_agent(self, filename):
        """
        Save agent model and metadata to file.
        
        Args:
            filename (str): Path to save the agent
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'stats': self.get_stats(),
            'memory': list(self.memory)
        }, filename)
        print(f"Saved Q-learning agent {self.agent_id} to {filename}")
    
    @classmethod
    def load_agent(cls, filename, agent_id=0):
        """
        Load agent from file.
        
        Args:
            filename (str): Path to the saved agent file
            agent_id (int): ID to assign to the loaded agent
            
        Returns:
            tuple: (loaded_agent, training_stats)
        """
        checkpoint = torch.load(filename, map_location='cpu', weights_only=False)
        
        agent = cls(agent_id=agent_id)
        agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.epsilon = checkpoint['epsilon']
        
        if 'memory' in checkpoint:
            agent.memory = deque(checkpoint['memory'], maxlen=agent.memory.maxlen)
        
        print(f"Loaded Q-learning agent from {filename}")
        return agent, checkpoint['stats']


def create_small_maze_no_video(agent_seed):
    """
    Create a small 13x13 maze without video recording for training ( way more efficient ).
    
    This function creates a standardized maze environment optimized for
    Q-learning training, without the overhead of video generation.
    
    Args:
        agent_seed (int): Random seed for maze generation
        
    Returns:
        Wrapped maze environment ready for training
    """
    size = 13
    ngolds = 8
    
    np.random.seed(agent_seed)
    env = maze = procedural_maze(size, size, ngolds)
    
    # Apply wrappers for gold gathering and stochastic rewards
    env = ActionGather(env, maze)
    env = StochasticGolds(env, maze, std=0.7)
    env = Monitor(env)
    return env


def train_q_learning_agent(episodes=1000, steps_per_episode=100, save_interval=100):
    """
    Train a Q-learning agent on procedurally generated mazes.
    
    This function implements the main training loop for the DQN agent,
    including periodic saving and target network updates.
    
    Args:
        episodes (int): Number of training episodes
        steps_per_episode (int): Maximum steps per episode
        save_interval (int): How often to save the agent (in episodes)
        
    Returns:
        tuple: (trained_agent, final_stats)
    """
    print(f"Starting Q-Learning Training")
    print(f"Episodes: {episodes}, Steps per episode: {steps_per_episode}")
    os.makedirs("q_learning_saves", exist_ok=True)
    
    agent = QLearningAgent(agent_id=0)
    
    for episode in range(episodes):
        # Create a new maze for each episode with unique seed
        maze_seed = episode * 42
        env = create_small_maze_no_video(maze_seed)
        
        obs = env.reset()
        total_episode_gold = 0.0
        steps = 0
        
        # Reset agent episode state
        if hasattr(agent, 'last_obs'):
            delattr(agent, 'last_obs')
        if hasattr(agent, 'last_action'):
            delattr(agent, 'last_action')
        
        # Run episode
        while steps < steps_per_episode:
            gold_received = 0.0
            done = False
            
            # Take action
            action = agent.act(obs, gold_received, done)
            if action is None:  # Episode ended
                break
                
            # Step environment
            obs, gold_received, done, info = env.step(action)
            total_episode_gold += gold_received
            steps += 1
            
            if done:
                # Handle episode end
                agent.act(obs, gold_received, True)
                break
        
        # If episode didn't end naturally, force end
        if steps >= steps_per_episode and hasattr(agent, 'episode_buffer') and agent.episode_buffer:
            agent.act(obs, 0.0, True)  # Force episode end
        
        env.close()
        
        # Update target network periodically
        if episode % 10 == 0:
            agent.update_target_network()
        
        # Print progress
        if episode % 100 == 0 or episode == episodes - 1:
            stats = agent.get_stats()
            print(f"Episode {episode:4d}: "
                  f"Avg Reward: {stats['avg_episode_reward']:6.2f}, "
                  f"Epsilon: {stats['epsilon']:.3f}, "
                  f"Memory: {stats['memory_size']:4d}, "
                  f"Avg Loss: {stats['avg_loss']:.4f}")
        
        # Save agent periodically
        if episode % save_interval == 0 and episode > 0:
            filename = f"q_learning_saves/q_agent_episode_{episode}.pth"
            agent.save_agent(filename)
    
    # Final save
    final_filename = f"q_learning_saves/q_agent.pth"
    agent.save_agent(final_filename)
    
    final_stats = agent.get_stats()
    print(f"Training completed")
    print(f"Final fitness: {final_stats['fitness']:.4f}")
    print(f"Total episodes: {final_stats['episodes']}")
    print(f"Final epsilon: {final_stats['epsilon']:.4f}")
    print(f"Memory size: {final_stats['memory_size']}")
    
    return agent, final_stats


if __name__ == "__main__":
    print("Starting Q-Learning Training")
    
    agent, stats = train_q_learning_agent(
        episodes=1000,
        steps_per_episode=100,
        save_interval=200
    )
    