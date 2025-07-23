"""
Evaluate Q-Learning agent performance in a new maze with video generation.

This module tests a trained Q-Learning agent in an unseen maze environment
and generates a video recording of the agent's behavior for analysis.
"""

import os
import numpy as np
from q_learning_solution import QLearningAgent
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

create_maze = maze.create_maze
ActionGather = wrappers.ActionGather
StochasticGolds = wrappers.StochasticGolds
Monitor = wrappers.Monitor



def evaluate_agent_with_video(agent_file, video_name="q_learning_demo", test_seed=999999):
    """
    Test a trained Q-Learning agent in a new maze and generate a video.
    
    This function loads a pre-trained Q-Learning agent and evaluates its performance
    in a completely new maze environment that it has never seen during training.
    A video is generated to visualize the agent's decision-making process.
    
    Args:
        agent_file (str): Path to the saved agent file (.pth format)
        video_name (str): Name for the output video file (without extension)
        test_seed (int): Random seed for maze generation (should be different from training seeds)
        
    Returns:
        dict: Performance metrics containing:
            - training_fitness (float): Agent's fitness during training
            - test_gold (float): Gold collected in test episode
            - steps (int): Number of steps taken
            - actions (list): Sequence of actions taken
            - generalization_ratio (float): Test performance / Training performance
            
    Raises:
        FileNotFoundError: If agent_file doesn't exist
    """
    # Load the trained agent
    if not os.path.exists(agent_file):
        print(f"File {agent_file} not found")
        return None
    
    agent, training_stats = QLearningAgent.load_agent(agent_file)
    agent.epsilon = 0.0  # Deterministic mode (no random exploration)
    
    
    # Create a new maze with video recording (seed never seen during training so I took 999999 randomly)
    np.random.seed(test_seed) 
    env = create_maze(video_prefix=f"./{video_name}", fps=4, overwrite_every_episode=True)
    
    # Initialize episode variables
    obs = env.reset()
    total_gold = 0
    steps = 0
    
    # Reset agent state for new episode
    if hasattr(agent, 'last_obs'):
        delattr(agent, 'last_obs')
    if hasattr(agent, 'last_action'):
        delattr(agent, 'last_action')
    
    
    # Run the episode
    while steps < 1000:  # Same step limit as training
        action = agent.select_action(obs, training=False)  # No exploration
        
        obs, gold, done, info = env.step(action)
        total_gold += gold
        steps += 1
        
        # Log gold collection events
        if gold > 0:
            print(f"Step {steps}: Gold collected! (+{gold:.1f}) - Total: {total_gold:.1f}")
        
        # Check if episode ended
        if done:
            print(f"Episode ended at step {steps}")
            break
    
    env.close()
    
    # Analyze performance
    training_fitness = training_stats['fitness']
    
    print(f"\n=== RESULTS ===")
    print(f"Training performance:     {training_fitness:.2f}")
    print(f"Test performance:         {total_gold:.2f}")
    print(f"Steps taken:              {steps}")
    
    

    
    print(f"Video saved: {video_name}.mp4")
    
    return {
        'training_fitness': training_fitness,
        'test_gold': total_gold,
        'steps': steps,
        'generalization_ratio': total_gold/training_fitness
    }


def demo_agent():
    """
    Demonstrate the trained agent with video generation.
    
    This is a convenience function that runs the evaluation with default parameters,
    loading the final trained agent and generating a demo video.
    
    Returns:
        dict: Performance metrics from evaluate_agent_with_video()
    """
    return evaluate_agent_with_video("q_learning_saves/q_agent.pth", "q_learning_demo")


if __name__ == "__main__":
    demo_agent() 