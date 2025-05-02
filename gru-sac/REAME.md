# Drone Obstacle Avoidance with DAC-SAC

This repository implements the Dual Experience Attention Convolution Soft Actor-Critic (DAC-SAC) algorithm for autonomous drone obstacle avoidance, as described in the research paper "Autonomous Obstacle Avoidance Algorithm for Unmanned Aerial Vehicles Based on Deep Reinforcement Learning".

## Overview

The DAC-SAC algorithm combines several components to improve drone obstacle avoidance:

1. **Soft Actor-Critic (SAC)**: A state-of-the-art reinforcement learning algorithm that balances exploration and exploitation through entropy maximization.

2. **Dual Experience Buffer Pool**: Separate buffers for successful and failed experiences, helping to better balance the training data.

3. **Self-Attention Mechanism**: Convolutional self-attention layers that dynamically adjust focus based on input features.

4. **CNN Architecture**: Specialized convolutional neural network for processing depth images captured by drone sensors.

5. **Delayed Learning**: Option to delay network updates to improve training stability.

## Files

- `Env.py`: Contains the drone environment implementation using AirSim, which handles the interaction between the agent and the simulation.
- `sac.py`: Implements the core DAC-SAC algorithm with its neural network architecture and training procedures.
- `train.py`: Provides command-line tools for training and testing the agent in the environment.

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/drone-obstacle-avoidance.git
cd drone-obstacle-avoidance
```

2. Install the required dependencies:
```
pip install torch numpy matplotlib airsim
```

3. Make sure you have AirSim and Unreal Engine 4 installed for the simulation environment.

## Usage

### Training

To train the agent from scratch:

```
python train.py --mode train --episodes 5000 --log_dir logs
```

To continue training from a saved model:

```
python train.py --mode train --episodes 5000 --continue_train --model_path logs/sac_model_episode_1000.pt
```

Use GPU acceleration (if available):

```
python train.py --mode train --episodes 5000 --gpu
```

Render the environment during training (warning: this slows down training significantly):

```
python train.py --mode train --episodes 5000 --render
```

### Testing

To test a trained agent:

```
python train.py --mode test --episodes 100 --model_path logs/sac_model_final.pt --render
```

Add exploration noise during testing:

```
python train.py --mode test --episodes 100 --model_path logs/sac_model_final.pt --noise 0.1
```

### Command-line Options

- `--mode`: Choose between 'train' or 'test' mode
- `--episodes`: Number of episodes to run
- `--render`: Enable visualization
- `--log_dir`: Directory to save models and logs
- `--model_path`: Path to a saved model file (for testing or continuing training)
- `--max_steps`: Maximum steps per episode
- `--noise`: Amount of exploration noise for testing (0.0 = deterministic)
- `--continue_train`: Continue training from a saved model
- `--gpu`: Use GPU acceleration if available

## Customization

### Modifying SAC Parameters

You can adjust the algorithm parameters by editing the `SACConfig` class in `sac.py`:

```python
class SACConfig:
    # Neural Network
    hidden_dim = 256          # Adjust for more/less complexity
    cnn_features = 64         # Features from CNN
    
    # SAC Parameters
    batch_size = 64           # Batch size for training
    gamma = 0.99              # Discount factor
    tau = 0.005               # Soft update rate
    lr_actor = 3e-4           # Learning rate for actor
    lr_critic = 3e-4          # Learning rate for critic
    alpha_init = 0.2          # Initial temperature
    
    # Buffer sizes
    success_buffer_size = 300000
    fail_buffer_size = 700000
    sample_success_ratio = 0.4  # Ratio of samples from success buffer
```

### Modifying the Environment

The drone environment settings can be customized in `Env.py` by modifying the `Config` class:

```python
class Config:
    depth_image_height = 84
    depth_image_width = 84
    max_lidar_distance = 20
    max_drone_speed = 5
```

## Results Visualization

During training, metrics are logged and periodically plotted to visualize performance:

- Episode rewards
- Success/collision rates
- Average steps per episode
- Critic loss

These plots are saved in the specified log directory and can help you monitor training progress.

## Acknowledgments

This implementation is based on the research described in "Autonomous Obstacle Avoidance Algorithm for Unmanned Aerial Vehicles Based on Deep Reinforcement Learning" by Yuan Gao, Ling Ren, Tianwei Shi, Teng Xu, and Jianbang Ding.