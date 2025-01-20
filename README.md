# Abel-RL: Reinforcement Learning for Symbolic Equation Solving

**Abel-RL** is a reinforcement learning (RL) framework designed for solving symbolic equations using RL agents.
The project integrates **Stable-Baselines3 (SB3)** and **custom environments** to explore algorithmic problem-solving.

## 📌 Features
- Custom symbolic equation environments based on 
- Support for **PPO-Mask**, **DQN**, **A2C**, and intrinsic reward mechanisms
- Performance profiling and logging for easy debugging
- Multi-environment parallelization for efficient training

## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/abel-rl.git
cd abel-rl
pip install -r requirements.txt
```

## 📁 Project Structure
```plaintext
abel-rl/
├── backup/          # Old/unused files
├── data/            # Checkpoints and trained models (ignored in Git)
├── develop/         # Experimental scripts and rough work
├── envs/            # Custom RL environments
├── profiling/       # Profiling scripts
├── tests/           # Unit tests
├── training/        # Training scripts
├── scripts/         # For running experiments
├── utils/           # Helper functions
├── .gitignore       # Ignore large/generated files
├── LICENSE          # MIT License
├── README.md        # Project documentation
├── requirements.txt # Python dependencies
```

## 🛠 Usage

### **1️⃣ Train a model**
Run the training script for solving a simple equation:

```bash
python training/train_single_eqn.py --agent_type ppo-mask --main_eqn "a*x+b"
```

### **2️⃣ Evaluate a trained model**

```bash
python training/eval_model.py --agent_path data/a*x+b/ppo-mask_trained_model.zip
```

### **3️⃣ Run performance profiling**

```bash
python profiling/profiler.py
```

## 📜 License
This project is licensed under the **MIT License** – see [LICENSE](LICENSE) for details.

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

Happy training! 🚀
