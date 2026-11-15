# project_robotics
# project_robotics

A robotics project combining reinforcement learning and a small world model.

A two-wheeled differential-drive robot, simulated in MuJoCo, must drive from a
start point to a goal while avoiding obstacles. The robot is non-holonomic: it
cannot move sideways, but unlike a car it can turn in place.

## Installation

Tested on Ubuntu 24.04 (WSL2) with Python 3.12. You can download the package and launch the program by doing the following steps :

```bash
git clone https://github.com/louisgrvs/project_robotics.git
cd project_robotics
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Checks

```bash
# the environment can be created
python -c "from pai_core.envs.nav_env import NavEnv; NavEnv().reset(seed=0); print('ok')"

# unit tests (use "python -m pytest", not "pytest", to run inside the virtual environment)
python -m pytest -v

# simulation throughput in steps per second, which sets the training budget
# (use as many parallel environments as you have CPU cores, I have 2 so I put "2")
python scripts/bench_env.py --n-envs 2

# watch episodes in the MuJoCo viewer (requires a display)
python scripts/view_env.py
```

```text
project_robotics/
├── README.md                      (jours 1, 20)
├── pyproject.toml                 (jour 1)       Declares pai_core as an installable package (pip install -e .)
├── .gitignore                     (jour 1)
│
├── assets/
│   └── mjcf/                                     MuJoCo models (XML files)
│       ├── robot.xml              (jours 2, 4)   The robot alone, with all its characteristics
│       └── arena.xml              (jours 2, 5)   The world: ground, obstacles, lights, options...
│
├── src/
│   └── pai_core/                                 Python library of the project
│       ├── robot/                                Mathematical model of the robot
│       │   ├── diff_drive.py      (jour 2)       Forward and inverse kinematics
│       │   └── odometry.py        (jour 4)       Estimates the robot's position from the wheel encoders
│       ├── sim/                                  Interface between the simulator and everything else
│       │   ├── sensors.py         (jour 4)       All of the robot's sensors
│       │   └── world.py           (jour 5)       Obstacles and random goals, A*
│       ├── perception/                           From sensors to observation
│       │   ├── features.py        (jour 5)       Builds the observation vector
│       │   └── occupancy_grid.py  (jour 8)       Log-odds occupancy grid
│       ├── envs/                                 Training environment
│       │   └── nav_env.py         (jours 5, 11)  Gymnasium navigation environment
│       ├── control/                              Classical navigation, without any learning
│       │   ├── go_to_goal.py      (jour 6)       Reaches a goal when there are no obstacles
│       │   └── dwa.py             (jour 6)       Dynamic Window Approach, handles obstacles
│       ├── rl/                                   Reinforcement learning
│       │   ├── train_ppo.py       (jours 10, 11) PPO training (Stable-Baselines3)
│       │   └── curriculum.py      (jour 11)      A "school" for training
│       ├── world_model/                          World models (naive and latent)
│       │   ├── dataset.py         (jour 14)      Reorganizes the data produced by collect_data.py
│       │   ├── naive.py           (jour 15)      The simplest possible world model
│       │   ├── encoders.py        (jour 16)      Maps observations into a latent space
│       │   ├── predictor.py       (jour 16)      Advances the latent state by one step
│       │   ├── losses.py          (jour 16)      Loss functions that prevent collapse
│       │   ├── train_wm.py        (jour 16)      Training of the world model
│       │   └── probes.py          (jour 17)      Tools to understand the latent space
│       ├── planning/                             Uses the world model to plan actions
│       │   ├── astar.py           (jour 5)*      #TODO
│       │   ├── mppi.py            (jour 18)      Generic MPPI planner
│       │   └── costs.py           (jour 18)      Cost functions
│       └── eval/                                 Determines which method works best
│           ├── scenarios.py       (jour 12)      200 fixed scenes to test PPO, MPPI and DWA
│           ├── metrics.py         (jour 12)      Evaluation metrics for each method
│           ├── benchmark.py       (jour 12)      Uses scenarios.py and metrics.py to produce all the results
│           └── wm_metrics.py      (jour 15)      World model error
│
├── scripts/                                      Experiments for the project
│   ├── sandbox_free_fall.py       (jour 1)       Free fall, with a comparison between Euler and RK4
│   ├── validate_kinematics.py     (jour 3)       Simulated robot compared to the theoretical motion
│   ├── exp_e1_odometry.py         (jour 4)       Experiment E1: how well does odometry work?
│   ├── view_env.py                (jour 5)*      #TODO
│   ├── bench_env.py               (jour 5)*      #TODO
│   ├── eval_classic.py            (jour 6)       First evaluation, before benchmark.py exists
│   ├── collect_data.py            (jour 14)      Data collection for training the world model
│   └── make_figures.py            (jour 19)      All figures and results
│
├── tests/                                        pytest tests
│   ├── test_kinematics.py         (jour 2)       Checks diff_drive.py
│   ├── test_odometry.py           (jour 4)*      #TODO
│   ├── test_nav_env.py            (jour 5)*      #TODO
│   ├── test_identity.py           (jour 10)      Checks that the transition from Gymnasium to ROS works
│   └── test_reward.py             (jour 11)      Checks that the reward function does what we want
│
├── ros2_ws/                                      colcon workspace (ROS 2 Jazzy)
│   └── src/
│       ├── pai_ros/               (jours 7-10)   ROS 2 package of the project
│       └── pai_interfaces/        (jour 10)      NavigateToGoal action (ament_cmake)
│
├── notebooks/                                    Notebooks for quick tests and trying out ideas
│
└── docs/
    ├── notes/                     (jour 3)       Technical notes on the physics and the chosen parameters
    ├── figures/                   (jours 1-19)   Experiment figures
    └── presentation/              (jour 20)      Slides presenting the project
```
```text
Bibliography :
Those are temporary and made by AI, I will check if those are true while I read them.
Mobile robotic and simulation
• Documentation officielle de MuJoCo : Overview, Computation, Modeling, XML Reference, Python.
• Documentation officielle de ROS 2 Jazzy : tutoriels Beginner et Intermediate, tf2, fichiers de lancement ; REP 103 et REP 105.
• Lynch et Park, Modern Robotics: Mechanics, Planning, and Control, 2017, chapitre 13.
• Siegwart, Nourbakhsh et Scaramuzza, Introduction to Autonomous Mobile Robots, 2e édition, 2011.
• Thrun, Burgard et Fox, Probabilistic Robotics, 2005.
• Brockett, « Asymptotic stability and feedback stabilization », 1983.
• Astolfi, « Exponential stabilization of a wheeled mobile robot via discontinuous control », 1999.
• Borenstein et Feng, « Measurement and correction of systematic odometry errors in mobile robots », 1996.
• Fox, Burgard et Thrun, « The dynamic window approach to collision avoidance », 1997.
Reinforcment learning and evaluation : 
• Sutton et Barto, Reinforcement Learning: An Introduction, 2e édition, 2018.
• OpenAI, Spinning Up in Deep RL ; documentations de Stable-Baselines3 et de Gymnasium.
• Schulman et al., « High-dimensional continuous control using generalized advantage estimation », 2016.
• Schulman et al., « Proximal policy optimization algorithms », 2017.
• Huang et al., « The 37 implementation details of proximal policy optimization », 2022.
• Ng, Harada et Russell, « Policy invariance under reward transformations », 1999.
• Pardo et al., « Time limits in reinforcement learning », 2018.
• Henderson et al., « Deep reinforcement learning that matters », 2018.
• Agarwal et al., « Deep reinforcement learning at the edge of the statistical precipice », 2021.
• Anderson et al., « On evaluation of embodied navigation agents », 2018.
World Models and Self-Supervised Learning :
• LeCun, « A path towards autonomous machine intelligence », 2022.
• Ha et Schmidhuber, « World models », 2018.
• Hafner et al., « Mastering diverse domains through world models » (DreamerV3), 2023.
• Hansen, Wang et Su, « Temporal difference learning for model predictive control » (TD-MPC), 2022.
• Hansen, Su et Wang, « TD-MPC2: Scalable, robust world models for continuous control », 2024.
• Grill et al., « Bootstrap your own latent » (BYOL), 2020.
• Bardes, Ponce et LeCun, « VICReg: Variance-invariance-covariance regularization for self-supervised learning », 2022.
• Assran et al., « Self-supervised learning from images with a joint-embedding predictive architecture » (I-JEPA), 2023.
• Assran et al., « V-JEPA 2: Self-supervised video models enable understanding, prediction and planning », 2025.
• Zhou et al., « DINO-WM: World models on pre-trained visual features enable zero-shot planning », 2024.
• Sobal et al., « Learning from reward-free offline data: A case for planning with latent dynamics models », 2025.
Learning model, uncertainty and planification :
• Chua et al., « Deep reinforcement learning in a handful of trials using probabilistic dynamics models » (PETS), 2018.
• Lakshminarayanan, Pritzel et Blundell, « Simple and scalable predictive uncertainty estimation using deep ensembles », 2017.
• Janner et al., « When to trust your model: Model-based policy optimization » (MBPO), 2019.
• Lambert et al., « Objective mismatch in model-based reinforcement learning », 2020.
• Williams et al., « Information theoretic MPC for model-based reinforcement learning » (MPPI), 2017.
• Alain et Bengio, « Understanding intermediate layers using linear classifier probes », 2016.
• Roy et Vetterli, « The effective rank: A measure of effective dimensionality », 2007.
• Tobin et al., « Domain randomization for transferring deep neural networks from simulation to the real world », 2017.
```
