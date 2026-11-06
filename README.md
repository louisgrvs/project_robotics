# project_robotics
A robotics project with a little worlds model and some RL.

```text
project_robotics/
├── README.md
├── pyproject.toml                 Declares pai_core as an installable package (pip install -e .)
├── .gitignore
│
├── assets/
│   └── mjcf/                      MuJoCo models (XML files)
│       ├── robot.xml              The robot alone, with all its characteristics
│       └── arena.xml              The world: ground, obstacles, lights, options...
│
├── src/
│   └── pai_core/                  Python library of the project
│       ├── robot/                 Mathematical model of the robot
│       │   ├── diff_drive.py      Forward and inverse kinematics
│       │   └── odometry.py        Estimates the robot's position from the wheel encoders
│       ├── sim/                   Interface between the simulator and everything else
│       │   ├── sensors.py         All of the robot's sensors
│       │   └── world.py           Obstacles and random goals, A*
│       ├── perception/            From sensors to observation
│       │   ├── features.py        Builds the observation vector
│       │   └── occupancy_grid.py  Log-odds occupancy grid: a probabilistic map of the world built from the lidar
│       ├── envs/                  Training environment
│       │   └── nav_env.py         Gymnasium navigation environment
│       ├── control/               Classical navigation, without any learning
│       │   ├── go_to_goal.py      Reaches a goal when there are no obstacles
│       │   └── dwa.py             Dynamic Window Approach, handles obstacles
│       ├── rl/                    Reinforcement learning
│       │   ├── train_ppo.py       PPO training (Stable-Baselines3)
│       │   └── curriculum.py      A "school" for training: we start with very easy scenes and make them harder and harder
│       ├── world_model/           World models (naive and latent)
│       │   ├── dataset.py         Reorganizes the data produced by collect_data.py to train the world model
│       │   ├── naive.py           The simplest possible world model
│       │   ├── encoders.py        Maps observations into a latent space
│       │   ├── predictor.py       Advances the latent state by one step
│       │   ├── losses.py          Loss functions that prevent the training from collapsing (BYOL-style mechanism)
│       │   ├── train_wm.py        Training of the world model
│       │   └── probes.py          Tools to understand how the latent space is organized
│       ├── planning/              Uses the world model to plan actions
│       │   ├── mppi.py            Generic MPPI planner, to choose which path to follow
│       │   └── costs.py           Cost functions
│       └── eval/                  Determines which method works best
│           ├── scenarios.py       200 fixed scenes to test PPO, MPPI and DWA
│           ├── metrics.py         Evaluation metrics for each method, to tell which one is actually the best
│           ├── benchmark.py       Uses scenarios.py and metrics.py to produce all the results
│           └── wm_metrics.py      World model error, if I have time
│
├── scripts/                       Experiments for the project
│   ├── sandbox_free_fall.py       Free fall, with a comparison between Euler and RK4
│   ├── validate_kinematics.py     Simulated robot compared to the theoretical motion
│   ├── exp_e1_odometry.py         Experiment E1: how well does odometry work?
│   ├── eval_classic.py            First evaluation, before benchmark.py exists
│   ├── collect_data.py            Data collection for training the world model
│   └── make_figures.py            All figures and results
│
├── tests/                         pytest tests
│   ├── test_kinematics.py         Checks diff_drive.py
│   ├── test_reward.py             Checks that the reward function does what we actually want
│   └── test_identity.py           Checks that the transition from Gymnasium to ROS actually works
│
├── ros2_ws/                       colcon workspace (ROS 2 Jazzy)
│   └── src/
│       ├── pai_ros/               ROS 2 package of the project
│       └── pai_interfaces/        NavigateToGoal action (ament_cmake)
│
├── notebooks/                     Notebooks for quick tests and trying out ideas
│
└── docs/
    ├── notes/                     Technical notes on the physics and the chosen parameters
    ├── figures/                   Experiment figures
    └── presentation/              Slides presenting the project
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
