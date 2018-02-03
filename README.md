# cheQers
Q-learning AI to play checkers

Current links:
https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
https://arxiv.org/pdf/1509.01549.pdf

# ToDo
- Move choice
    - Look to next start state
    - Might be able to pull this from an old commit
- Rewards
    - Separate from stepping
    - Reward function that applies reward after game (to the last state seen)
- Board dimensionality
    - Can be 8x4 instead of 8x8
- Model not shape
    - Should take model as an arg (instead of shape or as an alternate option)
- Init args
    - Nicer usability
- Save file size
    - Increases somewhere in checkers_learner.\_\_init\_\_
- Epsilon as function
    - Allow decrease of epsilon as Q-function converges