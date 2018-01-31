# cheQers
Q-learning AI to play checkers

Current links:
https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
https://arxiv.org/pdf/1509.01549.pdf

# ToDo
- Compare to previous output rather than predicting next for waaaay more efficient training and playing
- Generalize Q_Learner back to a generic Q_learner like Varun intended, adding the checkers_ai as a subclass
  - Or at least generalize so it takes in a model, loss function, and update function (and input and output shape/size) so it can run with any network compatible to checkers
  - Build a separation layer between checkers and Q_learner if doing generalized implementation rather than abstract class

# How to use
This section should be a thing later but for now it's simple enough that you can just look at \_\_init\_\_ and Q_player to figure things out
