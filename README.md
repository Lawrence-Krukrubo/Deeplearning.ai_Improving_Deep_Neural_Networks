# Improving_Deep_Neural_Networks

This project is the second course in the deeplearning.ai specialisation. It involves specific and deliberate steps in improving Deep Neural Networks, such as:-
* **Hyper-parameter tuning**
* **Regularization** 
* **Optimization**

#### The project is broken down into weekly tasks...

## Week 1:

### Key Concepts for this week include:
* Recall that different types of initializations lead to different results
* Recognize the importance of initialization in complex neural networks.
* Recognize the difference between train/dev/test sets
* Diagnose the bias and variance issues in your model
* Learn when and how to use regularization methods such as dropout or L2 regularization.
* Understand experimental issues in deep learning such as Vanishing or Exploding gradients and learn how to deal with them
* Use gradient checking to verify the correctness of your backpropagation implementation

### Tasks for week 1:
**Setting up your Machine Learning Application:**
This includes activities such as:-
* Understanding the importance and split of Train\Dev\Test sets.
* Bias \ Variance Trade-off.
* Basic recipe for machine learning.

**Regularizing your neural network:**
Activities here include:-
* L2 Regularization
* The Frobenius Norm Regularisation
* Dropout Regularisation.
* Understanding the Inverse-Dropout Technique.
* Other Regularisation methods.

**Setting up your optimization problem**
Activities here include:
* Normalisation of inputs - Z-score norm.
* Vanishing / Exploding Gradients.
* Weight initialisation for Deep Networks.
* Numerical approximation of gradients.
* Gradient checking.

### Programming Assignments Week 1:

#### 1. Initialization:
This first notebook, is titled **initialization**. It contains experiments that show the ideal weight initialization techniques to use. It's located in the part_1_initialisation folder. To follow along, kindly import the following modules:-

* `import matplotlib.pyplot as plt`
* `import sklearn`
* `import sklearn.datasets`
Then from the part_1_initialisation folder, import the following
* `from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation`
* `from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec`

#### Specific activities in initialization include:-
* Define a Neural Network model function.
* Zero initialization
* Random initialization
* Random initialization with small weights
* He initialization
* Conclusions

#### 2. Regularization:
Deep Learning models have so much flexibility and capacity that overfitting can be a serious problem, if the training dataset is not big enough. Sure it does well on the training set, but the learned network doesn't generalize to new examples that it has never seen!
This notebook demonstrates how to use **Regularization** in your deep learning models.
To follow along, kindly import the following:

* `import numpy as np`
* `import matplotlib.pyplot as plt`
* `import sklearn`
* `import sklearn.datasets`
* `import scipy.io`
Import these remaining files from the part_2 directory
* `from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec`
* `from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters`
* `from testCases import`

#### Specific activities in Regularization include:-
* Non-regularized model
* L2 Regularization
* Dropout 
* Forward propagation with dropout
* Backward propagation with dropout
* Conclusions

#### 3. Gradient Checking:
Backpropagation is quite challenging to implement, and sometimes has bugs that lead to wrong predictions and a few unpleasant situations. In this Notebook we shall debug the Back-propagtaion algorithm using a process called Gradient-Checking.
To follow along with this Notebook, kindly import:-

* `import numpy as np`
* `from testCases import`
import this from the part_3 folder
* `from gc_utils import sigmoid, relu, dictionary_to_vector, vector_to_dictionary, gradients_to_vector.`

#### Specific activities in Gradient-Checking include:-
* How does gradient checking work?
* 1-dimensional gradient checking
* N-dimensional gradient checking

## License:
This project and all its materials and docs abide under the MIT License.
