# Machine Learning Practice & Coursework

Weekly machine learning labs and exercises implemented in Python and Google Colab.

## 📚 Weekly Directory

| Week | Topic | Notebook | Colab Quick Link |
| :---: | :--- | :---: | :---: |
| **02** | Gradient Decent | [`View Notebook`](./week02_gradient_decent/gradientdecent.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hziqzmin/Machine-Learning-Practice/blob/main/week02_gradient_decent/gradientdecent.ipynb) |
| **03** | Decision Trees | [`View Notebook`](./week03_decision_tree/obesity_decision_tree.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hziqzmin/Machine-Learning-Practice/blob/main/week03_decision_tree/obesity_decision_tree.ipynb) |
| **04** | Support Vector Machine (SVM) | [`View Notebook`](./week04_support_vector_machine/svm_example.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hziqzmin/Machine-Learning-Practice/blob/main/week04_support_vector_machine/svm_example.ipynb) |
| **05** | Hidden Markov Model (HMM) | [`View Notebook`](./week05_hidden_markov_model/hmm_example.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hziqzmin/Machine-Learning-Practice/blob/main/week05_hidden_markov_model/hmm_example.ipynb) |

---

## 📝 Weekly Learning Notes

### Week 02: Gradient Descent
* **Core Concept:** First-order iterative optimization algorithm used to find a local minimum of a differentiable loss function by updating parameters in the opposite direction of the gradient.
* **Key Takeaway:** The learning rate ($\alpha$) determines convergence stability; too high causes divergence/overshooting, while too low results in slow training. Feature scaling (e.g., standardization) significantly speeds up convergence.
* **Key Tools & Techniques:** NumPy vectorization, Mean Squared Error (MSE), Batch vs. Mini-batch gradient updates.

### Week 03: Decision Trees
* **Core Concept:** Non-parametric supervised learning algorithm that partitions feature space into homogenous regions via greedy, recursive binary splitting.
* **Key Takeaway:** Tree splitting criteria (Gini Impurity vs. Information Gain/Entropy) measure node purity. Unconstrained trees overfit rapidly on training noise, requiring regularization through hyperparameter constraints (`max_depth`, `min_samples_split`, `min_samples_leaf`).
* **Key Tools & Techniques:** `scikit-learn` (`DecisionTreeClassifier`), tree visualization, feature importance evaluation.

### Week 04: Support Vector Machine (SVM)
* **Core Concept:** Supervised classification method that finds an optimal hyperplane maximizing the functional margin between data classes.
* **Key Takeaway:** The decision boundary depends only on the critical subset of points (support vectors), making SVM memory-efficient. Non-linear boundaries are resolved using the **Kernel Trick** (e.g., RBF kernel) without explicitly projecting points into higher dimensions.
* **Key Tools & Techniques:** `SVC`, linear vs. RBF kernels, tuning the regularization parameter $C$ and kernel coefficient $\gamma$.

### Week 05: Hidden Markov Model (HMM)
* **Core Concept:** Probabilistic graphical model for sequential time-series data where the system transitions between hidden states that probabilistically generate visible observations.
* **Key Takeaway:** Governed by three primary components: initial state probabilities ($\pi$), state transition probabilities ($A$), and emission probabilities ($B$). Solves three core problems: Evaluation (Forward algorithm), Decoding (Viterbi algorithm), and Learning (Baum-Welch expectation-maximization).
* **Key Tools & Techniques:** Transition/emission matrices, sequence prediction, temporal pattern modeling.
