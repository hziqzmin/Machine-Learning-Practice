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

### Week 02: Concept of Machine Learning (Regression & Gradient Descent)
* **Core Concept:** Finding an optimal hypothesis $H(x) = Wx + b$ (or $W^T X$ via matrix representation) that minimizes the Mean Squared Error (MSE) cost function: $\text{cost}(W, b) = \frac{1}{m}\sum_{i=1}^{m}(H(x^{(i)}) - y^{(i)})^2$.
* **Key Takeaway:** Parameter optimization is achieved using Gradient Descent ($W := W - \alpha \frac{\partial}{\partial W}\text{cost}(W)$). While linear regression with MSE forms a smooth convex function, applying MSE directly to logistic hypothesis ($g(z) = \frac{1}{1 + e^{-z}}$) produces local minima traps. Hence, classification requires the cross-entropy/log-loss cost function to guarantee convergence.
* **Key Topics:** Hypothesis representation, MSE cost function, Gradient Descent derivation, Convexity, Transition from Regression to Logistic Classification.

### Week 03: Decision Tree
* **Core Concept:** A non-parametric supervised model that constructs decision rules recursively by prioritizing features with the highest Information Gain ($Gain(S, A)$).
* **Key Takeaway:** Information Gain represents the expected reduction in total entropy (disorder/uncertainty) after knowing a feature: $Gain(S, A) = Entropy(S) - \sum_{v}\frac{|S_v|}{|S|}Entropy(S_v)$, where binary entropy is $-p_{\oplus}\log_2 p_{\oplus} - p_{\ominus}\log_2 p_{\ominus}$. Continuous features must be discretized into threshold boundaries, and deep trees can easily become complex, requiring feature selection and rule pruning.
* **Key Topics:** Entropy & Information Gain, Play Tennis example, Converting tree paths to IF-THEN rules, Continuous/complex feature handling, Advantages & Disadvantages (C4.5 / CART).

### Week 04: Support Vector Machine (SVM)
* **Core Concept:** A margin-based classifier that identifies the optimal maximum-margin hyperplane ($wx + b = 0$) maximizing the separation distance $M = \frac{2}{\|w\|}$, which is equivalent to minimizing $\frac{1}{2}\|w\|^2$.
* **Key Takeaway:** By transforming the primal objective into a dual optimization problem via Lagrange multipliers and Karush-Kuhn-Tucker (KKT) conditions, the classification function depends purely on vector dot products. Applying the **Kernel Trick** ($K(u, v) = \varphi(u) \cdot \varphi(v)$, e.g., Polynomial, RBF) maps non-linearly separable data (such as XOR) into higher-dimensional space where linear separation becomes feasible without explicit high-dimensional coordinates.
* **Key Topics:** Support Vectors & Maximum Margin, Primal vs. Dual problem, KKT conditions, Kernel Functions & Kernel Trick (Polynomial, RBF, Sigmoid), Solving non-linear XOR problems.

### Week 05: Statistical Models (HMM & Viterbi Algorithm)
* **Core Concept:** A probabilistic graphical model for sequence labeling where hidden states ($y$) generate observable features ($x$), decomposed via the 1st-order Markov assumption (transition probability $P(y_i|y_{i-1})$) and observation independence assumption (emission probability $P(x_i|y_i)$).
* **Key Takeaway:** Full joint sequence probability is given by $\arg\max \prod P(y_i|y_{i-1})P(x_i|y_i)$. Rather than evaluating all exponentially combinatorial sequences, the **Viterbi Algorithm** uses dynamic programming to efficiently determine the globally optimal hidden state path in linear time.
* **Key Topics:** Markov Chains vs. Hidden Markov Models (HMM), Transition & Emission probability matrices, Sequence labeling (Part-of-Speech tagging), Viterbi path decoding algorithm.
