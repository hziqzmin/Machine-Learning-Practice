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

---

## 📝 Weekly Learning Notes

### Week 02: Concept of Machine Learning (Regression & Gradient Descent)
* **Core Concept:** Formulating linear hypotheses ($H(x) = Wx + b$, matrix form $W^T X$) to minimize Mean Squared Error (MSE) cost: $\text{cost}(W) = \frac{1}{m}\sum (Wx^{(i)} - y^{(i)})^2$[cite: 7, 8, 10]. Derived parameter updates via Gradient Descent: $W := W - \alpha \frac{1}{m}\sum (Wx^{(i)} - y^{(i)})x^{(i)}$[cite: 14, 70].
* **Theoretical Insight:** MSE produces a smooth convex surface for linear regression, ensuring convergence to a global minimum[cite: 15]. For classification, passing predictions through the sigmoid function $g(z) = \frac{1}{1 + e^{-z}}$ into MSE introduces non-convex local minima traps, requiring cross-entropy / log-loss instead[cite: 17, 18, 20].
* **Hands-on Lab & Assignment:**
  * Implemented iterative gradient descent from scratch in PyTorch (`torch.tensor`) without high-level optimizers[cite: 71, 72].
  * Tracked cost minimization across 50 epochs and extended the pipeline from 1D scalar inputs to multi-dimensional feature matrices ($X \in \mathbb{R}^{m \times 2}$)[cite: 72, 73].

### Week 03: Decision Tree
* **Core Concept:** Supervised classification model that partitions data by maximizing Information Gain: $Gain(S, A) = Entropy(S) - \sum \frac{\vert{}S_v\vert{}}{\vert{}S\vert{}}Entropy(S_v)$, reducing dataset impurity (Shannon entropy: $-p_{\oplus}\log_2 p_{\oplus} - p_{\ominus}\log_2 p_{\ominus}$)[cite: 26, 29, 30].
* **Theoretical Insight:** Trees greedily select features that yield the purest subsets, which can be extracted directly as human-interpretable `IF-THEN` rule sets[cite: 31, 34, 36]. Continuous features require threshold discretization, and unconstrained trees overfit rapidly, necessitating depth constraints[cite: 35, 36, 81].
* **Hands-on Lab & Assignment:**
  * Preprocessed categorical data using `pandas` and `scikit-learn`'s `LabelEncoder` on the *PlayTennis* dataset, trained `DecisionTreeClassifier(criterion='entropy')`, and visualized tree splits using `graphviz`[cite: 77, 78, 79].
  * Modeled an *Obesity Level Estimation* dataset (UCI repository): excluded leakage features (`Height`, `Weight`), binarized a 7-class target into binary classification, and tuned `max_depth` to prevent overfitting[cite: 80, 81].

### Week 04: Support Vector Machine (SVM)
* **Core Concept:** Margin-based classification identifying the optimal hyperplane ($wx + b = 0$) maximizing the margin $M = \frac{2}{\Vert{}w\Vert{}}$, formulated as the convex quadratic optimization problem: minimize $\frac{1}{2}\Vert{}w\Vert{}^2$[cite: 40, 44, 45, 47].
* **Theoretical Insight:** Solving the dual formulation using Lagrange multipliers and KKT conditions reduces optimization entirely to inner products ($x_i^T x_j$)[cite: 47, 48, 49]. This enables the **Kernel Trick** ($K(u, v) = \varphi(u) \cdot \varphi(v)$, e.g., Linear, Polynomial, RBF) to linearly separate non-linear problems (such as XOR) in high-dimensional Hilbert space without explicitly computing high-dimensional coordinates[cite: 40, 51, 52].
* **Hands-on Lab & Assignment:**
  * Built an SMS Spam filter on the *SMSSpamCollection* dataset[cite: 84].
  * Preprocessed text using Keras `Tokenizer`, transformed raw sentences into fixed-length padded integer sequences (`max_length=60`), and evaluated linear `SVC` with soft-margin parameter $C$[cite: 86, 87].
  * Identified misclassifications and explored performance enhancements (e.g., hyperparameter tuning, RBF kernels, text vectorization approaches)[cite: 87, 88, 89].

### Week 05: Statistical Models (HMM & Viterbi Algorithm)
* **Core Concept:** Probabilistic generative model for sequential data decomposing joint probability $P(X, Y)$ via the 1st-order Markov assumption (transition matrix $A$: $P(y_i\vert{}y_{i-1})$) and conditional observation independence (emission matrix $B$: $P(x_i\vert{}y_i)$)[cite: 60, 64].
* **Theoretical Insight:** Directly scoring all potential label combinations is computationally prohibitive ($O(\vert{}S\vert{}^T)$). The **Viterbi Algorithm** resolves this decoding problem in polynomial time using dynamic programming to trace the maximum-likelihood hidden state trajectory[cite: 66].
* **Hands-on Lab & Assignment:**
  * Implemented hidden Markov models with `hmmlearn` (`hmm.MultinomialHMM`)[cite: 93, 94].
  * Defined initial state vectors ($\pi$), state transition matrices, and emission probability matrices to solve weather state inference from clothing observations (`Boots`, `Shoes` $\rightarrow$ `Rainy`, `Sunny`)[cite: 92, 93, 94].
  * Decoded optimal activity sequences (`walk, walk, clean, shop`) and evaluated log-likelihood scores using `model.decode()`[cite: 94, 95].
