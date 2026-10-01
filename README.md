# Machine Learning Practice & Coursework

Weekly machine learning labs and exercises implemented in Python and Google Colab.

## 📚 Weekly Directory

| Week | Topic | Notebook | Colab Quick Link |
| :---: | :--- | :---: | :---: |
| **02** | Regression & Gradient Descent | [`View Notebook`](./week02_gradient_decent/gradientdecent.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hziqzmin/Machine-Learning-Practice/blob/main/week02_gradient_decent/gradientdecent.ipynb) |
| **03** | Decision Tree | [`View Notebook`](./week03_decision_tree/obesity_decision_tree.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hziqzmin/Machine-Learning-Practice/blob/main/week03_decision_tree/obesity_decision_tree.ipynb) |
| **04** | Support Vector Machine (SVM) | [`View Notebook`](./week04_support_vector_machine/svm_example.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hziqzmin/Machine-Learning-Practice/blob/main/week04_support_vector_machine/svm_example.ipynb) |
| **05** | Hidden Markov Model (HMM) | [`View Notebook`](./week05_hidden_markov_model/hmm_example.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hziqzmin/Machine-Learning-Practice/blob/main/week05_hidden_markov_model/hmm_example.ipynb) |

---

## 📝 Weekly Learning Notes

### Week 02: Concept of Machine Learning (Regression & Gradient Descent)
* **Core Concept:** Finding an optimal linear hypothesis $H(x) = Wx + b$ by minimizing the Mean Squared Error (MSE) cost function via Gradient Descent.
* **Key Takeaway:** MSE yields a convex landscape for linear regression guaranteeing global convergence, whereas applying it to non-linear classification outputs creates local minima traps that require log-loss/cross-entropy.
* **Practice & Lab:** Implemented batch gradient descent from scratch using PyTorch tensors, tracking parameter convergence across 50 epochs before extending the pipeline from 1D scalar inputs to multi-dimensional feature matrices.
* **What I Understand:** It is like placing a ruler through scattered points on paper, starting with a random tilt, and repeatedly taking small steps downhill until our total measuring mistakes are as tiny as possible.

### Week 03: Decision Tree
* **Core Concept:** Non-parametric supervised classification that recursively splits data by maximizing Information Gain to reduce Shannon entropy.
* **Key Takeaway:** Tree splits generate interpretable `IF-THEN` rules; continuous attributes require dynamic thresholding, and unconstrained tree growth leads to overfitting that must be controlled via depth limits.
* **Practice & Lab:** Built an entropy-based decision tree with `scikit-learn` on the *PlayTennis* dataset visualized via `graphviz`, then cleaned UCI obesity data (dropping leakage features like height/weight) to train a regularized binary classifier.
* **What I Understand:** It is like a game of 20 Questions where the computer figures out which question removes the most confusion first, eventually building a clean yes/no flowchart anyone can easily read.

### Week 04: Support Vector Machine (SVM)
* **Core Concept:** Finding an optimal maximum-margin hyperplane ($wx + b = 0$) by minimizing $\frac{1}{2}\Vert{}w\Vert{}^2$ to maximize the geometric margin ($M = \frac{2}{\Vert{}w\Vert{}}$).
* **Key Takeaway:** Formulating the Lagrangian dual via KKT conditions expresses decision boundaries purely through vector inner products, enabling the **Kernel Trick** (e.g., Polynomial, RBF) to linearly separate non-linear data like XOR in higher dimensions.
* **Practice & Lab:** Preprocessed text using Keras `Tokenizer` with fixed sequence padding (`max_length=60`) on the *SMSSpamCollection* dataset, trained a linear `SVC` with soft-margin parameter $C$, and analyzed classification accuracy.
* **What I Understand:** Instead of just drawing any line that separates two groups, it looks for the widest possible safety street between them so new, unseen points are much less likely to end up on the wrong side.

### Week 05: Statistical Models (HMM & Viterbi Algorithm)
* **Core Concept:** Generative sequence modeling combining the 1st-order Markov assumption for hidden state transitions with conditional independence for visible emissions.
* **Key Takeaway:** Directly computing joint sequence probabilities over all paths is exponential, but the dynamic programming **Viterbi Algorithm** resolves decoding by finding the maximum-likelihood hidden state trajectory in linear time.
* **Practice & Lab:** Configured transition, emission, and start probability matrices using `hmmlearn` (`MultinomialHMM`) to infer hidden weather states from clothing choices and decoded optimal observation sequences (`walk, clean, shop`).
* **What I Understand:** Trying to guess what is happening behind closed doors (like the weather outside) just by watching the visible clues or habits that follow one another (like whether someone brought boots or sports shoes).
