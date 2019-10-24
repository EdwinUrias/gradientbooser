**Gradient-boosted tree classifier**
It is a machine learning technique used for regression analysis and for statistical classification problems, which produces a predictive model in the form of a set of weak prediction models, typically decision trees.
GBT builds trees one at a time, where each new tree helps correct the mistakes made by a previously trained tree.

**Functioning**
Gradient reinforcement models have a very low interpretation capacity because the second tree in the model no longer predicts the same objective as the original model, the later trees in the model seek to predict how far the original predictions of the truth were from use the residues of the previous trees In this way, each subsequent tree of the gradient increase model slowly reduces the general error of the previous trees.
This allows gradient increase models to have a very high predictive power but a low capacity for interpretation.

**Gradient-Boosted Objective**
Gradient reinforcement models have a very low interpretation capacity because the second tree in the model no longer predicts the same objective as the original model, the later trees in the model seek to predict how far the original predictions of the truth were from use the residues of the previous trees In this way, each subsequent tree of the gradient increase model slowly reduces the general error of the previous trees.
This allows gradient increase models to have a very high predictive power but a low capacity for interpretation.

**Advantages**
- Since boosted trees are derived from the optimization of an objective function, basically GBM can be used to solve almost all objective functions that we can write in gradient.
- Performs the optimization in the function space (instead of in the parameter space), which greatly facilitates the use of custom loss functions.
- Predictive power too high.

**Disadvantages**
- GBMs are more sensitive to overfitting if the data is noisy.
- Training usually takes longer due to the fact that trees are built sequentially.
- Los GBM son más difíciles de sintonizar que los RF. Generalmente hay tres parámetros: número de árboles, profundidad de árboles y tasa de aprendizaje, y cada árbol construido es generalmente poco profundo.

**I use Gradient-Boosted.**
Gradient enhancement can be used in the classification learning field. Yahoo and Yandex commercial web search engines use gradient boosting variants in their search engines.

*YAHOO
Our algorithms personalize and classify the content in our search and media products, boost ad selection, detect spam and prevent abuse. These algorithms generate value for advertisers, performance for publishers and productivity for consumers.*
