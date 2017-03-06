# EMalgorithmExample
This repository contains code examples for estimating a finite mixture model via the Expectation-Maximization (EM) algorithm for two maximum likelihood applications:

1. Multinomial logit model (Also allows for alternative-specific conditional logit model)
2. Model (1), combined with a normally-distributed continuous outcome
3. Model (2), but with censoring of the continuous outcome

## Folder Structure
The folder structure within each language's folder is outlined below:

- **General Functions**: Contains functions that can be used in any of the model: 
    * MLE objective functions for optimization (used in the Maximization step of the EM algorithm)
    * Type-specific probability updating (used in the Expectation step of the EM algorithm)
    * Logit probability prediction
- **mlogitOnly**: Contains scripts and functions to simulate and estimate the EM algorithm on a logit-only model
- **mlogitAndNormal**: Contains scripts and functions to simulate and estimate the EM algorithm on a logit-and-continuous model
- **mlogitAndNormalCensored**: Contains scripts and functions to simulate and estimate the EM algorithm on a logit-and-continuous model that also incorporates censoring of the continuous outcome

## Tips
Some miscellaneous tips for estimating EM algorithms:

- Finite mixture models are not generally globally concave
- Because of this, you will get different estimates based on different starting values
- General tip is to start from the solution (if a simulation) or start from the zero-types estimate (if not a simulation)
- The overall likelihood of the model should increase with each EM iteration
- If maximization is overly burdensome, you can loosen the convergence criteria at this step for the first few iterations of the algorithm. This may improve performance.
- Convergence is typically more difficult in models where there is not a continuous outcome (i.e. the "mlogitOnly" folder). This is because there is much less variation with which to pick up the unobserved types in discrete response models.
