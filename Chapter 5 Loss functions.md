The last three chapters described linear regression, shallow neural networks, and deep neural networks. Each represents a family of functions that map input to output, where the particular member of the family is determined by the model parameters $\phi$. When we train these models, we seek the parameters that produce the best possible mapping from input to output for the task we are considering. This chapter defines what is meant by the “best possible” mapping.

That definition requires a training dataset $\{x_i, y_i\}$ of input/output pairs. A *loss function* or *cost function* $L[\phi]$ returns a single number that describes the mismatch between the model predictions $f(x_i, \phi)$ and their corresponding ground-truth outputs $y_i$. During training, we seek parameter values $\phi$ that minimize the loss and hence map the training inputs to the outputs as closely as possible. We saw one example of a loss function in chapter 2; the least squares loss function is suitable for univariate regression problems for which the target is a real number $y \in \mathbb{R}$. It computes the sum of the squares of the deviations between the model predictions $f(x_i, \phi)$ and the true values $y_i$.

This chapter provides a framework that both justifies the choice of the least squares criterion for real-valued outputs and allows us to build loss functions for other prediction types. We consider *binary classification*, where the prediction $y \in \{0, 1\}$ is one of two categories, *multiclass classification*, where the prediction $y \in \{1, 2, \ldots, K\}$ is one of $K$ categories, and more complex cases. In the following two chapters, we address model training, where the goal is to find the parameter values that minimize these loss functions.

## 5.1 Maximum likelihood
In this section, we develop a recipe for constructing loss functions. Consider a model $f(x, \phi)$ with parameters $\phi$ that computes an output from input $x$. Until now, we have implied that the model directly computes a prediction $y$. We now shift perspective and consider the model as computing a *conditional probability distribution* $Pr(y|x)$ over possible outputs $y$ given input $x$. The loss encourages each training output $y_i$ to have a high probability under the distribution $Pr(y_i|x_i)$ computed from the corresponding input $x_i$ (figure 5.1).
### 5.1.1 Computing a distribution over outputs
This raises the question of exactly how a model $f(x, \phi)$ can be adapted to compute a probability distribution. The solution is simple. First, we choose a parametric distribution $Pr(y|\theta)$ defined on the output domain $Y$. Then we use the network to compute one or more of the parameters $\theta$ of this distribution.

For example, suppose the prediction domain is the set of real numbers, so $y \in \mathbb{R}$. Here, we might choose the univariate normal distribution, which is defined on $\mathbb{R}$. This distribution is defined by the mean $\mu$ and variance $\sigma^2$, so $\theta = \{\mu, \sigma^2\}$. The machine learning model might predict the mean $\mu$, and the variance $\sigma^2$ could be treated as an unknown constant.

### 5.1.2 Maximum likelihood criterion
The model now computes different distribution parameters $\theta_i = f(x_i, \phi)$ for each training input $x_i$. Each observed training output $y_i$ should have high probability under its corresponding distribution $Pr(y_i|\theta_i)$. Hence, we choose the model parameters $\phi$ so that they maximize the combined probability across all $I$ training examples:
$$
\begin{align}
\hat{\phi} &= argmax_{\phi} \left[ \prod_{i=1}^{I} Pr(y_i|x_i) \right] \\
&= argmax_{\phi} \left[ \prod_{i=1}^{I} Pr(y_i|\theta_i) \right] \\
&= argmax_{\phi} \left[ \prod_{i=1}^{I} Pr(y_i|f(x_i, \phi)) \right]  \\
\end{align} \tag{5.1}
$$


The combined probability term is the *likelihood* of the parameters, and hence equation 5.1 is known as the *maximum likelihood criterion*[^1].

Here we are implicitly making two assumptions. First, we assume that the data are identically distributed (the form of the probability distribution over the outputs $y_i$ is the same for each data point). Second, we assume that the conditional distributions $Pr(y_i|x_i)$ of the output given the input are independent, so the total likelihood of the training data decomposes as:

$$
Pr(y_1, y_2, \ldots , y_I|x_1, x_2, \ldots , x_I) = \prod_{i=1}^{I} Pr(y_i|x_i) \tag{5.2}
$$


In other words, we assume the data are *independent and identically distributed (i.i.d.)*.
### 5.1.3 Maximizing log-likelihood

The maximum likelihood criterion (equation 5.1) is not very practical. Each term $Pr(y_i|f(x_i, \phi))$ can be small, so the product of many of these terms can be tiny. It may be difficult to represent this quantity with finite precision arithmetic. Fortunately, we can equivalently maximize the logarithm of the likelihood:

$$
\begin{align}
\hat{\phi} &= argmax_{\phi} \left[ \prod_{i=1}^{I} Pr(y_i|f(x_i, \phi)) \right] \\
&= argmax_{\phi} \left[ \log \prod_{i=1}^{I} Pr(y_i|f(x_i, \phi)) \right] \\
&= argmax_{\phi} \left[ \sum_{i=1}^{I} \log Pr(y_i|f(x_i, \phi)) \right] 
\end{align} \tag{5.3}
$$

This *log-likelihood* criterion is equivalent because the logarithm is a monotonically increasing function: if $x > x'$, then $\log[x] > \log[x']$ and vice versa (figure 5.2). It follows that when we change the model parameters $\phi$ to improve the log-likelihood criterion, we also improve the original maximum likelihood criterion. It also follows that the overall maxima of the two criteria must be in the same place, so the best model parameters $\hat{\phi}$ are the same in both cases. However, the log-likelihood criterion has the practical advantage of using a sum of terms, not a product, so representing it with finite precision isn't problematic.
### 5.1.4 Minimizing negative log-likelihood

Finally, we note that, by convention, model fitting problems are framed in terms of minimizing a loss. To convert the maximum log-likelihood criterion to a minimization problem, we multiply by minus one, which gives us the *negative log-likelihood criterion*:

$$
\hat{\phi} = argmin_{\phi} \left[ - \sum_{i=1}^{I} \log Pr(y_i|f(x_i, \phi)) \right]
= argmin_{\phi} [ L[\phi] ] \tag{5.4}
$$
which is what forms the final loss function $L[\phi]$.

### 5.1.5 Inference

The network no longer directly predicts the outputs $y$ but instead determines a probability distribution over $y$. When we perform inference, we often want a point estimate rather than a distribution, so we return the maximum of the distribution:

$$
\hat{y} = argmax_y [Pr(y|f(x, \phi))]  \tag{5.5}
$$
(5.5)

It is usually possible to find an expression for this in terms of the distribution parameters $\theta$ predicted by the model. For example, in the univariate normal distribution, the maximum occurs at the mean $\mu$.

## 5.2 Recipe for constructing loss functions

The recipe for constructing loss functions for training data $\{x_i, y_i\}$ using the maximum likelihood approach is hence:

1. Choose a suitable probability distribution $Pr(y|\theta)$ defined over the domain of the predictions $y$ with distribution parameters $\theta$.
2. Set the machine learning model $f(x, \phi)$ to predict one or more of these parameters, so $\theta = f(x, \phi)$ and $Pr(y|\theta) = Pr(y|f(x, \phi))$.
3. To train the model, find the network parameters $\phi$ that minimize the negative log-likelihood loss function over the training dataset pairs $\{x_i, y_i\}$:

$$
\hat{\phi} = argmin_{\phi} [ L[\phi] ] = argmin_{\phi} \left[ - \sum_{i=1}^{I} \log Pr(y_i|f(x_i, \phi)) \right] \tag{5.6}
$$

4. To perform inference for a new test example $x$, return either the full distribution $Pr(y|f(x, \phi))$ or the maximum of this distribution.

We devote most of the rest of this chapter to constructing loss functions for common prediction types using this recipe.
## 5.3 Example 1: univariate regression

We start by considering univariate regression models. Here the goal is to predict a single scalar output $y \in \mathbb{R}$ from input $x$ using a model $f(x, \phi)$ with parameters $\phi$. Following the recipe, we choose a probability distribution over the output domain $y$. We select the univariate normal (figure 5.3), which is defined over $y \in \mathbb{R}$. This distribution has two parameters (mean $\mu$ and variance $\sigma^2$) and has a probability density function:

$$
Pr(y|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left[ -\frac{(y - \mu)^2}{2\sigma^2} \right] \tag{5.7}
$$
Second, we set the machine learning model $f(x, \phi)$ to compute one or more of the parameters of this distribution. Here, we just compute the mean so $\mu = f(x, \phi)$:

$$
Pr(y|f(x, \phi), \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left[ -\frac{(y - f(x, \phi))^2}{2\sigma^2} \right] \tag{5.8}
$$
We aim to find the parameters $\phi$ that make the training data $\{x_i, y_i\}$ most probable under this distribution (figure 5.4). To accomplish this, we choose a loss function $L[\phi]$ based on the negative log-likelihood:

$$
L[\phi] = - \sum_{i=1}^{I} \log \left[ Pr(y_i|f(x_i, \phi), \sigma^2) \right]
= - \sum_{i=1}^{I} \log \left[ \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left[ -\frac{(y_i - f(x_i, \phi))^2}{2\sigma^2} \right] \right] \tag{5.9}
$$
When we train the model, we seek parameters $\hat{\phi}$ that minimize this loss.
### 5.3.1 Least squares loss function

Now let’s perform some algebraic manipulations on the loss function. We seek:

$$
\begin{align}
\hat{\phi} &= argmin_{\phi} \left[ -\sum_{i=1}^{I} \log \left[ \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left[ -\frac{(y_i - f(x_i, \phi))^2}{2\sigma^2} \right] \right] \right] \\
&= argmin_{\phi} \left[ -\sum_{i=1}^{I} ( \log \frac{1}{\sqrt{2\pi\sigma^2}} - \frac{(y_i - f(x_i, \phi))^2}{2\sigma^2} ) \right] \\
&= argmin_{\phi} \left[ \sum_{i=1}^{I} \frac{(y_i - f(x_i, \phi))^2}{2\sigma^2} \right] \tag{5.10}
\end{align}
$$
where we have removed the first term between the second and third lines because it does not depend on $\phi$. We have removed the denominator between the third and fourth lines, as this is just a constant scaling factor that does not affect the position of the minimum.

The result of these manipulations is the least squares loss function that we originally introduced when we discussed linear regression in chapter 2:

$$
L[\phi] = \sum_{i=1}^{I} (y_i - f(x_i, \phi))^2 \tag{5.11}
$$
We see that the least squares loss function follows naturally from the assumptions that the prediction errors are (i) independent and (ii) drawn from a normal distribution with mean $\mu = f(x_i, \phi)$ (figure 5.4).

### 5.3.2 Inference

The network no longer directly predicts $y$ but instead predicts the mean $\mu = f(x, \phi)$ of the normal distribution over $y$. When we perform inference, we usually want a single “best” point estimate, so we take the maximum of the predicted distribution:

$$
\hat{y} = \argmax_y [Pr(y|f(x, \phi))] .
$$
(5.12)

For the univariate normal, the maximum position is determined by the mean parameter $\mu$ (figure 5.3). This is precisely what the model computed, so $\hat{y} = f(x, \phi)$.

### 5.3.3 Estimating variance

To formulate the least squares loss function, we assumed that the network predicted the mean of a normal distribution. The final expression in equation 5.11 (perhaps surprisingly) does not depend on the variance σ2. However, there is nothing to stop us from treating σ2 as a parameter of the model and minimizing equation 5.9 with respect to both the model parameters φ and the distribution variance σ2:

### 5.3.4 Heteroscedastic regression

The model above assumes that the variance of the data is constant everywhere. However, this might be unrealistic. When the uncertainty of the model varies as a function of the input data, we refer to this as *heteroscedastic* (as opposed to *homoscedastic*, where the uncertainty is constant).

A simple way to model this is to train a neural network $f(x, \phi)$ that computes both the mean and the variance. For example, consider a shallow network with two outputs. We denote the first output as $f_1(x, \phi)$ and use this to predict the mean, and we denote the second output as $f_2(x, \phi)$ and use it to predict the variance.

There is one complication; the variance must be positive, but we can't guarantee that the network will always produce a positive output. To ensure that the computed variance is positive, we pass the second network output through a function that maps an arbitrary value to a positive one. A suitable choice is the squaring function, giving:

$$
\begin{align}
\mu = f_1(x, \phi) \\
\sigma^2 = f_2(x, \phi)^2 
\end{align}\tag{5.14}
$$
which results in the loss function:

$$
\hat{\phi} = argmin_{\phi} \left[ -\sum_{i=1}^{I} \log \left[ \frac{1}{\sqrt{2\pi f_2(x_i, \phi)^2}} \exp \left[ -\frac{(y_i - f_1(x_i, \phi))^2}{2f_2(x_i, \phi)^2} \right] \right] \right] \tag{5.15}
$$
Homoscedastic and heteroscedastic models are compared in figure 5.5.

## 5.4 Example 2: binary classification

In *binary classification*, the goal is to assign the data $x$ to one of two discrete classes $y \in \{0, 1\}$. In this context, we refer to $y$ as a *label*. Examples of binary classification include (i) predicting whether a restaurant review is positive ($y = 1$) or negative ($y = 0$) from text data $x$ and (ii) predicting whether a tumor is present ($y = 1$) or absent ($y = 0$) from an MRI scan $x$.

Once again, we follow the recipe from section 5.2 to construct the loss function. First, we choose a probability distribution over the output space $y \in \{0, 1\}$. A suitable choice is the Bernoulli distribution, which is defined on the domain $\{0, 1\}$. This has a single parameter $\lambda \in [0, 1]$ that represents the probability that $y$ takes the value one (figure 5.6):

$$
Pr(y|\lambda) = 
\begin{cases} 
1 - \lambda & \text{if } y = 0 \\
\lambda & \text{if } y = 1 
\end{cases} \tag{5.16}
$$

which can equivalently be written as:

$$
Pr(y|\lambda) = (1 - \lambda)^{1-y} \cdot \lambda^y \tag{5.17}
$$
Second, we set the machine learning model $f(x, \phi)$ to predict the single distribution parameter $\lambda$. However, $\lambda$ can only take values in the range [0, 1], and we cannot guarantee that the network output will lie in this range. Consequently, we pass the network output through a function that maps the real numbers $\mathbb{R}$ to [0, 1]. A suitable function is the logistic sigmoid (figure 5.7):

$$
sig[z] = \frac{1}{1 + \exp[-z]} \tag{5.18}
$$
Hence, we predict the distribution parameter as $\lambda = sig[f(x, \phi)]$. The likelihood is now:

$$
Pr(y|x) = (1 - sig[f(x, \phi)])^{1-y} \cdot sig[f(x, \phi)]^y \tag{5.19}
$$
This is depicted in figure 5.8 for a shallow neural network model. The loss function is the negative log-likelihood of the training set:

$$
L[\phi] = \sum_{i=1}^{I} -\left[(1 - y_i) \log [1 - sig[f(x_i, \phi)]] + y_i \log [sig[f(x_i, \phi)]]\right] \tag{5.20}
$$
For reasons to be explained in section 5.7, this is known as the binary cross-entropy loss.

The transformed model output $sig[f(x, \phi)]$ predicts the parameter $\lambda$ of the Bernoulli distribution. This represents the probability that $y = 1$, and it follows that $1 - \lambda$ represents the probability that $y = 0$. When we perform inference, we may want a point estimate of $y$, so we set $y = 1$ if $\lambda > 0.5$ and $y = 0$ otherwise.

## 5.5 Example 3: multiclass classification

The goal of *multiclass classification* is to assign an input data example $x$ to one of $K > 2$ classes, so $y \in \{1, 2, \ldots, K\}$. Real-world examples include (i) predicting which of $K = 10$ digits $y$ is present in an image $x$ of a handwritten number and (ii) predicting which of $K$ possible words $y$ follows an incomplete sentence $x$.

We once more follow the recipe from section 5.2. We first choose a distribution over the prediction space $y$. In this case, we have $y \in \{1, 2, \ldots, K\}$, so we choose the *categorical distribution* (figure 5.9), which is defined on this domain. This has $K$ parameters $\lambda_1, \lambda_2, \ldots, \lambda_K$, which determine the probability of each category:
$$
Pr(y = k) = \lambda_k \tag{5.21}
$$
The parameters are constrained to take values between zero and one, and they must collectively sum to one to ensure a valid probability distribution.

Then we use a network $f(x, \phi)$ with $K$ outputs to compute these $K$ parameters from the input $x$. Unfortunately, the network outputs will not necessarily obey the aforementioned constraints. Consequently, we pass the $K$ outputs of the network through a function that ensures these constraints are respected. A suitable choice is the *softmax* function (figure 5.10). This takes an arbitrary vector of length $K$ and returns a vector of the same length but where the elements are now in the range [0, 1] and sum to one. The $k^{th}$ output of the softmax function is:

$$
softmax_k[z] = \frac{\exp[z_k]}{\sum_{k'=1}^{K} \exp[z_{k'}]} \tag{5.22}
$$
where the exponential functions ensure positivity, and the sum in the denominator ensures that the $K$ numbers sum to one.

The likelihood that input $x$ has label $y$ (figure 5.10) is hence:

$$
Pr(y = k|x) = softmax_k[f(x, \phi)] \tag{5.23}
$$
The loss function is the negative log-likelihood of the training data:

$$
L[\phi] = -\sum_{i=1}^{I} \log \left[ softmax_{y_i} [f(x_i, \phi)] \right]
= -\sum_{i=1}^{I} \left[ f_{y_i}[x_i, \phi] - \log \left( \sum_{k'=1}^{K} \exp [f_{k'}[x_i, \phi]] \right) \right],
$$
(5.24)

where $f_k[x, \phi]$ denotes the $k^{th}$ output of the neural network. For reasons that will be explained in section 5.7, this is known as the *multiclass cross-entropy loss*.

The transformed model output represents a categorical distribution over possible classes $y \in \{1, 2, \ldots, K\}$. For a point estimate, we take the most probable category $\hat{y} = argmax_k[Pr(y = k|f(x, \phi))]$. This corresponds to whichever curve is highest for that value of $x$ in figure 5.10.

### 5.5.1 Predicting other data types

In this chapter, we have focused on regression and classification because these problems are widespread. However, to make different types of predictions, we simply choose an appropriate distribution over that domain and apply the recipe in section 5.2. Figure 5.11 enumerates a series of probability distributions and their prediction domains. Some of these are explored in the problems at the end of the chapter.

## 5.6 Multiple outputs

Often, we wish to make more than one prediction with the same model, so the target output $y$ is a vector. For example, we might want to predict a molecule’s melting and boiling point (a multivariate regression problem), or the object class at every point in an image (a multivariate classification problem). While it is possible to define multivariate probability distributions and use a neural network to model their parameters as a function of the input, it is more usual to treat each prediction as independent.

Independence implies that we treat the probability $Pr(y|f(x, \phi))$ as a product of univariate terms for each element $y_d \in y$:

$$
Pr(y|f(x, \phi)) = \prod_{d} Pr(y_d|f_d[x, \phi]) \tag{5.25}
$$
where $f_d[x, \phi]$ is the $d^{th}$ set of network outputs, which describe the parameters of the distribution over $y_d$. For example, to predict multiple continuous variables $y_d \in \mathbb{R}$, we use a normal distribution for each $y_d$, and the network outputs $f_d[x, \phi]$ predict the means of these distributions. To predict multiple discrete variables $y_d \in {1, 2, \ldots, K}$, we use a categorical distribution for each $y_d$. Here, each set of network outputs $f_d[x, \phi]$ predicts the �K values that contribute to the categorical distribution for $y_d$​.

When we minimize the negative log probability, this product becomes a sum of terms:

$$
L[\phi] = -\sum_{i=1}^{I} \log [Pr(y_i|f(x_i, \phi))] = -\sum_{i=1}^{I} \sum_{d} \log [Pr(y_{id}|f_d[x_i, \phi])] \tag{5.26}
$$

where $y_{id}$is the $d^{th}$ output from the $i^{th}$ training example.

To make two or more prediction types simultaneously, we similarly assume the errors in each are independent. For example, to predict wind direction and strength, we might choose the von Mises distribution (defined on circular domains) for the direction and the exponential distribution (defined on positive real numbers) for the strength. The independence assumption implies that the joint likelihood of the two predictions is the product of individual likelihoods. These terms will become additive when we compute the negative log-likelihood.
### 5.7 Cross-entropy loss

In this chapter, we developed loss functions that minimize negative log-likelihood. However, the term *cross-entropy loss* is also commonplace. In this section, we describe the cross-entropy loss and show that it is equivalent to using negative log-likelihood.

The cross-entropy loss is based on the idea of finding parameters $\theta$ that minimize the distance between the empirical distribution $q(y)$ of the observed data $y$ and a model distribution $Pr(y|\theta)$ (figure 5.12). The distance between two probability distributions $q(z)$ and $p(z)$ can be evaluated using the Kullback-Leibler (KL) divergence:

$$
D_{KL}(q||p) = \int_{-\infty}^{\infty} q(z) \log [q(z)] dz - \int_{-\infty}^{\infty} q(z) \log [p(z)] dz \tag{5.27}
$$
Now consider that we observe an empirical data distribution at points $\{y_i\}^I_{i=1}$. We can describe this as a weighted sum of point masses:

$$
q(y) = \frac{1}{I} \sum_{i=1}^{I} \delta[y - y_i] \tag{5.28}
$$

where $\delta[\cdot]$ is the Dirac delta function. We want to minimize the KL divergence between the model distribution $Pr(y|\theta)$ and this empirical distribution:

$$
\begin{align}
\hat{\theta} &= argmin_{\theta} \left[ \int_{-\infty}^{\infty} q(y) \log [q(y)] dy - \int_{-\infty}^{\infty} q(y) \log [Pr(y|\theta)] dy \right] \\
&= argmin_{\theta} \left[ -\int_{-\infty}^{\infty} q(y) \log [Pr(y|\theta)] dy \right],
\end{align} \tag{5.29}
$$

where the first term disappears, as it has no dependence on $\theta$. The remaining second term is known as the *cross-entropy*. It can be interpreted as the amount of uncertainty that remains in one distribution after taking into account what we already know from the other. Now, we substitute in the definition of $q(y)$ from equation 5.28:

$$
\begin{align}
\hat{\theta} &= argmin_{\theta} \left[ \int_{-\infty}^{\infty} \left( \frac{1}{I} \sum_{i=1}^{I} \delta[y - y_i] \right) \log [Pr(y|\theta)] dy \right] \\
&= argmin_{\theta} \left[ -\sum_{i=1}^{I} \log [Pr(y_i|\theta)] \right],
\end{align} \tag{5.30}
$$

The product of the two terms in the first line corresponds to pointwise multiplying the point masses in figure 5.12a with the logarithm of the distribution in figure 5.12b. We are left with a finite set of weighted probability masses centered on the data points. In the last line, we have eliminated the constant scaling factor $1/I$, as this does not affect the position of the minimum.

In machine learning, the distribution parameters $\theta$ are computed by the model $f[x_i, \phi]$, so we have:

$$
\hat{\phi} = argmin_{\phi} \left[ -\sum_{i=1}^{I} \log [Pr(y_i|f[x_i, \phi])] \right] \tag{5.31}
$$
This is precisely the negative log-likelihood criterion from the recipe in section 5.2. It follows that the negative log-likelihood criterion (from maximizing the data likelihood) and the cross-entropy criterion (from minimizing the distance between the model and empirical data distributions) are equivalent.

## 5.8 Summary

We previously considered neural networks as directly predicting outputs y from data x. In this chapter, we shifted perspective to think about neural networks as computing the parameters $\theta$ of probability distributions $Pr(y|\theta)$ over the output space. This led to a principled approach to building loss functions. We selected model parameters $\phi$ that maximized the likelihood of the observed data under these distributions. We saw that this is equivalent to minimizing the negative log-likelihood.

The least squares criterion for regression is a natural consequence of this approach; it follows from the assumption that y is normally distributed and that we are predicting the mean. We also saw how the regression model could be (i) extended to estimate the uncertainty over the prediction and (ii) extended to make that uncertainty dependent on the input (the heteroscedastic model). We applied the same approach to both binary and multiclass classification and derived loss functions for each. We discussed how to tackle more complex data types and how to deal with multiple outputs. Finally, we argued that cross-entropy is an equivalent way to think about fitting models.

In previous chapters, we developed neural network models. In this chapter, we de- veloped loss functions for deciding how well a model describes the training data for a given set of parameters. The next chapter considers model training, in which we aim to find the model parameters that minimize this loss.

## Notes

Losses based on the normal distribution: Nix & Weigend (1994) and Williams (1996) investigated heteroscedastic nonlinear regression in which both the mean and the variance of the output are functions of the input. In the context of unsupervised learning, Burda et al. (2016) use a loss function based on a multivariate normal distribution with diagonal covariance, and Dorta et al. (2018) use a loss function based on a normal distribution with full covariance.

Robust regression: Qi et al. (2020) investigate the properties of regression models that min- imize mean absolute error rather than mean squared error. This loss function follows from assuming a Laplace distribution over the outputs and estimates the median output for a given input rather than the mean. Barron (2019) presents a loss function that parameterizes the de- gree of robustness. When interpreted in a probabilistic context, it yields a family of univariate probability distributions that includes the normal and Cauchy distributions as special cases.

Estimating quantiles: Sometimes, we may not want to estimate the mean or median in a regression task but may instead want to predict a quantile. For example, this is useful for risk models, where we want to know that the true value will be less than the predicted value 90% of the time. This is known as quantile regression (Koenker & Hallock, 2001). This could be done by fitting a heteroscedastic regression model and then estimating the quantile based on the predicted normal distribution. Alternatively, the quantiles can be estimated directly using quantile loss (also known as pinball loss). In practice, this minimizes the absolute deviations of the data from the model but weights the deviations in one direction more than the other. Recent work has investigated simultaneously predicting multiple quantiles to get an idea of the overall distribution shape (Rodrigues & Pereira, 2020).

Class imbalance and focal loss: Lin et al. (2017c) address data imbalance in classification problems. If the number of examples for some classes is much greater than for others, then the standard maximum likelihood loss does not work well; the model may concentrate on becoming more confident about well-classified examples from the dominant classes and classify less well- represented classes poorly. Lin et al. (2017c) introduce focal loss, which adds a single extra parameter that down-weights the effect of well-classified examples to improve performance.

Learning to rank: Cao et al. (2007), Xia et al. (2008), and Chen et al. (2009) all used the Plackett-Luce model in loss functions for learning to rank data. This is the listwise approach to learning to rank as the model ingests an entire list of objects to be ranked at once. Alternative approaches are the pointwise approach, in which the model ingests a single object, and the pairwise approach, where the model ingests pairs of objects. Chen et al. (2009) summarize different approaches for learning to rank.

Other data types: Fan et al. (2020) use a loss based on the beta distribution for predicting values between zero and one. Jacobs et al. (1991) and Bishop (1994) investigated mixture density networks for multimodal data. These model the output as a mixture of Gaussians (see figure 5.14) that is conditional on the input. Prokudin et al. (2018) used the von Mises distribution to predict direction (see figure 5.13). Fallah et al. (2009) constructed loss functions for prediction counts using the Poisson distribution (see figure 5.15). Ng et al. (2017) used loss functions based on the gamma distribution to predict duration.

Non-probabilistic approaches: It is not strictly necessary to adopt the probabilistic ap- proach discussed in this chapter, but this has become the default in recent years; any loss func- tion that aims to reduce the distance between the model output and the training outputs will suﬀice, and distance can be defined in any way that seems sensible. There are several well-known non-probabilistic machine learning models for classification, including support vector machines (Vapnik, 1995; Cristianini & Shawe-Taylor, 2000), which use hinge loss, and AdaBoost (Freund & Schapire, 1997), which uses exponential loss.

## Problems

**Problem 5.1** Show that the logistic sigmoid function $\text{sig}[z]$ maps $z = -\infty$ to 0, $z = 0$ to 0.5 and $z = \infty$ to 1 where:

$$
\text{sig}[z] = \frac{1}{1 + \exp[-z]} \tag{5.32}
$$

**Problem 5.2** The loss $L$ for binary classification for a single training pair $\{x, y\}$ is:

$$
L = -(1 - y) \log [1 - \text{sig}[f[x, \phi]]] - y \log [\text{sig}[f[x, \phi]]] \tag{5.33}
$$
where $\text{sig}[\cdot]$ is defined in equation 5.32. Plot this loss as a function of the transformed network output $\text{sig}[f[x, \phi]] \in [0, 1]$ (i) when the training label $y = 0$ and (ii) when $y = 1$.

**Problem 5.3*** Suppose we want to build a model that predicts the direction $y$ in radians of the prevailing wind based on local measurements of barometric pressure $x$. A suitable distribution over circular domains is the von Mises distribution (figure 5.13):

$$
Pr(y|\mu, \kappa) = \frac{\exp[\kappa \cos(y - \mu)]}{2\pi \cdot \text{Bessel}_0[\kappa]} \tag{5.34}
$$
where $\mu$ is a measure of the mean direction and $\kappa$ is a measure of the concentration (i.e., the inverse of the variance). The term $\text{Bessel}_0[\kappa]$ is a modified Bessel function of order 0. Use the recipe from section 5.2 to develop a loss function for learning the parameter $\mu$ of a model $f[x, \phi]$ to predict the most likely wind direction. Your solution should treat the concentration $\kappa$ as constant. How would you perform inference?

**Problem 5.4*** Sometimes, the outputs $y$ for input $x$ are multimodal (figure 5.14a); there is more than one valid prediction for a given input. Here, we might use a weighted sum of normal components as the distribution over the output. This is known as a *mixture of Gaussians* model. For example, a mixture of two Gaussians has parameters $\Theta = \{\lambda, \mu_1, \sigma_1^2, \mu_2, \sigma_2^2\}$:

$$
Pr(y|\mu_1, \mu_2, \sigma_1^2, \sigma_2^2) = \frac{\lambda}{\sqrt{2\pi\sigma_1^2}} \exp \left[ -\frac{(y - \mu_1)^2}{2\sigma_1^2} \right] + \frac{1 - \lambda}{\sqrt{2\pi\sigma_2^2}} \exp \left[ -\frac{(y - \mu_2)^2}{2\sigma_2^2} \right] \tag{5.35}
$$
where $\lambda \in [0, 1]$ controls the relative weight of the two components, which have means $\mu_1, \mu_2$ and variances $\sigma_1^2, \sigma_2^2$, respectively. This model can represent a distribution with two peaks (figure 5.14b) or a distribution with one peak but a more complex shape (figure 5.14c). Use the recipe from section 5.2 to construct a loss function for training a model $f[x, \phi]$ that takes input $x$, has parameters $\phi$, and predicts a mixture of two Gaussians. The loss should be based on $I$ training data pairs $\{x_i, y_i\}$. What problems do you foresee when performing inference?

**Problem 5.5** Consider extending the model from problem 5.3 to predict the wind direction using a mixture of two von Mises distributions. Write an expression for the likelihood $Pr(y|\theta)$ for this model. How many outputs will the network need to produce?

**Problem 5.6** Consider building a model to predict the number of pedestrians $y \in \{0, 1, 2, \ldots\}$ that will pass a given point in the city in the next minute, based on data $x$ that contains information about the time of day, the longitude and latitude, and the type of neighborhood. A suitable distribution for modeling counts is the Poisson distribution (figure 5.15). This has a single parameter $\lambda > 0$ called the rate that represents the mean of the distribution. The distribution has probability density function:

$$
Pr(y = k) = \frac{\lambda^k e^{-\lambda}}{k!} \tag{5.36}
$$
Design a loss function for this model assuming we have access to $I$ training pairs $\{x_i, y_i\}$.

**Problem 5.7** Consider a multivariate regression problem where we predict ten outputs, so $y \in \mathbb{R}^{10}$, and model each with an independent normal distribution where the means $\mu_d$ are predicted by the network, and variances $\sigma^2$ are constant. Write an expression for the likelihood $Pr(y|f[x, \phi])$. Show that minimizing the negative log-likelihood of this model is still equivalent to minimizing a sum of squared terms if we don’t estimate the variance $\sigma^2$.

**Problem 5.8*** Construct a loss function for making multivariate predictions $y \in \mathbb{R}^D_i$ based on independent normal distributions with different variances $\sigma_d^2$ for each dimension. Assume a heteroscedastic model so that both the means $\mu_d$ and variances $\sigma_d^2$ vary as a function of the data.

**Problem 5.9*** Consider a multivariate regression problem in which we predict the height of a person in meters and their weight in kilos from data $x$. Here, the units take quite different ranges. What problems do you see this causing? Propose two solutions to these problems.

**Problem 5.10** Extend the model from problem 5.3 to predict both the wind direction and the wind speed and define the associated loss function.
