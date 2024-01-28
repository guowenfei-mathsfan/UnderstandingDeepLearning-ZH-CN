前三章分别介绍了线性回归、浅层神经网络和深度神经网络。这些都属于函数家族，能够实现从输入到输出的映射，其具体的函数取决于模型参数 $\phi$。在训练这些模型时，我们的目标是找到能够为特定任务提供最优输入输出映射的参数。本章将详细阐述“最优映射”的含义。

要定义“最优映射”，首先需要一组训练数据集 $\{x_i, y_i\}$，即输入和输出的配对。损失函数（Loss Function）$L[\phi]$ 能够返回一个数值，这个数值描述了模型预测 $f(x_i, \phi)$ 与其对应的真实输出 $y_i$ 之间的不匹配程度。在训练过程中，我们追求的是能最小化损失的参数值 $\phi$，以使训练输入尽可能准确地映射到输出。例如，在第2章中，我们见到了一种损失函数——最小平方损失函数，适用于目标是实数 $y \in \mathbb{R}$ 的单变量回归问题。该函数通过计算模型预测 $f(x_i, \phi)$ 与真实值 $y_i$ 之间差异的平方和来进行计算。

本章还提出了一个框架，不仅证明了在实值输出场景下选择最小平方准则的适用性，还指导我们为其他类型的预测问题构建损失函数。我们将讨论包括二元分类（其中预测结果 $y \in \{0, 1\}$ 属于两个类别中的一个）和多类别分类（预测结果 $y \in \{1, 2, \ldots, K\}$ 属于 $K$ 个类别中的一个）在内的多种情形。在接下来的两章中，我们将探讨模型训练的过程，目标是找到能最小化这些损失函数的参数值。
## 5.1 最大似然
在本节中，我们将介绍构建损失函数的具体方法。设想一个计算输入 $x$ 到输出的模型 $f(x, \phi)$，其中 $\phi$ 是模型的参数。之前，我们认为模型直接输出预测结果 $y$。现在，我们改变思路，将模型视为计算给定输入 $x$ 时，可能的输出 $y$ 的条件概率分布 $Pr(y|x)$。这种损失函数的设计目的是使得每个训练输出 $y_i$ 在由对应输入 $x_i$ 计算得到的分布 $Pr(y_i|x_i)$ 中具有较高的概率（见图 5.1）。
#### 5.1.1 计算输出的分布
这引出了一个问题：模型 $f(x, \phi)$ 如何转化为计算概率分布的形式。答案很简单。首先，我们需要选定一个定义在输出域 $Y$ 上的参数化概率分布 $Pr(y|\theta)$。接着，我们利用神经网络来计算该分布的一个或多个参数 $\theta$。

例如，假设预测域是实数集，即 $y \in \mathbb{R}$。在这种情况下，我们可能选择单变量正态分布，它在 $\mathbb{R}$ 上有定义。该分布由均值 $\mu$ 和方差 $\sigma^2$ 所决定，因此 $\theta = \{\mu, \sigma^2\}$。机器学习模型可以用来预测均值 $\mu$，而方差 $\sigma^2$ 则可以视为一个待定的常数。
#### 5.1.2 最大似然准则
模型现在针对每个训练输入 $x_i$ 计算不同的分布参数 $\theta_i = f(x_i, \phi)$。我们的目标是使每个训练输出 $y_i$ 在其相应的分布 $Pr(y_i|\theta_i)$ 下具有较高概率。因此，我们选择模型参数 $\phi$，以最大化所有 $I$ 个训练样本的联合概率：
$$
\begin{align}
\hat{\phi} &= argmax_{\phi} \left[ \prod_{i=1}^{I} Pr(y_i|x_i) \right] \\
&= argmax_{\phi} \left[ \prod_{i=1}^{I} Pr(y_i|\theta_i) \right] \\
&= argmax_{\phi} \left[ \prod_{i=1}^{I} Pr(y_i|f(x_i, \phi)) \right]  \\
\end{align} \tag{5.1}
$$

这个联合概率项反映的是参数的似然（Likelihood），因此方程 5.1 称为最大似然准则（Maximum Likelihood Criterion）[^1]。

这里我们基于两个假设。首先，我们假设所有数据点的输出 $y_i$ 都服从相同的概率分布，即数据是同分布的。其次，我们认为给定输入的输出的条件分布 $Pr(y_i|x_i)$ 是相互独立的，因此整个训练数据集的似然可以表示为：

$$
Pr(y_1, y_2, \ldots , y_I|x_1, x_2, \ldots , x_I) = \prod_{i=1}^{I} Pr(y_i|x_i) \tag{5.2}
$$

换言之，我们假定数据是独立同分布（i.i.d.）的。

#### 5.1.3 最大化对数似然
尽管最大似然准则（方程 5.1）理论上有效，但在实际应用中并不方便。每个项 $Pr(y_i|f(x_i, \phi))$ 的值可能很小，导致这些项的乘积极小，难以用有限精度算法精确表示。幸运的是，我们可以通过最大化似然的对数来解决这个问题：

$$
\begin{align}
\hat{\phi} &= argmax_{\phi} \left[ \prod_{i=1}^{I} Pr(y_i|f(x_i, \phi)) \right] \\
&= argmax_{\phi} \left[ \log \prod_{i=1}^{I} Pr(y_i|f(x_i, \phi)) \right] \\
&= argmax_{\phi} \left[ \sum_{i=1}^{I} \log Pr(y_i|f(x_i, \phi)) \right] 
\end{align} \tag{5.3}
$$

由于对数是单调递增函数，对数似然准则与原始最大似然准则在数学上是等价的。这意味着，提高对数似然准则的同时，也就提高了最大似然准则。因此，两种准则的最大值位置是相同的，最优的模型参数 $\hat{\phi}$ 在两种情况下都是一致的。同时，对数似然准则通过求和而非乘积，避免了精度问题。
#### 5.1.4 最小化负对数似然

通常，模型拟合问题是以最小化损失的方式来定义的。为了将最大对数似然准则转换为一个最小化问题，我们通过乘以负一得到负对数似然准则：

$$
\hat{\phi} = argmin_{\phi} \left[ - \sum_{i=1}^{I} \log Pr(y_i|f(x_i, \phi)) \right]
= argmin_{\phi} [ L[\phi] ] \tag{5.4}
$$
这就构成了最终的损失函数 $L[\phi]$。

#### 5.1.5 推断

如今，网络不再直接预测输出 $y$，而是确定了一个关于 $y$ 的概率分布。在进行推断时，我们一般需要一个具体的估计值而不是整个分布，因此我们选择分布的最大值作为预测：

$$
\hat{y} = argmax_y [Pr(y|f(x, \phi))]  \tag{5.5}
$$
(5.5)

我们通常可以根据模型预测的分布参数 $\theta$ 来确定这个估计值。例如，在单变量正态分布中，最大值出现在均值 $\mu$ 处。


## 5.2 构建损失函数的步骤

根据最大似然方法，针对训练数据 $\{x_i, y_i\}$ 构建损失函数的步骤如下：

1. 选定一个适合预测结果 $y$ 的概率分布 $Pr(y|\theta)$，并确定其分布参数 $\theta$。
2. 设定机器学习模型 $f(x, \phi)$ 来预测这些参数中的一个或多个，即 $\theta = f(x, \phi)$，$Pr(y|\theta) = Pr(y|f(x, \phi))$。
3. 为训练模型，寻找最小化负对数似然损失函数的模型参数 $\phi$：

$$
\hat{\phi} = argmin_{\phi} [ L[\phi] ] = argmin_{\phi} \left[ - \sum_{i=1}^{I} \log Pr(y_i|f(x_i, \phi)) \right] \tag{5.6}
$$

4. 对于新的测试样例 $x$，返回完整分布 $Pr(y|f(x, \phi))$ 或此分布的最大值。

本章其余部分主要讨论如何使用这种方法为常见的预测类型构建损失函数。
## 5.3 示例 1：单变量回归

首先考虑单变量回归模型。这里的目标是用带有参数 $\phi$ 的模型 $f(x, \phi)$，从输入 $x$ 预测单一实数输出 $y \in \mathbb{R}$。遵循上述步骤，我们为输出域 $y$ 选择一个概率分布。我们选用单变量正态分布（见图 5.3），它定义在 $y \in \mathbb{R}$ 上。该分布有两个参数（均值 $\mu$ 和方差 $\sigma^2$），并具有概率密度函数：

$$
Pr(y|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left[ -\frac{(y - \mu)^2}{2\sigma^2} \right] \tag{5.7}
$$
接着，我们让机器学习模型 $f(x, \phi)$ 计算这个分布的一个或多个参数。在这里，我们只计算均值 $\mu = f(x, \phi)$：

$$
Pr(y|f(x, \phi), \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left[ -\frac{(y - f(x, \phi))^2}{2\sigma^2} \right] \tag{5.8}
$$
我们的目标是找到使训练数据 $\{x_i, y_i\}$ 在此分布下尽可能概率最高的参数 $\phi$（参见图 5.4）。为此，我们选择了基于负对数似然的损失函数 $L[\phi]$：

$$
L[\phi] = - \sum_{i=1}^{I} \log \left[ Pr(y_i|f(x_i, \phi), \sigma^2) \right]
= - \sum_{i=1}^{I} \log \left[ \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left[ -\frac{(y_i - f(x_i, \phi))^2}{2\sigma^2} \right] \right] \tag{5.9}
$$
在训练模型时，我们的目标是找到最小化这一损失的参数 $\hat{\phi}$。
#### 5.3.1 最小平方损失函数

我们对损失函数进行一系列代数操作，目的是寻找：

$$
\begin{align}
\hat{\phi} &= argmin_{\phi} \left[ -\sum_{i=1}^{I} \log \left[ \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left[ -\frac{(y_i - f(x_i, \phi))^2}{2\sigma^2} \right] \right] \right] \\
&= argmin_{\phi} \left[ -\sum_{i=1}^{I} ( \log \frac{1}{\sqrt{2\pi\sigma^2}} - \frac{(y_i - f(x_i, \phi))^2}{2\sigma^2} ) \right] \\
&= argmin_{\phi} \left[ \sum_{i=1}^{I} \frac{(y_i - f(x_i, \phi))^2}{2\sigma^2} \right] \tag{5.10}
\end{align}
$$
在这里，我们去除了与 $\phi$ 无关的项，并忽略了常数缩放因子，因为它不影响最小值的位置。

通过这些操作，我们得到了最小平方损失函数，这是我们在第2章讨论线性回归时首次提出的：

$$
L[\phi] = \sum_{i=1}^{I} (y_i - f(x_i, \phi))^2 \tag{5.11}
$$
最小平方损失函数的自然来源于两个假设：预测误差（i）是独立的，并且（ii）遵循均值为 $\mu = f(x_i, \phi)$ 的正态分布（参见图 5.4）。

#### 5.3.2 推断

网络现在不直接预测 $y$，而是预测 $y$ 的正态分布均值 $\mu = f(x, \phi)$。在进行推断时，我们通常寻求一个最佳的单点估计，因此我们选择预测分布的最大值：

$$
\hat{y} = \argmax_y [Pr(y|f(x, \phi))] .
$$
(5.12)

在单变量正态分布中，最大值位置由均值参数 $\mu$ 决定（参见图 5.3）。这正是模型所计算的，因此 $\hat{y} = f(x, \phi)$。
#### 5.3.3 估计方差

在制定最小平方损失函数时，我们假定网络预测了正态分布的均值。有趣的是，方程 5.11 中的最终表达式并不依赖于方差 $\sigma^2$。但我们可以将 $\sigma^2$ 视为模型的参数之一，并对模型参数 $\phi$ 和分布的方差 $\sigma^2$ 一起最小化方程 5.9：


5.13公式todo

在推断阶段，模型从输入中预测均值 $\mu = f[x, \hat{\phi}]$，同时我们在训练过程中得到了方差 $\hat{\sigma}^2$ 的估计。均值是最优预测，而方差反映了预测的不确定性。

#### 5.3.4 异方差回归

先前的模型假定数据方差是固定的，但这可能不太现实。当模型的不确定性随输入数据变化时，我们称之为异方差（与同方差相对，后者不确定性是固定的）。

一种处理这种情况的简单方法是训练一个神经网络 $f(x, \phi)$ 来同时计算均值和方差。举个例子，考虑一个输出两个值的浅层网络，其中第一个输出 $f_1(x, \phi)$ 预测均值，第二个输出 $f_2(x, \phi)$ 预测方差。

为了确保计算的方差始终为正，我们需要对网络的第二个输出应用一个能映射到正数的函数。一个好的选择是使用平方函数，得到：

$$
\begin{align}
\mu = f_1(x, \phi) \\
\sigma^2 = f_2(x, \phi)^2 
\end{align}\tag{5.14}
$$
这样就得到了以下损失函数：

$$
\hat{\phi} = argmin_{\phi} \left[ -\sum_{i=1}^{I} \log \left[ \frac{1}{\sqrt{2\pi f_2(x_i, \phi)^2}} \exp \left[ -\frac{(y_i - f_1(x_i, \phi))^2}{2f_2(x_i, \phi)^2} \right] \right] \right] \tag{5.15}
$$
图 5.5 对比了同方差和异方差模型。

## 5.4 示例 2：二元分类

在二元分类任务中，我们的目标是根据数据 $x$ 将其划分为两个离散类别之一 $y \in \{0, 1\}$。这里的 $y$ 被称为标签。二元分类的例子包括：（i）根据文本数据 $x$ 判断餐厅评论是正面（$y = 1$）还是负面（$y = 0$）；（ii）根据 MRI 扫描 $x$ 判断肿瘤是否存在（$y = 1$）或不存在（$y = 0$）。

我们再次按照第5.2节的步骤构建损失函数。首先，我们为输出空间 $y \in \{0, 1\}$ 选择了伯努利分布，这个分布定义在 $\{0, 1\}$ 上。它有一个参数 $\lambda \in [0, 1]$，表示 $y$ 取值为 1 的概率（见图 5.6）：

$$
Pr(y|\lambda) = 
\begin{cases} 
1 - \lambda & \text{if } y = 0 \\
\lambda & \text{if } y = 1 
\end{cases} \tag{5.16}
$$

也可以写成：

$$
Pr(y|\lambda) = (1 - \lambda)^{1-y} \cdot \lambda^y \tag{5.17}
$$
然后，我们设置机器学习模型 $f(x, \phi)$ 来预测单一参数 $\lambda$。但由于 $\lambda$ 只能在 [0, 1] 范围内取值，我们需要通过一个函数将网络输出映射到这个范围内。一个合适的函数是逻辑斯蒂 sigmoid 函数（见图 5.7）：

$$
sig[z] = \frac{1}{1 + \exp[-z]} \tag{5.18}
$$
因此，我们预测的分布参数为 $\lambda = sig[f(x, \phi)]$。现在的似然表达式为：

$$
Pr(y|x) = (1 - sig[f(x, \phi)])^{1-y} \cdot sig[f(x, \phi)]^y \tag{5.19}
$$
这在图 5.8 中展示了一个浅层神经网络模型。损失函数是训练集的负对数似然：

$$
L[\phi] = \sum_{i=1}^{I} -\left[(1 - y_i) \log [1 - sig[f(x_i, \phi)]] + y_i \log [sig[f(x_i, \phi)]]\right] \tag{5.20}
$$
由于第5.7节将会解释的原因，这称为二元交叉熵损失。

变换后的模型输出 $sig[f(x, \phi)]$ 预测了伯努利分布的参数 $\lambda$。这代表 $y = 1$ 的概率，所以 $1 - \lambda$ 代表 $y = 0$ 的概率。在进行推断时，如果我们需要 $y$ 的具体估计，那么当 $\lambda > 0.5$ 时我们设定 $y = 1$，否则设定 $y = 0$。

## 5.5 示例 3：多类别分类

多类别分类的目标是将输入数据 $x$ 分配给 $K > 2$ 个类别中的一个，即 $y \in \{1, 2, \ldots, K\}$。现实中的例子包括：（i）预测手写数字图像 $x$ 中的哪一个数字 $y$（$K = 10$）；（ii）预测不完整句子 $x$ 后面跟随的哪一个词汇 $y$（$K$ 个可能词汇）。

我们再次遵循第5.2节的步骤。首先，对于输出空间 $y \in \{1, 2, \ldots, K\}$，我们选择分类分布（见图 5.9）。这个分布有 $K$ 个参数 $\lambda_1, \lambda_2, \ldots, \lambda_K$，它们确定每个类别的概率：
$$
Pr(y = k) = \lambda_k \tag{5.21}
$$
参数被限制在零和一之间，并且总和必须为一，以形成有效的概率分布。

然后，我们利用具有 $K$ 个输出的网络 $f(x, \phi)$ 来从输入 $x$ 计算这 $K$ 个参数。为了确保网络输出符合约束，我们通过一个函数处理这 $K$ 个输出，这个函数是*softmax*函数（见图 5.10）。softmax 函数接受长度为 $K$ 的任意向量，并返回一个同样长度的向量，其元素位于 [0, 1] 范围内且总和为一。softmax 函数的第 $k$ 个输出是：

$$
softmax_k[z] = \frac{\exp[z_k]}{\sum_{k'=1}^{K} \exp[z_{k'}]} \tag{5.22}
$$
指数函数确保输出为正，分母的求和则保证这 $K$ 个数的总和为一。

因此，输入 $x$ 有标签 $y$ 的似然（见图 5.10）是：

$$
Pr(y = k|x) = softmax_k[f(x, \phi)] \tag{5.23}
$$
损失函数是训练数据的负对数似然：

$$
L[\phi] = -\sum_{i=1}^{I} \log \left[ softmax_{y_i} [f(x_i, \phi)] \right]
= -\sum_{i=1}^{I} \left[ f_{y_i}[x_i, \phi] - \log \left( \sum_{k'=1}^{K} \exp [f_{k'}[x_i, \phi]] \right) \right],
$$
(5.24)

其中 $f_k[x, \phi]$ 是神经网络的第 $k$ 个输出。由于将在第5.7节中解释的原因，这被称为多类别交叉熵损失。

模型输出的变换代表了 $y \in \{1, 2, \ldots, K\}$ 可能类别的分类分布。作为点估计，我们选择最可能的类别 $\hat{y} = argmax_k[Pr(y = k|f(x, \phi))]$，这对应于图 5.10 中对于该 $x$ 值最高的曲线。

### 5.5.1 预测其他数据类型

本章主要关注回归和分类，因为这些问题非常普遍。然而，为了预测不同类型的数据，我们只需选择适合该领域的分布，并应用第5.2节中的方法。图 5.11 列出了一系列概率分布及其预测领域。其中一些将在本章末尾的问题中进行探讨。
## 5.6 多输出预测

在许多情况下，我们需要使用同一个模型进行多个预测，因此目标输出 $y$ 是向量形式。例如，我们可能想同时预测分子的熔点和沸点（多变量回归问题），或者预测图像中每个点的物体类别（多变量分类问题）。虽然可以定义多变量概率分布，并利用神经网络模拟它们作为输入的函数参数，但更常见的做法是将每个预测视为独立的。

独立性意味着我们把概率 $Pr(y|f(x, \phi))$ 看作是对于每个元素 $y_d \in y$ 的单变量项的乘积：

$$
Pr(y|f(x, \phi)) = \prod_{d} Pr(y_d|f_d[x, \phi]) \tag{5.25}
$$
其中 $f_d[x, \phi]$ 是网络对于 $y_d$ 分布参数的第 $d$ 组输出。例如，对于预测多个连续变量 $y_d \in \mathbb{R}$，我们对每个 $y_d$ 使用正态分布，并由网络输出 $f_d[x, \phi]$ 预测这些分布的均值。对于预测多个离散变量 $y_d \in \{1, 2, \ldots, K\}$，我们对每个 $y_d$ 使用分类分布。在这种情况下，每组网络输出 $f_d[x, \phi]$ 预测对 $y_d$ 分类分布的贡献值。

最小化负对数概率时，这个乘积变为各项的求和：

$$
L[\phi] = -\sum_{i=1}^{I} \log [Pr(y_i|f(x_i, \phi))] = -\sum_{i=1}^{I} \sum_{d} \log [Pr(y_{id}|f_d[x_i, \phi])] \tag{5.26}
$$

其中 $y_{id}$ 是第 $i$ 个训练样本的第 $d$ 个输出。

为了同时进行两种或更多类型的预测，我们同样假设每种错误是独立的。比如，为了同时预测风向和风力，我们可能分别选择定义在圆形域的 von Mises 分布预测风向，以及定义在正实数上的指数分布预测风力。独立性假设意味着这两个预测的联合似然是单独似然的乘积。在计算负对数似然时，这些项会转化为加和形式。

## 5.7 Cross-entropy loss

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
