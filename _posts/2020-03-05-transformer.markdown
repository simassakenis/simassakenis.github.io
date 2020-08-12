---
layout: post
title:  "The Formulated Transfomer"
date:   2020-08-05 15:44:56 +0300
---

Motivation Motivation Motivation: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

Short, self-contained, complete.
$$ \newcommand{\bm}{\boldsymbol} $$


&nbsp;
### **Definition**

The Transformer is an encoder-decoder model for sequence-to-sequence tasks. That is, it takes a sequence of symbols as input, produces a sequence of continuous representations of those symbols using the encoder, and uses those representations to generate the output sequence one symbol at a time via the decoder. When generating the next symbol, the decoder has to receive not only the representations from the encoder but also the symbols it has already generated. Hence, the Transformer model actually always takes two sequences as input: the original input sequence and the unfinished output sequence. All symbols in these sequences are represented by their IDs in the specified vocabulary. The Transformer can thus be defined as a function

$$ \text{Transformer} : \{ 1, 2, \ldots, s \}^n \times \{ 1, 2, \ldots, s \}^m \rightarrow [0, 1]^{m \times s} $$

where

$$ \text{Transformer}(\bm{x}, \bm{y})_{i, j} = P(y_{i+1} = j \mid \bm{x}, y_1, \ldots, y_i). $$

Here, $$ n $$ is the length of the input sequence, $$ m $$ is the length of the unfinished output sequence, and $$ s $$ is the number of items in the vocabulary. In words, given sequences $$ \bm{x} $$ and $$ \bm{y} $$, the Transformer outputs a matrix whose rows are probability mass functions $$ P(y_2 = j \mid \bm{x}, y_1) $$, $$ P(y_3 = j \mid \bm{x}, y_1, y_2) $$, ..., $$ P(y_{m+1} = j \mid \bm{x}, y_1, \ldots, y_m) $$, where $$ j \in \{ 1, 2, \ldots, s \} $$.


&nbsp;
### **Modules**

- **One-Hot Encoding**: A function

    $$ \text{OneHot} : \{ 1, 2, \ldots, s \}^n \rightarrow \{ 0, 1 \}^{n \times d_\text{model}} $$

    defined by

    $$ \text{OneHot}(\bm{x})_{i,j} = \begin{cases} 1 & \text{if } j = x_i, \\ 0 & \text{otherwise}. \end{cases} $$

- **Softmax**: A function

    $$ \text{Softmax} : \mathbb{R}^{n \times d_\text{model}} \rightarrow [0, 1]^{n \times d_\text{model}} $$

    defined by

    $$ \text{Softmax}(\bm{X})_{i,j} = \frac{e^{X_{i,j}}}{\sum_{k=1}^{d_\text{model}} e^{X_{i,k}}}. $$

- **Scaled Dot-Product Attention**: A function

    $$ \text{Attention} : \mathbb{R}^{n \times d_k} \times \mathbb{R}^{p \times d_k} \times \mathbb{R}^{p \times d_v} \rightarrow \mathbb{R}^{n \times d_v} $$

    defined by

    $$ \text{Attention}(\bm{Q}, \bm{K}, \bm{V}) = \text{Softmax}\left( \frac{\bm{Q} \bm{K}^\top}{\sqrt{d_k}} \right) \bm{V}. $$

- **Autoregressive Attention Mask**: A function

    $$ \text{Mask} : \mathbb{R}^{n \times n} \rightarrow \mathbb{R}^{n \times n} $$

    defined by

    $$ \text{Mask}(\bm{X})_{i,j} = \begin{cases} -\infty & \text{if } j > i, \\ X_{i,j} & \text{otherwise}. \end{cases} $$

    Illustration:

    $$ \text{Mask}\left( \begin{bmatrix} \cdot & \cdot & \cdot & \cdot \\ \cdot & \cdot & \cdot & \cdot \\ \cdot & \cdot & \cdot & \cdot \\ \cdot & \cdot & \cdot & \cdot \end{bmatrix} \right) = \begin{bmatrix} \cdot & -\infty & -\infty & -\infty \\ \cdot & \cdot & -\infty & -\infty \\ \cdot & \cdot & \cdot & -\infty \\ \cdot & \cdot & \cdot & \cdot \end{bmatrix}. $$

- **Masked Scaled Dot-Product Attention**: A function

    $$ \text{MaskedAttention} : \mathbb{R}^{n \times d_k} \times \mathbb{R}^{n \times d_k} \times \mathbb{R}^{n \times d_v} \rightarrow \mathbb{R}^{n \times d_v} $$

    defined by

    $$ \text{MaskedAttention}(\bm{Q}, \bm{K}, \bm{V}) = \text{Softmax}\left( \frac{\text{Mask}\left(\bm{Q} \bm{K}^\top\right)}{\sqrt{d_k}} \right) \bm{V}. $$

- **Concatenation**: A function

    $$ \text{Concat} : \mathbb{R}^{n \times d_v} \times \cdots \times \mathbb{R}^{n \times d_v} \rightarrow \mathbb{R}^{n \times h d_v} $$

    defined by

    $$ \text{Concat}(\bm{X}_1, \ldots, \bm{X}_h)_{:, (k-1) d_v + 1: k d_v} = \bm{X}_k \quad (k = 1, 2, \ldots, h). $$

    Illustration:

    $$ \text{Concat}\left( \begin{bmatrix} \cdot & \cdot \\ \cdot & \cdot \end{bmatrix}, \begin{bmatrix} \cdot & \cdot \\ \cdot & \cdot \end{bmatrix}, \begin{bmatrix} \cdot & \cdot \\ \cdot & \cdot \end{bmatrix} \right) = \begin{bmatrix} \cdot & \cdot & \cdot & \cdot & \cdot & \cdot \\ \cdot & \cdot & \cdot & \cdot & \cdot & \cdot \end{bmatrix}. $$

- **Multi-Head Attention**: A function

    $$ \text{MultiHead} : \mathbb{R}^{n \times d_\text{model}} \times \mathbb{R}^{n \times d_\text{model}} \times \mathbb{R}^{n \times d_\text{model}} \rightarrow \mathbb{R}^{n \times d_\text{model}} $$

    defined by

    $$ \text{MultiHead}(\bm{Q}, \bm{K}, \bm{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \bm{W}^{O}, \quad \text{where} $$

    $$ \text{head}_i = \text{Attention}(\bm{Q} \bm{W}_i^{Q}, \bm{K} \bm{W}_i^{K}, \bm{V} \bm{W}_i^{V}), $$

    $$ \bm{W}_i^{Q} \in \mathbb{R}^{d_\text{model} \times d_k}, \quad \bm{W}_i^{K} \in \mathbb{R}^{d_\text{model} \times d_k}, \quad \bm{W}_i^{V} \in \mathbb{R}^{d_\text{model} \times d_v}, \quad \bm{W}^{O} \in \mathbb{R}^{h d_v \times d_\text{model}}. $$

- **Masked Multi-Head Attention**: A function

    $$ \text{MaskedMultiHead} : \mathbb{R}^{n \times d_\text{model}} \times \mathbb{R}^{n \times d_\text{model}} \times \mathbb{R}^{n \times d_\text{model}} \rightarrow \mathbb{R}^{n \times d_\text{model}} $$

    defined in the same way as $$ \text{MultiHead} $$ except that

    $$ \text{head}_i = \text{MaskedAttention}(\bm{Q} \bm{W}_i^{Q}, \bm{K} \bm{W}_i^{K}, \bm{V} \bm{W}_i^{V}). $$

- **Position-wise Feed-Forward Network**: A function

    $$ \text{FFN} : \mathbb{R}^{n \times d_\text{model}} \rightarrow \mathbb{R}^{n \times d_\text{model}} $$

    defined by

    $$ \text{FFN}(\bm{X}) = \text{max}(0, \bm{X} \bm{W}_1 + \bm{b}_1) \bm{W}_2 + \bm{b}_2, \quad \text{where} $$

    $$ \bm{W}_1 \in \mathbb{R}^{d_\text{model} \times d_\text{ff}}, \quad \bm{b}_1 \in \mathbb{R}^{d_\text{ff}}, \quad \bm{W}_2 \in \mathbb{R}^{d_\text{ff} \times d_\text{model}}, \quad \bm{b}_2 \in \mathbb{R}^{d_\text{model}}. $$

- **Layer normalization**: A function

    $$ \text{LayerNorm} : \mathbb{R}^{n \times d_\text{model}} \rightarrow \mathbb{R}^{n \times d_\text{model}} $$

    defined by

    $$ \text{LayerNorm}(\bm{X})_{i,j} = \gamma_j \left( \frac{X_{i,j} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} \right) + \beta_j, \quad \text{where} $$

    $$ \mu_i = \frac{1}{d_\text{model}} \sum_{j=1}^{d_\text{model}} X_{i,j}, \quad \sigma_i^2 = \frac{1}{d_\text{model}} \sum_{j=1}^{d_\text{model}} (X_{i,j} - \mu_i)^2, \quad \bm{\gamma} \in \mathbb{R}^{d_\text{model}}, \quad \bm{\beta} \in \mathbb{R}^{d_\text{model}}. $$

- **Cross Entropy**: A function

    $$ \text{CrossEntropy} : [0, 1]^s \times [0, 1]^s \rightarrow \mathbb{R} $$

    defined by

    $$ \text{CrossEntropy}(\bm{y}, \hat{\bm{y}}) = - \sum_{j=1}^s y_j \log (\hat{y}_j). $$


&nbsp;
### **Forward pass**

$$
\begin{array}{@{}lll}
     \text{Input:} & \bm{x} \in \{1, 2, \ldots, s\}^{n}. &\hspace{10pt} (n) \\
     \text{Embedding:} & \bm{X}_0 = \text{OneHot}(\bm{x}) \bm{W}_e + \bm{W}_p. &\hspace{10pt} (n \times d_\text{model}) \\
     \text{Encoder:} & \bm{X}_l' = \text{LayerNorm}(\bm{X}_{l-1} + \text{MultiHead}(\bm{X}_{l-1}, \bm{X}_{l-1}, \bm{X}_{l-1})), &\hspace{10pt} (n \times d_\text{model}) \\
     & \bm{X}_l = \text{LayerNorm}(\bm{X}_l' + \text{FFNet}(\bm{X}_l')) \quad \text{for } l = 1, \ldots, N. &\hspace{10pt} (n \times d_\text{model}) \\
     \text{Target:} & \bm{y} \in \{1, 2, \ldots, s\}^{m}. &\hspace{10pt} (m) \\
     \text{Embedding:} & \bm{Y}_0 = \text{OneHot}(\bm{y}_{-m}) \bm{W}_e + \bm{W}_p. &\hspace{10pt} (m \times d_\text{model}) \\
     \text{Decoder:} & \bm{Y}_l' = \text{LayerNorm}(\bm{Y}_{l-1} + \text{MaskedMultiHead}(\bm{Y}_{l-1}, \bm{Y}_{l-1}, \bm{Y}_{l-1})), &\hspace{10pt} (m \times d_\text{model}) \\
     & \bm{Y}_l'' = \text{LayerNorm}(\bm{Y}_{l-1} + \text{MultiHead}(\bm{Y}_l', \bm{X}_N, \bm{X}_N)), &\hspace{10pt} (m \times d_\text{model}) \\
     & \bm{Y}_l = \text{LayerNorm}(\bm{Y}_l'' + \text{FFNet}(\bm{Y}_l'')) \quad \text{for } l = 1, \ldots, N. &\hspace{10pt} (m \times d_\text{model}) \\
     \text{Output:} & \bm{Y} = \text{Softmax}\left( \bm{Y}_N \bm{W}_e^\top \right) &\hspace{10pt} (m \times s) \\
     \text{Loss:} & \text{loss} = \sum_{j=2}^{m} \text{CrossEntropy}(\text{OneHot}(y_j), \bm{Y}_{j-1,:}). \\
     \text{Prediction:} & y_{m+1} \sim \bm{Y}_{m,:}.
\end{array}
$$


&nbsp;
### **Parameters**


The set of all parameters in the Transformer is given by:

$$
\begin{align*}
    \bm{\theta} = \; &\{ \bm{W}_e \} & \text{embeddings} \\
    &\cup \bigcup_{l=1}^N \left\{ \bm{W}_{i,l}^Q, \bm{W}_{i,l}^K, \bm{W}_{i,l}^V, \bm{W}_l^O \;\middle|\; i \in \{1, 2, \ldots, h \} \right\} & \text{encoder MultiHeads} \\
    &\cup \bigcup_{l=1}^N \{ \bm{W}_{1,l}, \bm{b}_{1,l}, \bm{W}_{2,l}, \bm{b}_{2,l} \} & \text{encoder FFNets} \\
    &\cup \bigcup_{l=1}^N \{ \bm{\gamma}_l, \bm{\beta}_l, \bm{\gamma}_l', \bm{\beta}_l' \} & \text{encoder LayerNorms} \\
    &\cup \bigcup_{l=1}^N \left\{ \bm{W}_{i,l}^Q, \bm{W}_{i,l}^K, \bm{W}_{i,l}^V, \bm{W}_l^O \;\middle|\; i \in \{1, 2, \ldots, h \} \right\} & \text{decoder MaskedMultiHeads} \\
    &\cup \bigcup_{l=1}^N \left\{ \bm{W}_{i,l}^Q, \bm{W}_{i,l}^K, \bm{W}_{i,l}^V, \bm{W}_l^O \;\middle|\; i \in \{1, 2, \ldots, h \} \right\} & \text{decoder MultiHeads} \\
    &\cup \bigcup_{l=1}^N \{ \bm{W}_{1,l}, \bm{b}_{1,l}, \bm{W}_{2,l}, \bm{b}_{2,l} \} & \text{decoder FFNets} \\
    &\cup \bigcup_{l=1}^N \{ \bm{\gamma}_l, \bm{\beta}_l, \bm{\gamma}_l', \bm{\beta}_l', \bm{\gamma}_l'', \bm{\beta}_l'' \}. & \text{decoder LayerNorms}
\end{align*}
$$

Correspongingly, the total number of parameters in the Transformer is

$$
\begin{align*}
    \lambda = &\; s \cdot d_\text{model} & \text{embeddings} \\
    &+ N \cdot ( h \cdot (d_\text{model} \cdot d_k + d_\text{model} \cdot d_k + d_\text{model} \cdot d_v) + h d_v \cdot d_\text{model} ) & \text{encoder MultiHeads} \\
    &+ N \cdot ( d_\text{model} \cdot d_\text{ff} + d_\text{ff} + d_\text{ff} \cdot d_\text{model} + d_\text{model} ) & \text{encoder FFNets} \\
    &+ N \cdot ( d_\text{model} + d_\text{model} + d_\text{model} + d_\text{model} ) & \text{encoder LayerNorms} \\
    &+ N \cdot ( h \cdot (d_\text{model} \cdot d_k + d_\text{model} \cdot d_k + d_\text{model} \cdot d_v) + h d_v \cdot d_\text{model} ) & \text{decoder MaskedMultiHeads} \\
    &+ N \cdot ( h \cdot (d_\text{model} \cdot d_k + d_\text{model} \cdot d_k + d_\text{model} \cdot d_v) + h d_v \cdot d_\text{model} ) & \text{decoder MultiHeads} \\
    &+ N \cdot ( d_\text{model} \cdot d_\text{ff} + d_\text{ff} + d_\text{ff} \cdot d_\text{model} + d_\text{model} ) & \text{decoder FFNets} \\
    &+ N \cdot ( d_\text{model} + d_\text{model} + d_\text{model} + d_\text{model} + d_\text{model} + d_\text{model} ). & \text{decoder LayerNorms}
\end{align*}
$$


&nbsp;
### **Hyperparameters**


Positional encoding:

$$
\begin{align*}
    (\bm{W}_p)_{pos,2i} &= \sin(pos/10000^{2i/d_\text{model}}), \\
    (\bm{W}_p)_{pos,2i+1} &= \cos(pos/10000^{2i/d_\text{model}}) \quad \text{where}
\end{align*}
$$

$$ pos \in \{1, 2, \ldots, n\}, i \in \{1, 2, \ldots, d_\text{model}/2\}. $$


Vocabulary:

$$
\begin{align*}
    s &\approx 37\,000 & \text{byte-pair encoding.}
\end{align*}
$$

Epsilon in Layer Normalization:

$$
\begin{align*}
    \epsilon
\end{align*}
$$


Hyperparameter values (used in the paper):

$$
\begin{align*}
    & d_\text{model} = 512 & \text{dimension of vector representations inside the model} \\
    & d_\text{ff} = 2048 & \text{dimension of the hidden layer in feed-forward nets} \\
    & d_k = 64 & \text{dimension of the vector space for queries and keys} \\
    & d_v = 64 & \text{dimension of the vector space for values} \\
    & h = 8 & \text{number of heads in multihead attention} \\
    & N = 6 & \text{number of encoder layers, number of decoder layers} \\
\end{align*}
$$

With these values (and $$ s = 37\,000 $$), we get $$ \lambda = 63\,045\,632 $$.
