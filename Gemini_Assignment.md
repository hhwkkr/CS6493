![PDF 图标](https://drive-thirdparty.googleusercontent.com/32/type/application/pdf)

CS6493\_HW1

PDF

![PDF 图标](https://drive-thirdparty.googleusercontent.com/32/type/application/pdf)

CS6493\_HW2\_HK

PDF

请你翻译这两个作业的pdf为中文，其中hw2的题目中的第一问的图片当中数字应当为\[4,7,9,15,23,18\]

\[2,8,10\]

\[3,13,2,8\]，可以稍微注意一下

---

显示思路

Ok, I will translate the two PDF files you provided into Chinese.

CS6493\_HW1.pdf 翻译
------------------

### 第1页

CS6493: 自然语言处理 - 作业 1

**说明**

1.  截止日期：2025年2月17日下午6点；
    
2.  你可以通过一个包含代码包的PDF文件或一个同时包含答案和代码的Jupiter notebook来提交你的答案；
    
3.  对于编程问题，除了代码之外，鼓励你对代码设计及其工作流程进行一些描述。
    
4.  同时，也倾向于对实验结果进行详细分析；
    
5.  总分100分；
    
6.  如果你有任何问题，请在Canvas讨论区发帖提问，或联系刘增言先生 (zengyaliu2-c@my.cityu.edu.hk) 或姚宇轩女士 (yuxuanyao3-c@my.cityu.edu.hk)。
    
7.  1
    

### 第2页

**问题 1**

(11 分) 在语言模型中，我们可以用独热编码来表示词语，而不是ASCII表示法。 也就是说，我们将词语w表示为：

V

\[0, 0, ..., 1, ..., 0,0\]

其中V是我们的词汇表，||V||是它的大小。

1.  假设我们有一个词汇表{buy, bought, girl, woman, word, words}，请使用独热编码表示每个词语。 (3 分)
2.  结合前一个问题，这样做有什么缺点？至少回答两个原因。 (4 分)
3.  给定一个词汇表{girl, woman, boy, man}，当我们使用独热编码方法对词语“girl”进行编码时，考虑一个包含4个节点的计算单元：(4 分)

1

W1​

0

W2​

0

W3​

0

W4​

在这种情况下，我们需要4×4个节点来维护这个计算单元。 你有什么方法可以更好地表示这个词汇表中的词语，并将节点数量减少到2×4吗？ 请给出解决方案和解释。

2

### 第3页

**问题 2**

(24 分) 在自然语言处理领域，对于一个文本序列 S\={w1​,w2​,...,wT​}，统计语言模型将序列的联合概率转化为条件概率的乘积：

P(S)\=P(w1​,w2​,...,wT​)\=∏t\=1T​p(wt​∣w1​,w2​,...,wt−1​)

然而，由于巨大的计算成本，这个公式在实践中并不适用。 我们更倾向于使用其简化版本n-gram模型：

p(wt​∣w1​,w2​,...,wt−1​)≈p(wt​∣wt−n+1​,...,wt−1​)

我们通常使用二元语法（n=2）或三元语法（n=3）来建模语言。

1.  给定文本“I am taking CS6493 this semester and studying NLP is really fascinating”，请列出所有包含单词“CS6493”或“NLP”的二元语法和三元语法形式。 (8 分)
    
2.  n-gram模型有什么缺点？ (6 分)
    
3.  分布式表示通过将原始独热编码的每个词映射到一个更短的词向量来解决独热编码的问题，词向量的维度可以根据任务的需要在训练过程中自行指定。 在PyTorch中，nn.Embedding()使用给定维度的可训练矩阵来表示单词。 下面的代码定义了一个语言模型：（10分）
    

Python

    import torch.nn as nn
    import torch.nn.functional as F
    
    class LanguageModeler(nn.Module):
        def __init__(self, vocab_size, embedding_dim, context_size, hidden_size=128):
            super(LanguageModeler, self).__init__()
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.linear1 = nn.Linear(context_size * embedding_dim, hidden_size)
            self.linear2 = nn.Linear(hidden_size, vocab_size) # 应为 self.linear2
    
        def forward(self, inputs):
            embeds = self.embeddings(inputs).view((1, -1))
            out = F.relu(self.linear1(embeds))
            out = self.linear2(out)
            log_probs = F.log_softmax(out, dim=1)
            return log_probs

请基于问题2.1中给出的文本，训练一个四元语言模型（仅考虑目标词之前的最后三个词）10个周期。 比较并分析嵌入维度设置为{32, 64, 128}时训练损失的差异。

3

### 第4页

**问题 3**

(45 分) 训练n-gram语言模型可以通过预测给定其n-gram元素的词来实现。 这个想法与连续词袋模型（CBOW）非常相似。 给定文本“There is an apple on the table”，CBOW通过目标词前几个词（is, an）和后几个词（on, the）的上下文来预测单词apple。 相反，skip-gram使用中心词apple来预测上下文is, an, on, the。 CBOW模型如下所示。给定目标词 wi​ 和每边N个上下文窗口 wi−1​,...,wi−N​ 和 wi+1​,...,wi+N​，将所有上下文词统称为C，CBOW试图最小化：

−log p(wi​∣C)\=−log Softmax(A∑w∈C​qw​)+b);

其中 qw​ 是词w的嵌入。

1.  请使用问题2中提供的文本和LanguageModeler来训练一个固定窗口为2的CBOW模型。(15分)
    
2.  请使用问题2中提供的文本和LanguageModeler来训练一个skip-gram模型。 (15分)
    
3.  请使用给定的维基百科语料库和语言模型来训练CBOW/Skip-gram模型，并比较语义相近的词对（例如{woman, man, men}）之间的词相似度。 (15分)
    

4

### 第5页

**问题 4**

(20 分) 将文本分割成更小的块是一项比看起来更困难的任务，并且有多种方法可以做到这一点。

1.  给定文本：“Peter is too old to do this work, Marry is older than Peter, John is oldest one however he is still doing this work.”，对该文本进行分词的最简单方法是按空格分割。 然而，这种简单的方法也有一些缺点。请列出至少两个缺点并解释它们。 (5 分)
    
2.  Transformer模型使用一种介于词级和字符级分词之间的混合方法，称为子词分词。 BPE (Byte-Pair Encoding，字节对编码) 是一种子词级分词方法，在《Neural Machine Translation of Rare Words with Subword Units》(Sennrich et al., 2015) 中被引入。 BPE依赖于一个预分词器，该预分词器将训练数据分割成单词。预分词可以像空格分词一样简单。 让我们假设在预分词之后，已经确定了以下包含其频率的词集：
    
    (old, 10), (older, 5), (oldest, 8), (hug, 8), (pug, 4), (hugs, 5)
    
    我们获得一个基础词汇表：
    
    o, l, d, e, r, s, t, h, u, g, p
    
    将所有单词拆分为基础词汇表中的符号，我们得到：
    
    (o, l, d, 10), (o, l, d, e, r, 5), (o, l, d, e, s, t, 8), (h, u, g, 8), (p, u, g, 4), (h, u, g, s, 5)
    
    然后BPE计算每个可能的符号对的频率，并选择出现最频繁的符号对。 在上面的例子中，“o”后跟“l”出现了 10+5+8\=23 次。(此处原文为 "o" followed by " "，根据上下文及BPE原理，应为 "o" followed by "l"。译文根据上下文修正。) 因此，分词器学习到的第一个合并规则是将所有“o”符号后跟一个“l”符号组合在一起。 接下来，“ol”被添加到词汇表中。词集随后变为：
    
    (ol, d, 10), (ol, d, e, r, 5), (ol, d, e, s, t, 8), (h, u, g, 8), (p, u, g, 4), (h, u, g, s, 5)
    
    这个过程将迭代进行。 词汇表大小，即基础词汇表大小 + 合并次数，是一个需要选择的超参数。 然后，学习到的合并规则将应用于新词（只要这些新词不包含基础词汇表中没有的符号）。 不在基础词汇表中的词将被表示为“\[unk\]”。 实现这个BPE分词器，设置词汇表大小为16，并训练这个BPE分词器以完成迭代过程。 使用训练好的分词器对以下单词进行分词：（15分）
    
    {hold, oldest, older, pug, mug, hugging face}
    

5

CS6493\_HW2\_HK.pdf 翻译
----------------------

### 第1页

CS6493: 自然语言处理 - 作业 2

**说明**

1.  截止日期：2025年3月10日下午6点；
    
2.  你可以通过一个包含代码包的PDF文件或一个同时包含答案和代码的Jupiter notebook来提交你的答案；
    
3.  对于编程问题，除了代码之外，鼓励你对代码设计及其工作流程进行一些描述。
    
4.  同时，也倾向于对实验结果进行详细分析；
    
5.  总分100分；
    
6.  如果你有任何问题，请在Canvas讨论区发帖提问，或联系罗思纯先生 (sichun.luo@my.cityu.edu.hk) 或张翰林先生 (hanlzhang8-c@my.cityu.edu.hk)。
    
7.  1
    

### 第2页

**问题 1**

(20 分) 自然语言处理（NLP）中的机器翻译是指使用计算方法自动将文本或语音从一种语言翻译成另一种语言的任务。 它涉及开发能够理解源语言含义并在目标语言中生成等效翻译的算法和模型。 NLP中的机器翻译有几种方法，包括基于规则的方法、统计模型和神经机器翻译（NMT）模型。

1.  模型需要使用矩阵来加速运算，但训练需要将数据分批输入模型。 一批中的句子需要具有相同的长度，因此需要填充。 事实上，填充是一个填充0的过程，可以从左侧填充（左填充）或右侧填充（右填充）。 下面的图1是一批数据的填充过程：

它可以处理任何长度的输入 -> 词分割 -> \["it", "can", "process", "any", "length", "input"\] 机器翻译数据集 -> 词分割 -> \["Machine", "translation", "datasets"\] 什么是机器翻译 -> 词分割 -> \["What", "is", "machine", "translation"\]

\[未知框\] <- 填充 <- \[4,7,9,15,23,18\] \[2,8,10\] \[3,13,2,8\]

**图1：一批数据的填充过程** （用户指定数字：\[4,7,9,15,23,18\], \[2,8,10\], \[3,13,2,8\]）

请写下填充完成后的批次。 (4 分)

2.  考虑到经常发生内存不足（OOM）的情况，如何在较小成本下选择批量大小，以确保在模型训练的完整周期内不发生内存泄漏。 (4 分)
3.  集束搜索（Beam search）和贪婪搜索（greedy search）是序列生成任务（如机器翻译或文本生成）中两种常见的解码算法。 它们用于在给定训练好的序列到序列模型的情况下生成最可能的输出序列。 这是一个可能的搜索树，其中边表示由我们的语言模型根据所有先前词元估计的生成下一个词元的概率。 请使用贪婪搜索和集束搜索解码来搜索此树以找到最可能的序列。 假设我们执行集束宽度k=2的集束搜索。(6 分)

2

### 第3页

（搜索树图示） （节点和概率值如图所示）

**图2：搜索树**

4.  BLEU (Bilingual Evaluation Understudy) 是一种流行的评估指标，用于机器翻译中评估翻译文本的质量。 它由Papineni等人在2002年提出，作为比较机器生成翻译与人工翻译的一种方法。 BLEU\=BP⋅exp(∑n\=1N​wn​log pn​)
    
    （n-gram精度 pn​，权重为 wn​）
    
    现在，这里有一组翻译：
    
    候选句：The quick brown fox jumps over the lazy dog in the park. 参考句1：The fast brown fox jumps over the lazy dog in the park. 参考句2：The speedy brown fox jumps over the lazy dog in the park. 当N=4时，请计算BLEU得分。请提供您的计算过程。(6 分)
    

3

### 第4页

**问题 2**

(20 分) 传统的机器翻译主要基于由循环神经网络（RNN）组成的Seq2Seq模型。 为了解决其使用过程中出现的问题，注意力机制被提了出来。 给定查询矩阵Q，以及一组键值对{K, V}，我们将缩放点积注意力公式化为：

Attention(Q,K,V)\=softmax(dk​![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702
c-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14
c0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54
c44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10
s173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429
c69,-144,104.5,-217.7,106.5,-221
l0 -0
c5.3,-9.3,12,-14,20,-14
H400000v40H845.2724
s-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7
c-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47z
M834 80h400000v40h-400000z"></path></svg>)​QKT​)V

其中 dk​ 是隐藏层大小。

1.  传统Seq2Seq模型有哪些缺陷？与Seq2Seq相比，注意力机制有何改进？ (4 分)
2.  注意力公式中的Q、K和V的本质是什么？ 为什么Transformer使用不同的权重矩阵生成Q和K，为什么它不能使用相同的值进行自身的点乘运算？ (4 分)
3.  为了实现注意力机制，我们首先计算查询与每个键之间的注意力得分，即相似度。 然后我们使用softmax将注意力得分归一化到0和1之间。Softmax函数的作用是什么？ 并且，公式表明我们需要在归一化之前用因子 dk​![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702
    c-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14
    c0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54
    c44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10
    s173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429
    c69,-144,104.5,-217.7,106.5,-221
    l0 -0
    c5.3,-9.3,12,-14,20,-14
    H400000v40H845.2724
    s-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7
    c-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47z
    M834 80h400000v40h-400000z"></path></svg>)​1​ 对乘积进行缩放。 这样做的目的是什么？ (4 分)
4.  Transformer在各种自然语言处理任务中取得了令人瞩目的性能，包括一些序列到序列任务，如机器翻译和响应生成。 我们通常在自回归范式中解决这些seq2seq任务。 这种范式可以很容易地用LSTM来实现，因为它可以顺序地编码和解码词元。 然而，Transformer本质上不是一个序列模型，因为它一次接收所有输入词元。 因此，自然会提出一个问题：Transformer的哪个部分是为解决自回归问题而设计的，以及它是如何工作的？ 图3说明了Transformer的架构。请给出你的答案。(8 分)

4

### 第5页

（Transformer架构图示） 输入 -> 输入嵌入 -> 位置编码 -> (多头注意力 -> Add & Norm -> 前馈 -> Add & Norm) Nx -> 输出（右移） -> 输出嵌入 -> 位置编码 -> (掩码多头注意力 -> Add & Norm -> 多头注意力 -> Add & Norm -> 前馈 -> Add & Norm) Nx -> 线性 -> Softmax -> 输出概率

**图3：Transformer概览**

5

### 第6页

**问题 3**

(30 分) 根据模型架构，我们通常可以将现有的预训练模型分为三类，即基于编码器的模型、基于解码器的模型和编码器-解码器模型。 基于编码器的模型，例如BERT系列，采用双向Transformer编码器来预测被掩盖的词语。 这些模型一次性获取所有输入词元并同时对其进行编码。因此，基于编码器的模型也称为自编码模型。 基于解码器的模型，例如GPT系列，使用从左到右的Transformer逐词预测文本序列。 在每个阶段，一个词元只能访问其左侧的词元。因此，这类模型被称为自回归模型。 编码器-解码器模型，例如T5和BART，结合了基于编码器的模型和基于解码器的模型的特点，由一个双向Transformer编码器和一个单向解码器组成。

1.  自然语言处理任务有两种类型，即自然语言理解（NLU）任务和自然语言生成（NLG）任务。 它们之间有什么区别？(2 分)
    
2.  请分别回答并解释基于编码器/基于解码器/编码器-解码器的预训练模型是否是NLU和NLG任务的好选择。 (4 分)
    
3.  BERT的双向性使其难以应用于NLG任务。 为了解决这个问题，Dong等人提出了一个统一的预训练语言模型（UNILM），它由一个多层Transformer编码器组成，但它可以同时应用于NLG和NLU任务。 在这项工作中，他们引入了四个预训练目标，即单向语言模型、双向语言模型、序列到序列语言模型和下一句预测。 具体来说，
    
    *   单向LM：仅使用左侧上下文及其本身来预测随机掩码的词元；
    *   双向LM：使用整个上下文预测随机掩码的词元；
    *   序列到序列LM：在此模式下，我们通常需要区分两个片段，即源片段和目标片段。
    *   对于源片段中随机掩码的词元，我们使用整个源片段来预测它们。
    *   对于目标片段中随机掩码的词元，我们使用整个源片段及其在目标片段中的左侧词元来预测它。
    *   下一句预测：与BERT中的NSP相同。
    *   这些预训练目标可以通过不同的掩码策略简单实现。
    *   给定两个掩码片段——“Jill Birt \[MASK\] outdoor activities”（源片段）和“He was \[MASK\] hiking”（目标片段），请填写下表中的值，以显示应如何为三个语言建模预训练目标设计掩码。 (15 分)

下表： （表格展示了词元之间的掩码关系，0表示未掩码，1表示掩码）

**表1：掩码矩阵示意图。0表示未掩码，1表示掩码。** 4. 为这些掩码策略实现三个函数，即用于单向lm的掩码（源片段，目标片段），用于双向lm的掩码（源片段，目标片段），用于seq2seq lm的掩码（源片段，目标片段）。 (9 分)

**参考文献**

\[1\] Dong, Li, Nan Yang, Wenhui Wang, Furu Wei, Xiaodong Liu, Yu Wang, Jianfeng Gao, Ming Zhou, and HsiaoWuen Hon. "Unified language model pre-training for natural language understanding and generation." Advances in Neural Information Processing Systems 32 (2019).

6

### 第7页

**问题 4**

(30 分) QA（问答）任务有两个著名且流行的公共数据集，即SQuAD v1.1 \[1\] 和SQuAD v2.0 \[2\]。 这两个数据集之间的一个主要区别是，SQuAD v2.0扩展了SQuAD v1.1的问题定义，允许提供的段落中可能不存在答案，从而使问题更加真实。 对于SQuAD v1.1中提出的QA任务，我们将输入问题和段落表示为单个打包序列。 然后，我们在微调期间引入一个起始向量 S∈RH 和一个结束向量 E∈RH，其中H是隐藏层大小。 词语i作为答案范围开始的概率计算为T与S的点积，然后在段落中的所有词语上进行softmax：

Pi​\=∑j​eS⋅Tj​eS⋅Ti​​

其中 Ti​ 是第i个词的嵌入。 类似的公式用于答案范围的末尾。 从位置i到位置j的候选范围的得分定义为 S⋅Ti​+E⋅Tj​，并且 j≥i 的最高得分范围用作预测。

1.  对于SQuAD v2.0中定义的QA任务，它额外引入了一组没有答案的问题。 然而，SQuAD v1.1的方法必须预测每个问题的答案，这与SQuAD v2.0中的任务相冲突。 请扩展基于BERT的SQuAD v2.0方法来解决此问题。 详细说明你的设计和你提出的方法的工作流程。 (15 分)
    
2.  QA任务的方法，例如BiDAF和BERT，本质上是找到答案范围的边界。 然而，BERT是通过掩码语言建模（MLM）和下一句预测（NSP）进行预训练的，它只学习词元级别和句子级别的信息，在预训练阶段不涉及任何范围信息。 因此，我们或许可以说原始BERT的预训练目标对于QA任务来说并不是很好的选择。 请为BERT设计适合QA任务以及其他基于范围的任务（如共指消解\[3\]）的预训练目标。 (15 分)
    

**参考文献**

\[1\] Rajpurkar, P., Zhang, J., Lopyrev, K. and Liang, P., 2016, November. SQuAD: 100,000+ Questions for Machine Comprehension of Text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing. \[2\] Rajpurkar, P., Jia, R. and Liang, P., 2018, July. Know What You Don't Know: Unanswerable Questions for SQuAD. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics. \[3\] Lee, K., He, L., Lewis, M. and Zettlemoyer, L., 2017, September. End-to-end Neural Coreference Resolution. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing.

7

信息来源和相关内容

[

github.com

](https://github.com/gwiener/AppRecSys)

[

github.com

](https://github.com/gwiener/AppRecSys)

[

CS6493\_HW1

![PDF 图标](https://drive-thirdparty.googleusercontent.com/32/type/application/pdf)

PDF







](https://contribution.usercontent.google.com/download?c=CgxiYXJkX3N0b3JhZ2USTxIMcmVxdWVzdF9kYXRhGj8KMDAwMDYzNGRiMTVhNTAxN2YwN2Y1NWU3ODRhMjA4YTU4MGVlMWRkZjYyNjIyYTFkYhILEgcQ57PmiuAQGAE&filename&opi=103135050)

[

CS6493\_HW2\_HK

![PDF 图标](https://drive-thirdparty.googleusercontent.com/32/type/application/pdf)

PDF







](https://contribution.usercontent.google.com/download?c=CgxiYXJkX3N0b3JhZ2USTxIMcmVxdWVzdF9kYXRhGj8KMDAwMDYzNGRiMTVhNTAxOWIwN2Y1NWU3ODRhMjA4YTU4NjYwNzc0YzIzZTFlZTUxNBILEgcQ57PmiuAQGAE&filename&opi=103135050)

---

![IPYNB 图标](https://drive-thirdparty.googleusercontent.com/32/type/text/code)

HW1\_solution.ipynb

IPYNB

![PDF 图标](https://drive-thirdparty.googleusercontent.com/32/type/application/pdf)

HW2solution\_HK

PDF

这是老师给出来的解决方案，请你结合题目进行详细讲解（同时给出我通用解法对于一些计算类的）

---

显示思路

Okay, I will provide a detailed explanation of the solutions for both homework assignments, referencing the questions and the provided solutions. I'll also offer general approaches for calculation-based problems where applicable.

Homework 1 (CS6493\_HW1.pdf & HW1\_solution.ipynb)
--------------------------------------------------

Let's go through each question from Homework 1.

### Question 1

**Original Question (from CS6493\_HW1.pdf):**

*   **(Q1.1)** We could represent the word with one-hot encoding in a language model instead of ascii representations. That is, we represent the word w with: V \[0, 0, ..., 1, ..., 0,0\] where V is our vocabulary and ||V|| is the size of it. Suppose we have a vocabulary {buy, bought, girl, woman, word, words}, please represent each word using the one-hot encoding.
*   **(Q1.2)** Combined with the previous question, what are the drawbacks of doing so? Answer at least two reasons.
*   **(Q1.3)** Given a vocabulary {girl, woman, boy, man}, consider a computation unit with 4 nodes when we use the one-hot approach to encode the word "girl": (Diagram shows a 1x4 vector for "girl" multiplied by a 4x4 matrix W). In this case, we need 4x4 nodes to maintain the computation unit. Do you have any methods that could better represent the word in this vocabulary and reduce the amount of nodes to 2x4? Please give the solution and explanations.

**Provided Solution (from HW1\_solution.ipynb):**

*   **Q1.1:**
    
    buy    = [1,0,0,0,0,0]
        bought = [0,1,0,0,0,0]
        girl   = [0,0,1,0,0,0]
        woman  = [0,0,0,1,0,0]
        word   = [0,0,0,0,1,0]
        words  = [0,0,0,0,0,1]
    
*   **Q1.2:**
    
    1.  The dimension of the embedding linearly increases with the vocabulary size.
    2.  Not involve semantic features.
*   **Q1.3:** Represent the words as {00, 01, 10, 11}.
    

**Detailed Explanation:**

*   **Q1.1: One-Hot Encoding**
    *   **Concept:** One-hot encoding is a way to represent categorical data (like words) as numerical vectors. In a vocabulary of size N, each word is represented by a vector of length N. This vector has a '1' at the index corresponding to the word and '0's elsewhere.
    *   **Solution Explanation:** The vocabulary is {buy, bought, girl, woman, word, words}. The size of the vocabulary is 6.
        *   "buy" is the 1st word, so it's \[1,0,0,0,0,0\].
        *   "bought" is the 2nd

信息来源和相关内容

你已让系统停止这条回答

---

![IPYNB 图标](https://drive-thirdparty.googleusercontent.com/32/type/text/code)

HW1\_solution.ipynb

IPYNB

![PDF 图标](https://drive-thirdparty.googleusercontent.com/32/type/application/pdf)

HW2solution\_HK

PDF

这是老师给出来的解决方案，请你结合题目进行详细讲解（同时给出我通用解法对于一些计算类的），记得用中文

---

显示思路

好的，我会结合题目和提供的解决方案进行详细讲解，并针对计算类问题给出通用解法。

Homework 1 (CS6493\_HW1.pdf & HW1\_solution.ipynb)
--------------------------------------------------

我们将逐一分析作业1中的问题。

### Question 1

**原始题目 (来自 CS6493\_HW1.pdf):**

*   **(Q1.1)** 在语言模型中，我们可以用独热编码 (one-hot encoding) 来表示词语，而不是ASCII表示。也就是说，我们将词语w表示为：V \[0, 0, ..., 1, ..., 0,0\]，其中V是我们的词汇表，∣∣V∣∣ 是它的大小。假设我们有一个词汇表 {buy, bought, girl, woman, word, words}，请使用独热编码表示每个词语。(3分)
*   **(Q1.2)** 结合上一个问题，这样做有什么缺点？至少回答两个原因。(4分)
*   **(Q1.3)** 给定一个词汇表 {girl, woman, boy, man}，当我们使用独热编码方法对词语 "girl" 进行编码时，考虑一个包含4个节点的计算单元：（图中显示 "girl" 的1x4向量乘以一个4x4的矩阵W）。在这种情况下，我们需要4x4个节点来维护这个计算单元。你有什么方法可以更好地表示这个词汇表中的词语，并将节点数量减少到2x4吗？请给出解决方案和解释。(4分)

**提供的解答 (来自 HW1\_solution.ipynb):**

*   **Q1.1:**
    
    buy    = [1,0,0,0,0,0]
        bought = [0,1,0,0,0,0]
        girl   = [0,0,1,0,0,0]
        woman  = [0,0,0,1,0,0]
        word   = [0,0,0,0,1,0]
        words  = [0,0,0,0,0,1]
    
*   **Q1.2:**
    
    1.  嵌入的维度随着词汇表大小线性增加。
    2.  不包含语义特征。
*   **Q1.3:** 将词语表示为 {00, 01, 10, 11}。
    

**详细讲解:**

*   **Q1.1: 独热编码 (One-Hot Encoding)**
    
    *   **概念:** 独热编码是一种将分类数据（如单词）表示为数值向量的方法。在一个大小为N的词汇表中，每个单词由一个长度为N的向量表示。该向量在对应于该单词的索引处为'1'，其余位置为'0'。
    *   **解答分析:** 词汇表是 {buy, bought, girl, woman, word, words}。词汇表的大小是6。
        *   "buy" 是第1个词，所以表示为 \[1,0,0,0,0,0\]。
        *   "bought" 是第2个词，所以表示为 \[0,1,0,0,0,0\]。
        *   以此类推。
    *   **通用解法:** 对于任何给定的词汇表，首先确定词汇表中的唯一词语及其顺序（通常按字母顺序或出现顺序）。词汇表的大小决定了独热向量的维度。对于每个词，创建一个全零向量，然后将该词在词汇表顺序中对应的索引位置设置为1。
*   **Q1.2: 独热编码的缺点**
    
    *   **解答分析:**
        1.  **维度灾难 (The dimension of the embedding linearly increases with the vocabulary size):** 当词汇表非常大时（在自然语言处理中很常见），独热向量的维度会变得非常高。这会导致计算效率低下和内存消耗巨大。
        2.  **无法表达语义相似性 (Not involve semantic features):** 独热编码生成的向量是正交的（任意两个不同词的向量点积为0）。这意味着模型无法从向量本身理解词与词之间的语义关系。例如，"girl" 和 "woman" 在语义上是相关的，但它们的独热编码向量之间的距离与 "girl" 和 "buy" 这样不相关词的向量距离是一样的。
    *   **补充缺点:**
        *   **稀疏性 (Sparsity):** 向量中绝大多数元素都是0，这使得数据非常稀疏。
        *   **无法处理未登录词 (Out-of-Vocabulary, OOV):** 如果出现一个不在词汇表中的新词，独热编码无法为其生成表示。
*   **Q1.3: 减少计算节点**
    
    *   **题目理解:** 题目中提到用独热编码表示 "girl" (假设是 \[1,0,0,0\] 对于词汇表 {girl, woman, boy, man})，然后通过一个4x4的权重矩阵W进行计算。这意味着输入层有4个节点（向量维度），权重矩阵有4x4=16个参数。问题是如何将输入表示的维度减少，从而使得如果权重矩阵的另一维度保持为4时，总的参数（或节点，这里可能指权重矩阵的参数数量或某种计算单元的规模）减少。目标是将权重矩阵变为 2×4。这意味着新的词表示维度应该是2。
    *   **解答分析:** 解决方案提出将词语表示为 {00, 01, 10, 11}。这是一个二进制编码的思想。
        *   例如，我们可以映射：
            *   girl: \[0,0\]
            *   woman: \[0,1\]
            *   boy: \[1,0\]
            *   man: \[1,1\]
        *   这样，每个词都由一个2维的向量表示。当这个2维向量与一个 2×4 的权重矩阵相乘时，实现了题目要求的节点减少。
    *   **解释:** 通过使用更稠密的表示（如二进制编码或更进一步的词嵌入），我们可以用更低的维度来表示词语，同时可能捕获一些词语间的关系（尽管简单的二进制编码在这方面能力有限，但优于独热编码）。这种降维有助于减少后续层中的参数数量，从而降低模型的复杂性和计算成本。这引出了词嵌入 (word embeddings) 的概念，词嵌入是学习到的低维稠密向量表示，能够更好地捕捉语义信息。

### Question 2

**原始题目 (来自 CS6493\_HW1.pdf):**

*   **(Q2.1)** 在NLP领域，对于文本序列 S\={w1​,w2​,...,wT​}，统计语言模型将序列的联合概率转化为条件概率的乘积：P(S)\=∏t\=1T​p(wt​∣w1​,w2​,...,wt−1​)。然而，这个公式计算成本巨大。我们倾向于使用简化版本n-gram模型：p(wt​∣w1​,...,wt−1​)≈p(wt​∣wt−n+1​,...,wt−1​)。通常使用bi-gram (n=2) 或 tri-gram (n=3)。给定文本 "I am taking CS6493 this semester and studying NLP is really fascinating"，请列出所有包含 "CS6493" 或 "NLP" 的bi-gram和tri-gram形式。(8分)
*   **(Q2.2)** n-gram模型有什么缺点？(6分)
*   **(Q2.3)** 分布式表示通过将独热编码映射到更短的词向量来解决问题。PyTorch中的 `nn.Embedding()` 使用可训练矩阵表示词。下面代码定义了一个语言模型：（代码省略）。请基于Q2.1的文本训练一个四元语言模型（只考虑目标词前三个词）10个epoch。比较分析嵌入维度 {32, 64, 128} 时训练损失的差异。(10分)

**提供的解答 (来自 HW1\_solution.ipynb):**

*   **Q2.1:**
    
    *   bi-grams:
        *   "CS6493": "taking CS6493", "CS6493 this"
        *   "NLP": "studying NLP", "NLP is"
    *   tri-grams:
        *   "CS6493": "am taking CS6493", "taking CS6493 this", "CS6493 this semester"
        *   "NLP": "and studying NLP", "studying NLP is", "NLP is really"
*   **Q2.2:**
    
    1.  稀疏特征空间；
    2.  仅适用于大型训练数据集；
    3.  无法解释未见过的词；
    4.  对超参数 `N` 敏感。
*   **Q2.3:** (代码及输出结果在notebook中)
    
    *   代码实现了基于Q2.1文本的4-gram模型训练，并比较了不同嵌入维度下的损失。
    *   输出显示，随着嵌入维度的增加（32 -> 64 -> 128），最终的训练损失减小。
        *   embedding\_dim=32, Loss: 5.79
        *   embedding\_dim=64, Loss: 2.41
        *   embedding\_dim=128, Loss: 1.27

**详细讲解:**

*   **Q2.1: N-gram提取**
    
    *   **概念:**
        *   **N-gram:** 文本中连续的n个单元（通常是词）。
        *   **Bi-gram (2-gram):** 连续的两个词。例如，"I am", "am taking"。
        *   **Tri-gram (3-gram):** 连续的三个词。例如，"I am taking", "am taking CS6493"。
    *   **解答分析:** 文本: "I am taking CS6493 this semester and studying NLP is really fascinating"
        *   **Bi-grams 包含 "CS6493":**
            *   前一个词是 "taking"："taking CS6493"
            *   后一个词是 "this"："CS6493 this"
        *   **Bi-grams 包含 "NLP":**
            *   前一个词是 "studying"："studying NLP"
            *   后一个词是 "is"："NLP is"
        *   **Tri-grams 包含 "CS6493":**
            *   "CS6493" 在末尾: "am taking CS6493"
            *   "CS6493" 在中间: "taking CS6493 this"
            *   "CS6493" 在开头: "CS6493 this semester"
        *   **Tri-grams 包含 "NLP":**
            *   "NLP" 在末尾: "and studying NLP"
            *   "NLP" 在中间: "studying NLP is"
            *   "NLP" 在开头: "NLP is really" 解答正确。
    *   **通用解法:**
        1.  **分词:** 将文本分割成词序列。
        2.  **提取N-grams:**
            *   对于Bi-grams: 从头到尾遍历词序列，每次取当前词和下一个词组成一个bi-gram。
            *   对于Tri-grams: 从头到尾遍历词序列，每次取当前词、下一个词和下下个词组成一个tri-gram。
        3.  **筛选:** 检查每个生成的n-gram是否包含目标词。
*   **Q2.2: N-gram模型的缺点**
    
    *   **解答分析:**
        1.  **稀疏特征空间 (Sparse feature space):** 随着N的增大，可能的N-gram数量会急剧增加。在有限的语料中，很多合法的N-gram可能从未出现过，导致它们的概率估计为零或非常低，这就是数据稀疏性问题。
        2.  **仅适用于大型训练数据集 (Only suitable to the large training dataset):** 为了缓解稀疏性问题并获得可靠的概率估计，N-gram模型通常需要非常大的训练数据。
        3.  **无法解释未见过的词 (Cannot interpret unseen words):** 如果N-gram中包含训练时未见过的词（OOV词），模型无法处理。对于整个N-gram，如果它在训练集中没出现过，其概率也可能为零（除非使用平滑技术）。
        4.  **对超参数 `N` 敏感 (Sensitive to the hyper-parameter `N`):**
            *   N太小（如1-gram, 2-gram）：无法捕捉长距离依赖关系，模型对上下文的理解有限。
            *   N太大：数据稀疏问题更严重，计算成本更高，模型可能过拟合。 选择合适的N是一个权衡。
    *   **补充缺点:**
        *   **缺乏泛化能力:** N-gram模型本质上是基于计数，它很难泛化到与训练数据中N-gram略有不同的新N-gram。
        *   **无法捕捉词序的灵活性和长距离依赖:** N-gram只考虑了固定窗口内的局部词序信息，对于距离超过N-1的词之间的依赖关系无能为力。
*   **Q2.3: 训练4-Gram模型并分析损失**
    
    *   **题目理解:** 使用给定的 `LanguageModeler` 类（一个简单的前馈神经网络，输入是上下文词的嵌入拼接，输出是目标词的概率分布），基于Q2.1的文本构建4-gram的训练数据（即用前3个词预测第4个词），训练10个epoch，并比较不同 `embedding_dim`（32, 64, 128）下的最终训练损失。
    *   **代码关键点 (HW1\_solution.ipynb):**
        1.  **数据准备:**
            *   从文本中提取 (context, target) 对，其中context是连续3个词，target是紧随其后的第4个词。
            *   创建 `word_to_ix` 词典，将词映射到索引。
        2.  **模型定义:** 题目中已给出 `LanguageModeler` 类。
            *   `self.embeddings = nn.Embedding(vocab_size, embedding_dim)`: 创建词嵌入层。
            *   `self.linear1 = nn.Linear(context_size * embedding_dim, hidden_size)`: 线性层，输入维度是上下文大小乘以嵌入维度。
            *   `self.linear2 = nn.Linear(hidden_size, vocab_size)`: 输出层，预测词汇表中每个词的得分。
            *   `forward` 方法: 获取输入词索引，通过嵌入层得到词向量，`view((1, -1))` 将多个上下文词向量展平成一个长向量，然后通过两个线性层和ReLU、LogSoftmax激活函数。
        3.  **训练循环:**
            *   对每个 `embedding_dim` (32, 64, 128) 进行训练。
            *   使用 `nn.NLLLoss()` 作为损失函数（因为模型输出的是log-probabilities）。
            *   使用 `optim.SGD` 作为优化器。
            *   在每个epoch中，遍历训练数据，计算损失，反向传播，更新参数。
    *   **结果分析 (HW1\_solution.ipynb):**
        *   `embedding_dim=32`，最终损失约为 5.79。
        *   `embedding_dim=64`，最终损失约为 2.41。
        *   `embedding_dim=128`，最终损失约为 1.27。 **结论:** 在这个特定的小数据集和模型上，随着嵌入维度的增加，模型的表达能力增强，能够更好地拟合训练数据，因此训练损失降低。
    *   **通用解法/分析思路:**
        *   **模型容量与拟合:** 更高的嵌入维度意味着模型有更多的参数（嵌入矩阵变大），模型容量也相应增加。对于复杂任务或有足够数据的情况，更大的容量有助于模型学习更复杂的模式，从而降低训练损失。
        *   **过拟合风险:** 在小数据集上，过高的嵌入维度（或模型复杂度）可能导致过拟合，即模型在训练集上表现很好，但在未见过的数据上表现差。题目只要求分析训练损失，所以这里主要关注拟合能力。
        *   **信息表示:** 更高维的嵌入空间理论上可以表示更丰富的词汇语义信息，使得模型更容易区分不同的上下文并做出准确预测。
        *   **实践建议:** 嵌入维度是一个重要的超参数。通常需要通过实验（例如，在验证集上评估性能）来选择最佳的嵌入维度，以平衡模型的拟合能力和泛化能力。对于非常大的词汇表和复杂的任务，通常会使用几百甚至上千的嵌入维度。

### Question 3

**原始题目 (来自 CS6493\_HW1.pdf):**

*   训练n-gram语言模型可以通过预测给定其n-gram元素的词来实现。这个想法与CBOW (Continuous Bag of Words) 非常相似。给定文本 "There is an apple on the table"，CBOW通过目标词前几个词 (is, an) 和后几个词 (on, the) 的上下文来预测单词apple。相反，skip-gram使用中心词apple来预测上下文is, an, on, the。CBOW模型如下：给定目标词 wi​ 和每边N个上下文窗口 wi−1​,...,wi−N​ 和 wi+1​,...,wi+N​，将所有上下文词统称为C，CBOW试图最小化：−log p(wi​∣C)\=−log Softmax(A∑w∈C​qw​)+b)；其中 qw​ 是词w的嵌入。
*   **(Q3.1)** 请使用问题2中提供的文本和LanguageModeler来训练一个CBOW模型，固定窗口为2。(15分)
*   **(Q3.2)** 请使用问题2中提供的文本和LanguageModeler来训练一个skip-gram模型。(15分)
*   **(Q3.3)** 请使用给定的维基百科语料库和语言模型来训练CBOW/Skip-gram模型，并比较语义相近的词对（例如{woman, man, men}）之间的词相似度。(15分)

**提供的解答 (来自 HW1\_solution.ipynb):**

*   **Q3.1 CBOW Model:** (代码及输出结果在notebook中)
    
    *   代码修改了数据准备部分以适应CBOW：对于每个目标词，其上下文是左右各 `window_size` (这里是2) 个词。
    *   `LanguageModeler` 的 `forward` 方法修改为对上下文词嵌入求和 ( `embeds = torch.sum(embeds, dim=0)` )，然后通过线性层预测目标词。
    *   使用 `embedding_dim=128`，最终训练损失约为 0.798。
*   **Q3.2 Skip-gram Model:** (代码及输出结果在notebook中)
    
    *   代码修改了数据准备部分以适应Skip-gram：对于每个中心词，其目标是预测其上下文词（左右各 `window_size` 个词）。
    *   `LanguageModeler` 的 `forward` 方法接收中心词的嵌入，`linear2` 层的输出维度变为 `context_length*vocab_size`，然后 `view` 成 `(1, self.context_length, -1)` 以便为每个上下文位置预测一个词。损失函数分别计算每个上下文词的损失并累加（或者NLLLoss可以直接处理）。
    *   使用 `embedding_dim=128`，最终训练损失约为 14.42。
*   **Q3.3 Wikipedia corpus:**
    
    *   "The logic of Question 3.3 is the same as Question 3.2." (实际应为3.1和3.2的逻辑，即训练CBOW或Skip-gram，然后在更大的语料库上进行，并增加词向量相似度计算步骤)

**详细讲解:**

*   **Q3.1: 训练CBOW模型**
    
    *   **CBOW概念:** Continuous Bag-of-Words (CBOW) 模型的思想是用一个词的上下文（周围的词）来预测这个词本身。它将上下文词的词向量（通常是求和或求平均）作为输入，然后通过一个神经网络层来预测中心词。
    *   **代码关键点 (HW1\_solution.ipynb):**
        1.  **数据准备 (`training_data`):**
            *   窗口大小 `window_size = 2`。
            *   遍历文本，对于每个词 `text_list[i]`（作为目标词 `target_word`），其上下文 `context` 是 `text_list[i-window_size:i]` (左侧窗口) 和 `text_list[i+1:i+1+window_size]` (右侧窗口) 的拼接。
        2.  **模型修改 (`LanguageModeler` for CBOW):**
            *   输入是上下文词的索引列表。
            *   `self.embeddings(inputs)` 得到上下文中每个词的嵌入。
            *   `embeds = torch.sum(embeds, dim=0)`: 这是CBOW的核心操作之一，将上下文词的嵌入向量加起来形成一个综合的上下文表示。题目公式中也是 ∑w∈C​qw​。
            *   然后通过 `linear1` 和 `linear2` 预测目标词。
    *   **结果分析:** 最终损失为0.798。这个值本身没有绝对意义，需要对比。但代码成功实现了CBOW的训练流程。
    *   **通用解法 (CBOW):**
        1.  **确定窗口大小 `N`。**
        2.  **生成训练样本:** 对语料库中的每个词 wi​（目标词）：
            *   其上下文 C 为 wi−N​,...,wi−1​,wi+1​,...,wi+N​。
            *   训练样本为 (C,wi​)。
        3.  **模型结构:**
            *   输入层: 上下文 C 中所有词的词嵌入。
            *   合并层: 将上下文词嵌入聚合成单个向量（如求和、平均）。
            *   隐藏层 (可选)。
            *   输出层: Softmax层，输出词汇表中每个词作为中心词的概率。
        4.  **目标:** 最大化给定上下文 C 时目标词 wi​ 的对数似然。
*   **Q3.2: 训练Skip-gram模型**
    
    *   **Skip-gram概念:** Skip-gram模型的思想与CBOW相反，它用一个中心词来预测其上下文（周围的词）。中心词的词向量作为输入，模型尝试预测窗口内的每个上下文词。
    *   **代码关键点 (HW1\_solution.ipynb):**
        1.  **数据准备 (`training_data`):**
            *   窗口大小 `window_size = 2`。
            *   遍历文本，对于每个词 `text_list[i]`（作为中心词 `central_word`），其目标上下文 `target_context` 是 `text_list[i-window_size:i]` 和 `text_list[i+1:i+1+window_size]` 的词列表。
        2.  **模型修改 (`LanguageModeler` for Skip-gram):**
            *   输入是中心词的索引。
            *   `self.embeddings(inputs)` 得到中心词的嵌入。
            *   `self.linear2 = nn.Linear(hidden_size, context_length*vocab_size)`: 输出层的维度需要改变，因为我们要为上下文中的每个位置都预测一个词。`context_length` 是总的上下文词数量 ( `window_size*2` )。
            *   `out = out.view(1, self.context_length, -1)`: 将输出重塑为 (batch\_size, context\_length, vocab\_size)，这样每个上下文位置都有一个词汇表大小的概率分布。
            *   `log_probs = F.log_softmax(out, dim=2)`: 在词汇表维度上进行softmax。
            *   损失函数 `nn.NLLLoss()` 会将 `log_probs` (形状 `(N, C, V)` 或 `(C,V)` 如果batch=1) 和 `target_context_idx` (形状 `(N, C)` 或 `(C)`) 进行比较。这里实现中 `torch.squeeze(log_probs)` 之后，`log_probs` 形状是 `(context_length, vocab_size)`，`target_context_idx` 形状是 `(context_length)`，`NLLLoss` 会正确计算。
    *   **结果分析:** 最终损失为14.42。Skip-gram通常比CBOW训练慢，因为每个中心词会产生多个训练样本（中心词 -> 每个上下文词）。
    *   **通用解法 (Skip-gram):**
        1.  **确定窗口大小 `N`。**
        2.  **生成训练样本:** 对语料库中的每个词 wi​（中心词）：
            *   其上下文 C 中的每个词 wj​∈{wi−N​,...,wi−1​,wi+1​,...,wi+N​}。
            *   训练样本为 (wi​,wj​)。
        3.  **模型结构:**
            *   输入层: 中心词 wi​ 的词嵌入。
            *   隐藏层 (可选，常用于Word2Vec的实现，但这里简单模型可能没有)。
            *   输出层: Softmax层，输出词汇表中每个词作为 wj​ 的概率。
        4.  **目标:** 最大化给定中心词 wi​ 时其所有上下文词 wj​ 的对数似然之和。
*   **Q3.3: 使用维基百科语料库并比较词相似度**
    
    *   **解答说明:** 解决方案中指出逻辑与Q3.1和Q3.2相同。这意味着需要：
        1.  **获取并预处理维基百科语料库:** 这通常是一个很大的文本集合，需要进行清洗、分词等操作。
        2.  **训练CBOW或Skip-gram模型:** 使用这个更大的语料库训练在Q3.1或Q3.2中定义的模型。由于语料库更大，训练时间会更长，可能需要调整超参数（如学习率、epoch数、嵌入维度）。
        3.  **提取词向量:** 训练完成后，`model.embeddings.weight` 就包含了词汇表中每个词的词向量。
        4.  **计算词相似度:** 对于给定的词对（如 {woman, man, men}），提取它们的词向量。计算词向量之间的余弦相似度。
            *   **余弦相似度公式:** similarity(A,B)\=∣∣A∣∣⋅∣∣B∣∣A⋅B​
            *   语义相近的词（如 "woman" 和 "girl"，或 "man" 和 "boy"）应该具有较高的余弦相似度。
            *   可以比较不同词对的相似度，例如 "woman" 与 "man" 的相似度，"man" 与 "men" 的相似度。
    *   **通用步骤 (词向量相似度分析):**
        1.  **选择预训练词向量或自行训练:** 如Word2Vec (CBOW, Skip-gram), GloVe, FastText。
        2.  **加载词向量:** 将词向量加载到内存中，通常是一个词到向量的映射。
        3.  **获取目标词的向量:** 对于要比较的词，从加载的词向量中查找它们的向量表示。
        4.  **计算余弦相似度:** 对每对词的向量应用余弦相似度公式。
        5.  **分析结果:** 比较不同词对的相似度得分，验证是否符合语义直觉。例如，同义词、近义词的相似度应该较高。

### Question 4

**原始题目 (来自 CS6493\_HW1.pdf):**

*   将文本分割成更小的块是一项比看起来更困难的任务。
*   **(Q4.1)** 给定文本: "Peter is too old to do this work, Marry is older than Peter, John is oldest one however he is still doing this work."，一个简单的分词方法是按空格分割。然而，这种简单方法有一些缺点。请列出至少两个缺点并解释。(5分)
*   **(Q4.2)** Transformer模型使用一种介于词级和字符级之间的混合方法，称为子词分词。BPE (Byte-Pair Encoding) 是一种子词级分词方法。BPE依赖于预分词器将训练数据分割成词。假设预分词后，得到词及其频率：(old, 10), (older, 5), (oldest, 8), (hug, 8), (pug, 4), (hugs, 5)。基础词汇表为：o, l, d, e, r, s, t, h, u, g, p。将所有词拆分为基础词汇表中的符号。BPE然后计算每对可能符号对的频率，并选择最频繁的。例如 "o"后跟"l" 出现 10+5+8\=23 次 (题目原文此处为 "o" followed by " "，根据上下文应为 "o" followed by "l"，已在翻译中修正)。第一个合并规则是将 "o" 和 "l" 组合成 "ol"。然后 "ol" 加入词汇表。此过程迭代进行。词汇表大小（基础词汇表大小+合并次数）是超参数。学习到的合并规则应用于新词。实现BPE分词器，设词汇表大小为16，训练并完成迭代过程。用训练好的分词器对 {hold, oldest, older, pug, mug, hugging face}进行分词。(15分)

**提供的解答 (来自 HW1\_solution.ipynb):**

*   **Q4.1:**
    
    1.  非常大的词汇量；
    2.  无法处理词汇表之外的单词；
    3.  无法捕捉相似词之间的语义关系。
    
    _(注意：这里的回答更像是独热编码或简单词袋模型的缺点，而不是空格分词本身的直接缺点。空格分词的主要问题与标点符号、复合词、OOV词处理等有关。下面详细讲解中会补充。)_
    
*   **Q4.2:** (代码及输出结果在notebook中)
    
    *   代码首先从给定的词和频率中构建了初始字符词汇表。
    *   然后有一个计算子词对频率的逻辑（但实现方式似乎是计算所有可能子串的频率，而不是严格的BPE相邻字节对）。
    *   接着选取了频率最高的5个子词（'ol', 'old', 'ld', 'ug', 'olde'）加入词汇表，使得总词汇表大小达到 11 (初始字符) + 5 = 16。
    *   最后，使用这个构建的词汇表对新词进行分词，采用的是一种最长匹配的策略。如果遇到不在词汇表中的单个字符，则标记为 '\[unk\]'。
    *   输出: `[['h', 'old'], ['olde', 's', 't'], ['olde', 'r'], ['p', 'ug'], ['[unk]', 'ug'], ['h', 'ug', 'g', '[unk]', '[unk]', 'g', '[unk]', '[unk]', '[unk]', 'e']]`

**详细讲解:**

*   **Q4.1: 空格分词的缺点**
    
    *   **解答分析 (老师的答案):**
        1.  **非常大的词汇量 (Very large vocabulary size):** 如果将每个按空格切分的单元都视为一个独立的词，词汇表可能会非常大，尤其在包含很多专有名词、数字、罕见词的文本中。
        2.  **无法处理词汇表之外的单词 (Cannot deal with out of vocablary words):** 如果测试时遇到训练时未曾见过的词（OOV词），模型将无法处理。
        3.  **无法捕捉相似词之间的语义关系 (Cannot capture the semantic relations between similar words):** 这更像是后续词表示（如独热编码）的缺点，而不是空格分词本身的直接问题。空格分词后，如果每个词都被视为独立单元，那么 "run" 和 "running" 会被视为两个完全不同的词。
    *   **更直接针对空格分词的缺点:**
        1.  **标点符号处理:** "work," 和 "work" 会被视为不同的词。需要额外的标点符号清洗步骤。
        2.  **多词表达 (Multi-word expressions) / 复合词:** 像 "New York" 或 "state-of-the-art" 这样的多词单元，如果被空格分开，其整体语义可能会丢失。
        3.  **连字符和缩写:** 如何处理 "state-of-the-art" 中的连字符，或 "don't" (do not) 这样的缩写词。简单空格分词会将 "don't" 分为 "don't"。
        4.  **无空格语言:** 对于像中文、日文、泰文这样不使用空格分隔词语的语言，空格分词完全不适用。
        5.  **词形变化:** "older", "oldest" 会被视作与 "old" 完全不同的词，无法体现它们的形态学关系，导致数据稀疏。
*   **Q4.2: BPE分词器实现**
    
    *   **BPE概念:**
        
        1.  **初始化:** 词汇表最初只包含所有单个字符。将语料库中的每个词拆分成字符序列，并统计词频。
        2.  **迭代合并:**
            *   统计当前所有词（已部分合并的）中相邻符号对的出现频率。
            *   选择频率最高的符号对 (A, B)。
            *   将 (A, B) 合并成一个新的符号 AB。
            *   将 AB 添加到词汇表中。
            *   在语料库中，所有出现的 (A, B) 对都替换为 AB。
        3.  \*\*终止条件:\*\*达到预设的词汇表大小，或者达到预设的合并次数。
    *   **代码分析 (HW1\_solution.ipynb):**
        
        1.  **初始词汇表 (`vocab`):** 正确地从输入单词中提取了所有单个字符：`['o', 'l', 'd', 'e', 'r', 's', 't', 'h', 'u', 'g', 'p']` (11个字符)。
        2.  **频率计算 (`freqs`):** 这里的实现（`for i, _ in enumerate(word[0]): for j in range(i+1, len(word[0])):`）计算的是词内所有长度大于1的子串的频率，并累加了词频。这与标准BPE计算 _相邻_ 符号对的频率有所不同。标准BPE是在当前词表示的基础上（例如，初始是字符序列，如 `(o, l, d, 10)`），统计相邻对如 `(o, l)` 在所有词中的出现次数。
        3.  **合并规则学习:** 代码直接对计算出的子串频率排序，并选取了频率最高的5个子串加入词汇表： `vocab.append(sorted_freq_tuples[i][0])`。选出的前5个是 `'ol', 'old', 'ld', 'ug', 'olde'`。这样词汇表大小为 11 + 5 = 16。
            *   _标准BPE的合并过程是迭代的：_ 找到频率最高的 _相邻对_，合并它，更新词的表示，然后 _重新统计_ 新的相邻对频率，再合并，直到达到词汇表大小。
        4.  **分词应用 (`tokenizations`):**
            *   对于给定的新词，它尝试从词的开头开始，找到词汇表中最长的匹配子词。
            *   如果找不到匹配（即使是单个字符也不在词汇表中，这通常是针对未在初始字符集中出现的字符），则输出 `[unk]`。
            *   **示例 "oldest":**
                *   词汇表包含 'olde', 's', 't'。
                *   会匹配到 'olde'，剩余 'st'。
                *   's' 在词汇表中，剩余 't'。
                *   't' 在词汇表中。
                *   所以 "oldest" -> `['olde', 's', 't']`。这与老师的输出一致。
            *   **示例 "mug":**
                *   'm' 不在初始字符词汇表 `['o', 'l', 'd', 'e', 'r', 's', 't', 'h', 'u', 'g', 'p']` 中，也不在合并后的词汇表中。
                *   因此，'m' 会被处理为 `[unk]`。
                *   剩余 'ug'。'ug' 在合并后的词汇表中。
                *   所以 "mug" -> `['[unk]', 'ug']`。这与老师的输出一致。
            *   **示例 "huggingface":**
                *   'h' -> 'h'
                *   'u' -> 'u'
                *   'g' -> 'g' (这里 'hug' 也是一个可能的token，但分解为了 h, u, g，然后下一个 g。实际代码逻辑是贪婪匹配最长前缀)
                *   实际上，老师给的 `huggingface` 分词结果是 `['h', 'ug', 'g', '[unk]', '[unk]', 'g', '[unk]', '[unk]', '[unk]', 'e']`。我们来推演一下：
                    *   `h` in vocab. `tokenization = ['h']`. `subword = "uggingface"`
                    *   `u` in vocab (no, 'ug' is). Longest match is `ug`. `tokenization = ['h', 'ug']`. `subword = "gingface"`
                    *   `g` in vocab. `tokenization = ['h', 'ug', 'g']`. `subword = "ingface"`
                    *   `i` not in vocab. `tokenization = ['h', 'ug', 'g', '[unk]']`. `subword = "ngface"`
                    *   `n` not in vocab. `tokenization = ['h', 'ug', 'g', '[unk]', '[unk]']`. `subword = "gface"`
                    *   `g` in vocab. `tokenization = ['h', 'ug', 'g', '[unk]', '[unk]', 'g']`. `subword = "face"`
                    *   `f` not in vocab. `tokenization = ['h', 'ug', 'g', '[unk]', '[unk]', 'g', '[unk]']`. `subword = "ace"`
                    *   `a` not in vocab. `tokenization = ['h', 'ug', 'g', '[unk]', '[unk]', 'g', '[unk]', '[unk]']`. `subword = "ce"`
                    *   `c` not in vocab. `tokenization = ['h', 'ug', 'g', '[unk]', '[unk]', 'g', '[unk]', '[unk]', '[unk]']`. `subword = "e"`
                    *   `e` in vocab. `tokenization = ['h', 'ug', 'g', '[unk]', '[unk]', 'g', '[unk]', '[unk]', '[unk]', 'e']`. `subword = ""` 这个推演与老师的输出一致。这说明分词阶段的逻辑是正确的，但是BPE学习阶段的频率统计和合并规则选择与标准BPE有差异。
    *   **通用解法 (标准BPE实现思路):**
        
        1.  **预分词和词频统计:** 将训练语料按空格（或其他预分词逻辑）切分成词，并统计每个词的频率。
        2.  **初始化词汇表:** 包含所有单个字符。
        3.  **将词表示为字符序列:** 例如，`("o", "l", "d"): 10`, `("o", "l", "d", "e", "r"): 5`。
        4.  **迭代合并:** 循环 `k` 次 ( `k` = 目标词汇表大小 - 初始字符数量)：
            *   **统计相邻对频率:** 遍历当前所有词的表示（由字符或已合并的子词构成），统计所有相邻符号对的频率。例如，对于 `(o,l,d):10`，相邻对有 `(o,l)` 频率+10，`(l,d)` 频率+10。对于 `(ol, d, e, r):5`（假设 'ol' 已合并），相邻对有 `(ol,d)` 频率+5，`(d,e)` 频率+5，`(e,r)` 频率+5。
            *   **选择最佳对:** 找到频率最高的相邻符号对，例如 `(A, B)`。
            *   **添加新符号:** 将新符号 `AB` 添加到词汇表中。
            *   **更新词表示:** 在所有词的表示中，将所有出现的相邻 `(A, B)` 替换为 `AB`。
        5.  **分词新文本:** 对于新词，贪婪地使用学习到的词汇表（优先匹配最长的已知子词）进行切分。如果字符不在初始词汇表中，则标记为未知。

Homework 2 (CS6493\_HW2\_HK.pdf & HW2solution\_HK.pdf)
------------------------------------------------------

现在我们来看作业2。

### Question 1

**原始题目 (来自 CS6493\_HW2\_HK.pdf):**

*   机器翻译涉及将文本从一种语言自动翻译到另一种语言。
*   **(Q1.1)** 模型需要使用矩阵加速运算，但训练需要批量输入数据。批次中的句子需要等长，所以需要填充 (padding)。图1显示了一个批次的填充过程。（图示：三句话分词后变成数字序列，然后进行padding）。根据你之前指定的数字：`[4,7,9,15,23,18]`, `[2,8,10]`, `[3,13,2,8]`。请写下填充完成后的批次。(4分)
*   **(Q1.2)** 考虑到经常发生内存不足 (OOM)，如何以较小代价选择批量大小 (batch size) 以确保在模型训练的完整周期中不发生内存泄漏？(4分)
*   **(Q1.3)** 集束搜索 (Beam search) 和贪婪搜索 (greedy search) 是序列生成中常见的解码算法。图2是一个搜索树，边表示生成下一个词元的概率。请使用贪婪搜索和集束搜索 (k=2) 来找到最可能的序列。(6分) (图示搜索树)
*   **(Q1.4)** BLEU 是一种流行的机器翻译评估指标。公式为 BLEU\=BP⋅exp(∑n\=1N​wn​log pn​)。给定候选句和两个参考句，当N=4时计算BLEU得分。(6分)
    *   候选句: The quick brown fox jumps over the lazy dog in the park.
    *   参考句1: The fast brown fox jumps over the lazy dog in the park.
    *   参考句2: The speedy brown fox jumps over the lazy dog in the park.

**提供的解答 (来自 HW2solution\_HK.pdf):**

*   **Q1.1:** (如果使用Canvas宣布的修正后的token id) `[4,7,9,15,23,18]` `[2,8,10,0,0,0]` `[3,13,2,8,0,0]`
    
*   **Q1.2:** 确定一个批次大小后，按照批次长度从大到小进行训练。(原文：After determining a batch size, train according to the batch length from large to small.)
    
    _(这个答案比较简洁，可能指先用最大长度的样本确定一个能跑通的batch size，或者是在处理变长序列时，将相似长度的样本组织在一起以减少padding。)_
    
*   **Q1.3:** (图示搜索过程)
    
    *   Greedy: A -> AD -> ADK
    *   Beam (k=2):
        *   Step 1: A (0.6), B (0.3) -> 保持 A, B
        *   Step 2:
            *   From A: AD (0.6_0.5=0.3), AE (0.6_0.3=0.18), AF (0.6\*0.2=0.12)
            *   From B: BH (0.3_0.7=0.21), BI (0.3_0.3=0.09)
            *   Top 2: AD (0.3), BH (0.21) -> 保持 AD, BH _(注意，老师的图示是 AD, BHI，这里可能在图上直接跳了一步或表示了路径，实际beam search是一步步扩展的。老师图上标注的是 Beam: AD, BH I (I在H之后))_
        *   Step 3 (基于AD, BH):
            *   From AD: ADK (0.3\*0.99 = 0.297)
            *   From BH: BHL (0.21_0.6 = 0.126), BHM (0.21_0.4 = 0.084)
            *   Top 2 overall (假设只考虑当前最优的两个路径的扩展): ADK (0.297), BHL (0.126)
    *   老师图中的最终结果: Greedy: ADK, Beam: ADK, BHL
*   **Q1.4:** p1​\=11/12≈0.917 p2​\=9/11≈0.818 p3​\=8/10\=0.8 p4​\=7/9≈0.778 BP\=1 BLEU\=0.8266
    

**详细讲解:**

*   **Q1.1: Padding**
    
    *   **概念:** 在批量处理序列数据时，同一批次内的所有序列通常需要具有相同的长度。Padding是在较短序列的末尾（或开头）添加特殊标记（通常是0）以使其达到与批次中最长序列相同的长度。
    *   **解答分析:** 原始序列（根据题目指定）：
        
        1.  `[4,7,9,15,23,18]` (长度6)
        2.  `[2,8,10]` (长度3)
        3.  `[3,13,2,8]` (长度4) 批次中最长序列的长度是6。因此，所有序列都需要填充到长度6。
        
        *   序列1: `[4,7,9,15,23,18]` (无需填充)
        *   序列2: `[2,8,10]` -> `[2,8,10,0,0,0]` (填充3个0)
        *   序列3: `[3,13,2,8]` -> `[3,13,2,8,0,0]` (填充2个0) 老师的解答与此一致。
    *   **通用解法:**
        1.  确定批次中所有序列的最大长度 `max_len`。
        2.  对于每个序列，如果其长度小于 `max_len`，则在其末尾（或开头，取决于填充策略，右填充更常见）添加 `max_len - current_len` 个填充符号（通常是0）。
*   **Q1.2: 选择批量大小以避免OOM**
    
    *   **OOM (Out Of Memory):** 当模型和数据所需的GPU显存（或内存）超过可用显存时发生。批量大小是影响显存占用的一个关键因素。
    *   **解答分析:** "确定一个批次大小后，按照批次长度从大到小进行训练。"
        *   这个答案有些模糊。一种可能的解释是：为了找到一个安全的`batch_size`，可以先用包含最长序列的样本（这些样本通常最耗显存）来测试不同的`batch_size`。从小`batch_size`开始逐渐增加，直到不发生OOM。
        *   另一种解释与动态批处理有关：对数据按长度排序，然后将长度相似的样本组成批次。这样做的好处是每个批次内的填充量最小，从而提高了计算效率并可能在一定程度上缓解显存压力（因为平均序列长度可能更短）。但是，这并不能直接帮你“选择”一个初始的、能避免OOM的`batch_size`。
    *   **通用策略/建议:**
        1.  **经验初始值:** 根据GPU显存大小和模型大小，可以有一个经验性的初始`batch_size`（例如，对于12GB显存的GPU和中等大小的Transformer，可以从8, 16, 32等开始尝试）。
        2.  **二分查找/逐步增加:**
            *   从小`batch_size`（如1或2）开始，运行一个完整的训练迭代（或几个迭代）。
            *   如果未发生OOM，逐渐增加`batch_size`（例如加倍或增加一个固定值）。
            *   重复此过程，直到发生OOM。此时，最后一个未导致OOM的`batch_size`或略小于它的值可以作为候选。
        3.  **考虑序列长度:** 如果数据集中序列长度差异很大，最长的序列会决定显存瓶颈。可以在测试`batch_size`时，特意使用包含最长序列的批次。
        4.  **梯度累积 (Gradient Accumulation):** 如果单个样本也很大，导致即使`batch_size=1`也OOM，或者希望使用更大的有效批量大小但受限于显存，可以使用梯度累积。即，计算几个小批次的梯度，不清零，累积它们，然后用累积的梯度更新模型参数一次。这模拟了使用大`batch_size`的效果。
        5.  **混合精度训练 (Mixed Precision Training):** 使用如FP16（半精度浮点数）可以减少显存占用约一半，并加速计算，通常与梯度缩放结合使用以保持训练稳定性。
        6.  **监控工具:** 使用 `nvidia-smi` (Linux/Windows) 或类似工具实时监控GPU显存使用情况。
*   **Q1.3: 贪婪搜索与集束搜索 (Greedy Search & Beam Search)**
    
    *   **概念:**
        
        *   **贪婪搜索:** 在每一步解码时，选择当前概率最高的词元作为输出，然后基于这个选择进行下一步。非常简单快速，但容易陷入局部最优，可能错过全局最优序列。
        *   **集束搜索 (Beam Search):** 在每一步解码时，保留 `k` 个（`k` 称为集束宽度 beam width）当前最优的候选序列。在下一步，从这 `k` 个序列分别扩展，再次选出所有扩展结果中总概率最高的 `k` 个序列。这是一种启发式搜索，试图在计算效率和搜索质量之间取得平衡。
    *   **解答分析 (基于题目图示):** 假设初始概率为1。
        
        *   **贪婪搜索 (Greedy Search):**
            
            1.  **Step 0 -> 1:** A (0.6), B (0.3), C (0.1)。最高的是A。序列: A。
            2.  **Step 1 (from A) -> 2:** D (0.5), E (0.3), F (0.2)。最高的是D。序列: AD (总概率 0.6 \* 0.5 = 0.3)。
            3.  **Step 2 (from D) -> 3:** K (0.99)。最高的是K。序列: ADK (总概率 0.3 \* 0.99 = 0.297)。 最终贪婪搜索结果: ADK。
        *   **集束搜索 (Beam Search, k=2):**
            
            1.  **Step 0 -> 1:**
                
                *   A: prob = 0.6
                *   B: prob = 0.3
                *   C: prob = 0.1
                *   Top 2 (beam): \[ (A, 0.6), (B, 0.3) \]
            2.  **Step 1 -> 2 (扩展A和B):**
                
                *   扩展 A:
                    *   AD: prob = 0.6 \* 0.5 = 0.30
                    *   AE: prob = 0.6 \* 0.3 = 0.18
                    *   AF: prob = 0.6 \* 0.2 = 0.12
                *   扩展 B:
                    *   BH: prob = 0.3 \* 0.7 = 0.21
                    *   BI: prob = 0.3 \* 0.3 = 0.09
                *   所有候选路径: (AD, 0.30), (AE, 0.18), (AF, 0.12), (BH, 0.21), (BI, 0.09)
                *   Top 2 (beam): \[ (AD, 0.30), (BH, 0.21) \]
            3.  **Step 2 -> 3 (扩展AD和BH):**
                
                *   扩展 AD:
                    *   ADK: prob = 0.30 \* 0.99 = 0.297
                *   扩展 BH:
                    *   BHL: prob = 0.21 \* 0.6 = 0.126
                    *   BHM: prob = 0.21 \* 0.4 = 0.084
                *   所有候选路径: (ADK, 0.297), (BHL, 0.126), (BHM, 0.084)
                *   Top 2 (beam): \[ (ADK, 0.297), (BHL, 0.126) \] 最终集束搜索结果，如果只看路径: ADK, BHL。这与老师的解答图示一致。
    *   **通用解法 (Beam Search):**
        
        1.  初始化 `k` 个候选序列（通常是最高概率的第一个词，或者一个特殊起始符后最高概率的第一个词）。
        2.  在每个解码步骤 `t`:
            *   对于当前beam中的每个候选序列：
                *   获取该序列的最后一个词。
                *   使用语言模型预测基于当前序列的下一个词的概率分布。
                *   生成所有可能的扩展序列及其累积概率（通常是log概率求和）。
            *   从所有生成的扩展序列中，选择累积概率最高的 `k` 个作为新的beam。
        3.  重复步骤2，直到达到最大长度或生成了结束符。
        4.  最终选择beam中概率最高的完整序列作为输出（有时会使用长度惩罚等进行调整）。
*   **Q1.4: 计算BLEU得分**
    
    *   **BLEU概念:** Bilingual Evaluation Understudy (BLEU) 是一种衡量机器翻译输出与一个或多个专业人工参考翻译之间相似度的指标。它计算n-gram（通常1到4-gram）的匹配精度，并引入简短惩罚因子（Brevity Penalty, BP）来惩罚过短的翻译。
        
    *   **公式:** BLEU\=BP⋅exp(∑n\=1N​wn​log pn​)
        
        *   pn​: n-gram的精度（匹配的n-gram数 / 候选句中n-gram总数），通常使用修正后的精度（clipped precision），即一个候选n-gram的计数不能超过它在任何单个参考句中出现的最大次数。
        *   wn​: n-gram的权重，通常是均匀权重，如 wn​\=1/N (这里N=4，所以 wn​\=0.25)。
        *   BP: 简短惩罚。如果候选句长度 c 小于参考句长度 r，则 BP\=exp(1−r/c)；否则 BP\=1。当有多个参考句时，r 通常是与候选句长度最接近的参考句的长度。
    *   **计算步骤:**
        
        1.  **候选句 (cand):** "The quick brown fox jumps over the lazy dog in the park." (长度 c = 12)
            
        2.  **参考句1 (ref1):** "The fast brown fox jumps over the lazy dog in the park." (长度 r1 = 12)
            
        3.  **参考句2 (ref2):** "The speedy brown fox jumps over the lazy dog in the park." (长度 r2 = 12)
            
        4.  **计算简短惩罚 (BP):**
            
            *   候选句长度 c = 12。
            *   选择与c最接近的参考句长度。这里 r1=12, r2=12。所以 r = 12。
            *   因为 c = r，所以 BP\=1。
        5.  **计算各n-gram的修正后精度 (pn​):**
            
            *   **1-grams (unigrams):**
                
                *   Cand: The, quick, brown, fox, jumps, over, the, lazy, dog, in, the, park (12个)
                *   Ref1: The, fast, brown, fox, jumps, over, the, lazy, dog, in, the, park
                *   Ref2: The, speedy, brown, fox, jumps, over, the, lazy, dog, in, the, park
                *   匹配的1-grams (考虑裁剪):
                    *   The: cand中3次。Ref1中3次，Ref2中3次。Max\_ref\_count = 3。Clipped\_count = min(3,3) = 3。
                    *   quick: cand中1次。Ref1中0次，Ref2中0次。Max\_ref\_count = 0。Clipped\_count = min(1,0) = 0。
                    *   brown: cand中1次。Ref1中1次，Ref2中1次。Max\_ref\_count = 1。Clipped\_count = min(1,1) = 1。
                    *   fox: cand中1次。Ref1中1次，Ref2中1次。Max\_ref\_count = 1。Clipped\_count = min(1,1) = 1。
                    *   jumps: cand中1次。Ref1中1次，Ref2中1次。Max\_ref\_count = 1。Clipped\_count = min(1,1) = 1。
                    *   over: cand中1次。Ref1中1次，Ref2中1次。Max\_ref\_count = 1。Clipped\_count = min(1,1) = 1。
                    *   lazy: cand中1次。Ref1中1次，Ref2中1次。Max\_ref\_count = 1。Clipped\_count = min(1,1) = 1。
                    *   dog: cand中1次。Ref1中1次，Ref2中1次。Max\_ref\_count = 1。Clipped\_count = min(1,1) = 1。
                    *   in: cand中1次。Ref1中1次，Ref2中1次。Max\_ref\_count = 1。Clipped\_count = min(1,1) = 1。
                    *   park: cand中1次。Ref1中1次，Ref2中1次。Max\_ref\_count = 1。Clipped\_count = min(1,1) = 1。
                *   总匹配数 (clipped) = 3+0+1+1+1+1+1+1+1+1 = 11。
                *   候选句中1-gram总数 = 12。
                *   p1​\=11/12。
            *   **2-grams (bigrams):**
                
                *   Cand: (The,quick), (quick,brown), (brown,fox), (fox,jumps), (jumps,over), (over,the), (the,lazy), (lazy,dog), (dog,in), (in,the), (the,park) (11个)
                *   匹配项 (只列出匹配的):
                    *   (brown,fox): Ref1, Ref2 均有。Clipped = 1。
                    *   (fox,jumps): Ref1, Ref2 均有。Clipped = 1。
                    *   (jumps,over): Ref1, Ref2 均有。Clipped = 1。
                    *   (over,the): Ref1, Ref2 均有。Clipped = 1。
                    *   (the,lazy): Ref1, Ref2 均有。Clipped = 1。
                    *   (lazy,dog): Ref1, Ref2 均有。Clipped = 1。
                    *   (dog,in): Ref1, Ref2 均有。Clipped = 1。
                    *   (in,the): Ref1, Ref2 均有。Clipped = 1。
                    *   (the,park): Ref1, Ref2 均有。Clipped = 1。
                *   (The,quick) 不匹配。 (quick,brown) 不匹配。
                *   总匹配数 (clipped) = 9。
                *   候选句中2-gram总数 = 11。
                *   p2​\=9/11。
            *   **3-grams (trigrams):**
                
                *   Cand: (The,quick,brown), (quick,brown,fox), ..., (in,the,park) (10个)
                *   匹配项 (只列出匹配的):
                    *   (brown,fox,jumps): Ref1, Ref2 均有。Clipped = 1。
                    *   (fox,jumps,over): Ref1, Ref2 均有。Clipped = 1。
                    *   (jumps,over,the): Ref1, Ref2 均有。Clipped = 1。
                    *   (over,the,lazy): Ref1, Ref2 均有。Clipped = 1。
                    *   (the,lazy,dog): Ref1, Ref2 均有。Clipped = 1。
                    *   (lazy,dog,in): Ref1, Ref2 均有。Clipped = 1。
                    *   (dog,in,the): Ref1, Ref2 均有。Clipped = 1。
                    *   (in,the,park): Ref1, Ref2 均有。Clipped = 1。
                *   总匹配数 (clipped) = 8。
                *   候选句中3-gram总数 = 10。
                *   p3​\=8/10。
            *   **4-grams (four-grams):**
                
                *   Cand: (The,quick,brown,fox), ..., (dog,in,the,park) (9个)
                *   匹配项 (只列出匹配的):
                    *   (brown,fox,jumps,over): Ref1, Ref2 均有。Clipped = 1。
                    *   (fox,jumps,over,the): Ref1, Ref2 均有。Clipped = 1。
                    *   (jumps,over,the,lazy): Ref1, Ref2 均有。Clipped = 1。
                    *   (over,the,lazy,dog): Ref1, Ref2 均有。Clipped = 1。
                    *   (the,lazy,dog,in): Ref1, Ref2 均有。Clipped = 1。
                    *   (lazy,dog,in,the): Ref1, Ref2 均有。Clipped = 1。
                    *   (dog,in,the,park): Ref1, Ref2 均有。Clipped = 1。
                *   总匹配数 (clipped) = 7。
                *   候选句中4-gram总数 = 9。
                *   p4​\=7/9。
        6.  **计算BLEU:**
            
            *   wn​\=0.25 for n=1,2,3,4.
            *   log p1​\=log(11/12)≈log(0.91667)≈−0.0870
            *   log p2​\=log(9/11)≈log(0.81818)≈−0.2007
            *   log p3​\=log(8/10)\=log(0.8)≈−0.2231
            *   log p4​\=log(7/9)≈log(0.77778)≈−0.2513
            *   ∑wn​log pn​\=0.25×(−0.0870−0.2007−0.2231−0.2513)
            *   \=0.25×(−0.7621)≈−0.190525
            *   BLEU\=BP⋅exp(−0.190525)\=1⋅exp(−0.190525)≈0.8265
        
        老师的解答 p1​\=11/12, p2​\=9/11, p3​\=8/10, p4​\=7/9, BP\=1, BLEU\=0.8266 是正确的。
        
    *   **通用解法 (BLEU计算):**
        
        1.  **准备数据:** 候选句，一个或多个参考句。
        2.  **确定N:** 通常为4。
        3.  **计算BP:**
            *   c = 候选句长度。
            *   r = 与 c 最接近的参考句长度。
            *   If c\>r, BP\=1. Else, BP\=exp(1−r/c).
        4.  **对于每个n从1到N:**
            *   提取候选句中的所有n-grams，并计数每个n-gram的出现次数 ( `count_cand_ngram` )。
            *   对于每个在候选句中出现的n-gram：
                *   找出它在 _每个_ 参考句中出现的最大次数 ( `max_ref_count_ngram` )。
                *   该n-gram的裁剪计数 `clipped_count_ngram = min(count_cand_ngram, max_ref_count_ngram)`。
            *   pn​\=(∑clipped\_count\_ngram)/(∑count\_cand\_ngram)。 (分母是候选句中n-gram总数，分子是裁剪后的匹配总数)。
        5.  **计算BLEU:** BLEU\=BP⋅exp(∑n\=1N​wn​logpn​) (通常 wn​\=1/N; 如果任何 pn​\=0，则 logpn​ 会是负无穷，通常会使用平滑或者如果一个 pn​\=0 则整个BLEU为0，或者只加非零的 pn​ )。在实际计算中，为了避免 log(0)，如果 pn​\=0，通常会加一个很小的平滑数，或者直接导致该项为负很大的值，使得整体BLEU趋近于0。许多库会处理这个问题。

### Question 2

**原始题目 (来自 CS6493\_HW2\_HK.pdf):**

*   传统机器翻译主要基于Seq2Seq模型 (由RNN组成)。为解决其问题，提出了注意力机制。给定查询矩阵Q，键值对{K,V}，缩放点积注意力为: Attention(Q,K,V)\=softmax(dk​![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702
    c-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14
    c0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54
    c44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10
    s173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429
    c69,-144,104.5,-217.7,106.5,-221
    l0 -0
    c5.3,-9.3,12,-14,20,-14
    H400000v40H845.2724
    s-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7
    c-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47z
    M834 80h400000v40h-400000z"></path></svg>)​QKT​)V。
*   **(Q2.1)** 传统Seq2Seq模型有哪些缺陷？注意力机制相比Seq2Seq有何改进？(4分)
*   **(Q2.2)** 注意力公式中Q, K, V的本质是什么？为什么Transformer使用不同权重矩阵生成Q和K，为什么不能用相同值做点乘？(4分)
*   **(Q2.3)** 实现注意力，首先计算Q和每个K的注意力得分（相似度）。然后用softmax归一化。Softmax函数的作用是什么？公式中在归一化前用因子 dk​![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702
    c-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14
    c0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54
    c44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10
    s173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429
    c69,-144,104.5,-217.7,106.5,-221
    l0 -0
    c5.3,-9.3,12,-14,20,-14
    H400000v40H845.2724
    s-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7
    c-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47z
    M834 80h400000v40h-400000z"></path></svg>)​1​ 缩放乘积，目的是什么？(4分)
*   **(Q2.4)** Transformer在各种NLP任务中表现优异。我们通常在自回归范式中解决seq2seq任务。LSTM可以顺序编解码。Transformer本质上不是顺序模型。那么Transformer哪部分设计用于解决自回归问题，如何工作？图3是Transformer架构。(8分)

**提供的解答 (来自 HW2solution\_HK.pdf):**

*   **Q2.1:**
    *   **缺陷 (三选一):**
        *   固定长度上下文：编码器生成固定长度向量，可能丢失长序列信息。
        *   信息瓶颈：固定长度向量作为解码器初始状态，压缩表示可能成为瓶颈。
        *   缺乏对齐：Seq2Seq不显式建模输入输出对齐。
    *   **改进 (三选一):**
        *   可变长度上下文：解码器可动态关注输入不同部分。
        *   增强信息流：解码器可访问编码器所有隐藏状态的加权组合，克服瓶颈。
        *   对齐建模：注意力显式建模输入输出对齐。
*   **Q2.2:**
    *   Q, K, V: 输入词向量矩阵X的线性变换。
    *   为何不同权重生成Q, K: 可以增加表达能力，注意力得分矩阵的泛化能力更高。
*   **Q2.3:**
    *   **Softmax作用:**
        *   点积值域 (−∞,+∞)，直接与V相乘值域也如此。Softmax将其归一化到(0,1)区间。
        *   同时起到缩放梯度的作用（防止负数和过大结果导致梯度问题）。
    *   **Scale作用 (dk​![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702
        c-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14
        c0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54
        c44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10
        s173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429
        c69,-144,104.5,-217.7,106.5,-221
        l0 -0
        c5.3,-9.3,12,-14,20,-14
        H400000v40H845.2724
        s-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7
        c-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47z
        M834 80h400000v40h-400000z"></path></svg>)​1​):** 矩阵点乘可能导致值指数级增大，使softmax梯度非常小，用它缩放避免此问题。
*   **Q2.4:**
    *   最重要的是掩码多头注意力 (Masked Multi-Head Attention) (6分)，它使Transformer能以自回归方式工作。具体来说，掩码注意力帮助模型在解码阶段只关注之前的词元。
    *   此外，位置编码 (Positional Encoding) (2分) 在建模序列特征方面也起重要作用。PE使Transformer能区分不同位置的词元。

**详细讲解:**

*   **Q2.1: Seq2Seq缺陷与注意力改进**
    
    *   **传统Seq2Seq缺陷总结:** 主要问题在于将整个输入序列压缩成一个固定大小的上下文向量 (context vector)，通常是编码器最后一个时刻的隐藏状态。
        1.  **信息丢失/瓶颈:** 对于长序列，这个固定大小的向量很难捕捉所有重要信息，导致信息丢失。这个向量成为解码器唯一的信息来源，形成“信息瓶颈”。
        2.  **对长序列性能下降:** 随着输入序列长度增加，模型性能急剧下降。
        3.  **无法有效对齐:** 解码器在生成每个词时，无法知道应该重点关注输入序列的哪个部分。
    *   **注意力机制的改进:**
        1.  **动态上下文:** 允许解码器在生成每个输出词时，动态地“关注”输入序列的不同部分。它不是依赖单一的固定上下文向量，而是为每个输出词计算一个加权的上下文向量，这个向量是编码器所有隐藏状态的加权和。
        2.  **缓解信息瓶颈:** 通过直接访问编码器的所有隐藏状态，信息流更直接，减少了信息在单一固定向量中被压缩的程度。
        3.  **显式对齐:** 注意力权重本身就提供了一种输入和输出之间的“软对齐”信息，显示了在生成某个输出词时，输入序列的哪些部分贡献更大。
        4.  **提升长序列性能:** 由于能够更好地处理长距离依赖和信息流动，注意力机制显著改善了模型在长序列上的表现。
*   **Q2.2: Q, K, V的本质及不同权重矩阵**
    
    *   **Q (Query), K (Key), V (Value) 的本质:**
        *   在自注意力 (self-attention) 中（如Transformer编码器/解码器内部），Q, K, V通常是同一个输入序列（例如，词嵌入加上位置编码）经过 _不同_ 的线性变换（乘以不同的权重矩阵 WQ​,WK​,WV​）得到的。
        *   Q\=XWQ​
        *   K\=XWK​
        *   V\=XWV​
        *   可以把注意力机制类比于信息检索：
            *   **Query (Q):** 代表当前时间步或当前位置想要查询的信息（“我正在寻找什么？”）。
            *   **Key (K):** 代表序列中其他位置可以被查询的“标签”或“索引”（“这些位置有什么信息？”）。
            *   **Value (V):** 代表序列中其他位置实际存储的信息内容。
        *   通过计算Q和所有K的相似度（通常是点积），得到注意力权重，这些权重然后用于对所有的V进行加权求和，得到最终的输出。
    *   **为什么Transformer使用不同权重矩阵生成Q和K (而不是用X自身做点乘 XXT)?**
        
        1.  **赋予不同角色/功能空间:** 使用不同的线性变换 WQ​,WK​,WV​ 可以将原始输入映射到不同的表示空间，使得Q, K, V可以扮演不同的角色。如果都用X，那么Q和K就在同一个空间，限制了模型从不同角度捕捉信息的能力。例如，Q可能更关注“我需要什么类型的依赖关系”，而K则表示“我能提供什么类型的依赖关系”。
        2.  **增加模型表达能力和灵活性:** 不同的权重矩阵 (WQ​,WK​) 允许模型学习如何最好地将输入转换为适合计算相似度的查询和键。这比直接使用原始输入X进行点积（XXT）提供了更大的灵活性和表达力。模型可以学习到最优的投影方式来提取最相关的特征用于比较。
        3.  **提高泛化能力:** 老师答案中提到“注意力得分矩阵的泛化能力更高”。通过学习到的投影，模型可以更好地泛化到未见过的数据，而不是仅仅依赖原始输入特征之间的直接关系。不同的投影可以帮助模型识别更抽象和鲁棒的模式。
        
        *   如果Q和K来自相同的原始向量X且没有经过不同的线性变换（即 WQ​\=WK​\=I 或 WQ​\=WK​\=I 但 WQ​,WK​ 功能类似），那么计算出的相似度可能更多地反映了词元自身的某些固有属性，而不是它们在特定上下文中的动态关系。不同的 WQ​,WK​ 允许模型根据上下文动态地调整查询和键的表示，从而计算出更具上下文感知性的注意力权重。
*   **Q2.3: Softmax和缩放因子的作用**
    
    *   **Softmax函数的作用:**
        1.  **归一化为概率分布:** QKT 计算出的原始注意力得分（也叫logits）可以是任意实数。Softmax函数将这些得分转换成一个概率分布，即所有得分都转换到 (0, 1) 区间，并且它们的和为1。这使得注意力权重可以被解释为“关注到输入序列各个部分的概率”。
        2.  **突出重要部分:** Softmax具有“赢者通吃”的特性（虽然不完全是，但会放大差异），它会使得原始得分中较高的值在转换后获得更高的概率，而较低的值获得更低的概率，从而帮助模型更集中地关注那些真正相关的部分。
        3.  **梯度处理:** 如老师答案所述，确保输出值为正，并且值域受控，有助于梯度的稳定传播。
    *   **缩放因子 dk​![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702
        c-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14
        c0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54
        c44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10
        s173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429
        c69,-144,104.5,-217.7,106.5,-221
        l0 -0
        c5.3,-9.3,12,-14,20,-14
        H400000v40H845.2724
        s-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7
        c-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47z
        M834 80h400000v40h-400000z"></path></svg>)​1​ 的作用:**
        *   dk​ 是键（K）向量的维度（也是查询Q向量的维度）。
        *   **防止点积过大:** 当 dk​ 的值较大时，Q⋅K 的点积结果的方差会增大，可能会导致一些点积结果变得非常大（绝对值大）。
        *   **避免Softmax梯度饱和:** 如果点积结果过大，Softmax函数可能会进入其饱和区域（即输入值很大或很小时，梯度接近于0）。在这些区域，梯度非常小，会导致模型训练缓慢或停止学习。
        *   **稳定训练:** 通过除以 dk​![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702
            c-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14
            c0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54
            c44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10
            s173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429
            c69,-144,104.5,-217.7,106.5,-221
            l0 -0
            c5.3,-9.3,12,-14,20,-14
            H400000v40H845.2724
            s-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7
            c-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47z
            M834 80h400000v40h-400000z"></path></svg>)​ 进行缩放，可以将点积结果的方差控制在1左右（如果Q和K的元素初始方差为1），使得Softmax的输入值保持在一个更合理的范围内，从而产生更平滑的注意力分布和更稳定的梯度，有助于训练过程。
*   **Q2.4: Transformer如何处理自回归任务**
    
    *   **自回归 (Auto-regressive):** 指模型在生成序列时，每一步的输出都依赖于之前已经生成的输出。例如，在翻译任务中，生成第 t 个目标词时，需要依赖第 1,...,t−1 个已生成的目标词。
    *   **Transformer中的关键设计:**
        1.  **掩码多头注意力 (Masked Multi-Head Attention) 在解码器中 (6分):**
            *   这是Transformer解码器能够进行自回归生成的关键。
            *   在解码器对目标序列进行自注意力计算时（即Q, K, V都来自目标序列本身），会使用一个“后续掩码”(look-ahead mask / subsequent mask)。
            *   这个掩码是一个上三角形矩阵（或下三角形，取决于实现），它将当前位置之后的所有位置的注意力权重设置为一个非常小的负数（通常是负无穷），这样在Softmax之后这些位置的权重就变成了0。
            *   **效果:** 这确保了在预测第 i 个词时，模型只能关注到第 i 个词本身以及它之前（1 到 i−1）的词，而不能“看到”或利用它之后的词 (i+1 到末尾)。这就强制了自回归的特性。
        2.  **位置编码 (Positional Encoding) (2分):**
            *   Transformer本身没有像RNN那样的循环结构来天然地处理序列顺序。它同时处理输入序列中的所有词元（并行计算）。
            *   为了让模型能够利用词元在序列中的位置信息，Transformer在词嵌入中加入了位置编码。
            *   位置编码是与词元位置相关的向量，它为每个位置提供了一个独特的信号。这样，即使词元相同，但如果它们在序列中的位置不同，模型也能区分它们。
            *   这对于序列任务至关重要，因为词序通常包含重要的语义信息。
        3.  **解码器的整体结构:** 解码器在每个时间步 t：
            *   接收之前生成的目标词序列 (y1​,...,yt−1​) 作为输入（经过嵌入和位置编码）。
            *   通过掩码自注意力机制处理这个部分生成的目标序列。
            *   然后通过编码器-解码器注意力机制（也叫交叉注意力），关注编码器输出的整个源序列表示。
            *   最后通过前馈网络预测下一个目标词 yt​ 的概率分布。
            *   这个过程逐词进行，直到生成结束符或达到最大长度。

### Question 3

**原始题目 (来自 CS6493\_HW2\_HK.pdf):**

*   预训练模型可分为：基于编码器的（如BERT，自编码），基于解码器的（如GPT，自回归），编码器-解码器模型（如T5, BART）。
*   **(Q3.1)** 自然语言理解 (NLU) 和自然语言生成 (NLG) 任务有什么区别？(2分)
*   **(Q3.2)** 请分别回答并解释基于编码器/解码器/编码器-解码器的预训练模型是否是NLU和NLG任务的好选择。(4分)
*   **(Q3.3)** BERT的双向性使其难用于NLG。UNILM通过不同掩码策略实现统一模型。给定掩码片段 "Jill Birt \[MASK\] outdoor activities" (源) 和 "He was \[MASK\] hiking" (目标)，填写表格显示三种语言模型预训练目标的掩码设计。(15分) (表格省略，但有行列标签)
*   **(Q3.4)** 为这些掩码策略实现三个函数：单向LM掩码，双向LM掩码，seq2seq LM掩码。(9分)

**提供的解答 (来自 HW2solution\_HK.pdf):**

*   **Q3.1:**
    *   NLU关注理解和解释人类语言输入，而NLG专注于生成类人语言输出。
*   **Q3.2:**
    *   基于编码器的模型：NLU任务的好选择。可用于NLG，但非首选。
    *   基于解码器的模型：NLG任务的好选择。通常不直接用于NLU，因为它只捕获单向特征。
    *   编码器-解码器模型：NLU和NLG任务的好选择。
*   **Q3.3:** (表格形式展示了0和1的掩码矩阵)
    *   **单向LM掩码:** 对角线及以上为1 (mask)，对角线以下为0 (attend)。表示每个token只能attend到自身及之前的tokens。
    *   **双向LM掩码:** 全0矩阵。表示每个token可以attend到所有tokens。
    *   **Seq2Seq LM掩码:**
        *   源片段内部：双向可见（全0子矩阵）。
        *   目标片段内部：单向可见（下三角为0，对角线及以上为1的子矩阵）。
        *   源片段对目标片段：目标片段可以attend到所有源片段tokens（对应子矩阵全0）。
        *   目标片段对源片段：源片段不能attend到目标片段tokens（对应子矩阵全1）。
*   **Q3.4:** (Python代码实现)
    *   `mask_for_unidirectional_lm`: 返回 `torch.triu(torch.ones(max_len, max_len), diagonal=1)` (上三角为1，对角线为0)。
    *   `mask_for_bidirectional_lm`: 返回 `torch.zeros(max_len, max_len)`。
    *   `mask_for_seq2seq_lm`: 构造一个分块矩阵，源->源是全0，源->目标是全1，目标->源是全0，目标->目标是上三角为1。 _(注意：老师代码中 `mask_for_seq2seq_lm` 对于源到目标是 `torch.ones(src_len, tar_len)`，这意味着源片段的token不能attend到目标片段的token，这在标准seq2seq中是合理的，因为编码器处理完源之后，解码器才开始生成目标。而目标到源是 `torch.zeros(tar_len, src_len)`，表示目标可以attend到所有源。这与UNILM论文中描述的Seq2Seq LM任务中“目标段中的随机掩码词元，我们使用整个源段及其在目标段中的左侧词元来预测它”是一致的。)_

**详细讲解:**

*   **Q3.1: NLU vs NLG**
    
    *   **自然语言理解 (NLU - Natural Language Understanding):** 目标是让机器能够“理解”人类语言的含义。任务通常是将非结构化的文本输入映射到结构化的表示，或从中提取信息、进行分类等。
        *   例子：文本分类（情感分析、主题识别）、命名实体识别、关系抽取、问答（理解问题并从文本中找到答案）、机器阅读理解、意图识别。
    *   **自然语言生成 (NLG - Natural Language Generation):** 目标是让机器能够以自然、流畅、准确的方式“生成”人类语言文本。
        *   例子：机器翻译（生成另一种语言的文本）、文本摘要（生成简短的文本摘要）、对话系统（生成回复）、故事生成、代码生成（从描述生成代码）。
*   **Q3.2: 不同预训练模型架构与任务的匹配**
    
    *   **基于编码器的模型 (Encoder-based, e.g., BERT):**
        *   **架构特点:** 通常使用双向Transformer编码器（如BERT的Masked Language Model, MLM，可以同时看到左右上下文来预测被mask的词）。
        *   **NLU任务:** 非常适合。因为NLU任务通常需要对整个输入文本有深刻的双向上下文理解。例如，情感分析需要理解整个句子的含义。
        *   **NLG任务:** 不直接适合生成自由文本。因为其预训练方式和结构（尤其是双向性）不符合NLG任务从左到右逐词生成的自回归特性。可以用于NLG任务的某些方面，如作为NLG模型的编码器部分，或用于基于编辑/填充的生成任务，但不是纯粹的自由文本生成的首选。
    *   **基于解码器的模型 (Decoder-based, e.g., GPT):**
        *   **架构特点:** 通常使用单向Transformer解码器（从左到右处理信息，每个词只能看到它之前的词）。预训练目标通常是标准的语言模型（预测下一个词）。
        *   **NLG任务:** 非常适合。其自回归的特性天然匹配文本生成过程。
        *   **NLU任务:** 可以用于NLU任务，但可能不是最优选择，因为它们在预训练时只考虑了单向上下文。对于需要深度双向理解的任务，可能不如BERT类模型。但通过prompting等技术，GPT类模型在NLU任务上也表现出强大能力。
    *   **编码器-解码器模型 (Encoder-Decoder, e.g., T5, BART):**
        *   **架构特点:** 包含一个编码器（通常是双向的，用于理解输入序列）和一个解码器（通常是自回归的，用于生成输出序列）。
        *   **NLU任务:** 适合。可以将NLU任务构建成seq2seq的形式。例如，文本分类可以看作是将输入序列映射到一个类别标签序列（通常是一个词）。
        *   **NLG任务:** 非常适合。这是它们设计的初衷，如机器翻译、摘要等。BART使用去噪自编码器作为预训练目标，T5将所有任务都转换为文本到文本的格式。
*   **Q3.3: UNILM掩码设计 (0表示可以attend，1表示不可以attend/被mask)** 上下文：源片段 "Jill Birt \[M1\] outdoor activities" (S1 S2 S3 S4 S5)，目标片段 "He was \[M2\] hiking" (T1 T2 T3 T4)。其中 \[M1\] 是源的掩码词，\[M2\] 是目标的掩码词。我们要预测 \[M1\] 和 \[M2\]。
    
    1.  **单向LM (Unidirectional LM):** 预测被mask的词时，只能使用其左侧上下文和自身。
        
        *   对于源片段和目标片段都应用从左到右的掩码。
        *   当预测 S3(\[M1\]) 时，S3只能attend到 S1, S2, S3。
        *   当预测 T3(\[M2\]) 时，T3只能attend到 T1, T2, T3。
        *   这意味着注意力矩阵中，(row, col) 如果 col > row，则为1 (mask)，否则为0 (attend)。
        *   老师的表格表示的是 (i,j) 位置的值代表 token i 是否可以 attend 到 token j。1代表mask。因此，对于单向LM，当 j\>i 时， Mij​\=1。对角线 Mii​\=0。
    2.  **双向LM (Bidirectional LM):** 预测被mask的词时，可以使用整个上下文。
        
        *   对于源片段和目标片段都应用双向可见性。
        *   当预测 S3(\[M1\]) 时，S3可以attend到 S1, S2, S3, S4, S5, T1, T2, T3, T4 (如果是联合处理的话，但UNILM中通常是分开处理或有特定段标记)。如果仅在片段内双向，则S3可attend到S1-S5。
        *   UNILM的Bidirectional LM通常应用于单个片段，类似于BERT的MLM。如果两个片段是拼接的，那么就是全局双向。
        *   老师的表格显示了一个全0矩阵，表示所有token之间都可以相互attend。这是最宽松的全局双向注意力。
    3.  **序列到序列LM (Sequence-to-Sequence LM):**
        
        *   **预测源片段中的掩码词 (如S3=\[M1\]):** 可以使用整个源片段。S3可以attend到S1, S2, S3, S4, S5。但不能看到目标片段。
        *   **预测目标片段中的掩码词 (如T3=\[M2\]):** 可以使用整个源片段 (S1-S5) 以及目标片段中该词左侧的词和自身 (T1, T2, T3)。
        *   **掩码矩阵构造:**
            *   **源 -> 源 (S\_i attend S\_j):** 全0 (双向可见)。
            *   **源 -> 目标 (S\_i attend T\_j):** 全1 (源不能看到目标)。
            *   **目标 -> 源 (T\_i attend S\_j):** 全0 (目标可以完全看到源)。
            *   **目标 -> 目标 (T\_i attend T\_j):** 单向可见 (如果 j\>i, Mij​\=1, 否则 Mij​\=0 )。 老师提供的表格与此描述一致。
*   **Q3.4: 掩码函数实现**
    
    *   `mask_for_unidirectional_lm(source_seg, target_seg)`:
        *   `max_len = len(source_seg) + len(target_seg)`
        *   `torch.triu(torch.ones(max_len, max_len), diagonal=1)`: 创建一个上三角矩阵（不包括对角线），值为1。这意味着位置 i 的token只能attend到位置 j≤i 的token。这符合单向从左到右的注意力。
    *   `mask_for_bidirectional_lm(source_seg, target_seg)`:
        *   `return torch.zeros(max_len, max_len)`: 全0矩阵，表示所有token之间都可以相互attend，即双向注意力。
    *   `mask_for_seq2seq_lm(source_seg, target_seg)`:
        *   `src_len, tar_len = len(source_seg), len(target_seg)`
        *   `encode_mask = torch.cat((torch.zeros(src_len, src_len), torch.ones(src_len, tar_len)), dim=1)`
            *   左上块 (src->src): `torch.zeros(src_len, src_len)` (源内部双向)
            *   右上块 (src->tar): `torch.ones(src_len, tar_len)` (源不能attend到目标)
        *   `decode_mask = torch.cat((torch.zeros(tar_len, src_len), torch.triu(torch.ones(tar_len, tar_len), diagonal=1)), dim=1)`
            *   左下块 (tar->src): `torch.zeros(tar_len, src_len)` (目标可以attend到所有源)
            *   右下块 (tar->tar): `torch.triu(torch.ones(tar_len, tar_len), diagonal=1)` (目标内部单向)
        *   `return torch.cat((encode_mask, decode_mask), dim=0)`: 将两部分拼接起来。 这个实现与Q3.3的分析一致。在注意力计算中，通常是将mask中为1的位置的注意力分数设为一个非常小的负数，这样softmax后这些位置的概率就接近于0。

### Question 4

**原始题目 (来自 CS6493\_HW2\_HK.pdf):**

*   QA任务有两个著名数据集SQuAD v1.1和v2.0。v2.0允许段落中可能没有答案。对于v1.1，将问题和段落表示为单个序列。引入起始向量S和结束向量E。词i是答案开始的概率为 Pi​\=∑j​eS⋅Tj​eS⋅Ti​​，结束类似。候选span(i,j)的得分为 S⋅Ti​+E⋅Tj​。
*   **(Q4.1)** SQuAD v2.0引入了无答案问题。v1.1的方法必须预测答案。请扩展基于BERT的SQuAD v2.0方法来解决此问题。详细说明设计和工作流程。(15分)
*   **(Q4.2)** QA方法如BiDAF和BERT本质是找答案边界。BERT预训练目标MLM和NSP只学习词元和句子级信息，不含span信息。因此原始BERT预训练目标对QA可能不是最佳。请为BERT设计适合QA任务和其他基于span的任务（如共指消解）的预训练目标。(15分)

**提供的解答 (来自 HW2solution\_HK.pdf):**

*   **Q4.1:** (来自BERT论文的答案)
    
    *   将无答案问题视为答案span的开始和结束都在\[CLS\]词元。
    *   预测时，比较无答案span的得分 snull​\=S⋅C+E⋅C (C是\[CLS\]的表示) 与最佳非空span的得分 si,j∗​\=maxj≥i​(S⋅Ti​+E⋅Tj​)。
    *   当 si,j∗​≥snull​+τ 时预测非空答案，阈值 τ 在开发集上选择以最大化F1。
*   **Q4.2:** (来自SpanBERT论文的答案)
    
    *   **Span Masking:** 随机采样文本片段(span)进行mask，而不是随机mask单个token。优先mask完整单词。Span长度从几何分布中采样。
    *   **Span Boundary Objective (SBO):** 目标是仅使用span边界观察到的token的表示来预测被mask的span中的每个token。
        *   对于被mask的span (xs​,...,xe​) 中的每个token xi​，使用其左右边界外部的token xs−1​ 和 xe+1​ 的输出编码，以及 xi​ 相对于左边界的相对位置嵌入 pi−s+1​ 来预测 xi​。
        *   yi​\=f(xs−1​,xe+1​,pi−s+1​)，f 是一个2层前馈网络。
        *   然后用 yi​ 预测 xi​，计算交叉熵损失，类似MLM。

**详细讲解:**

*   **Q4.1: BERT处理SQuAD v2.0 (无答案问题)**
    
    *   **SQuAD v1.1 vs v2.0:**
        *   v1.1: 每个问题在给定的段落中保证有答案。模型任务是预测答案文本在段落中的起始和结束位置。
        *   v2.0: 问题可能在段落中没有答案。模型不仅要能找到答案，还要能判断何时没有答案。
    *   **BERT (v1.1) 的做法:**
        1.  输入: 将问题(Q)和段落(P)拼接成一个序列：`[CLS] Q [SEP] P [SEP]`。
        2.  输出: 对于段落P中的每个词元 Ti​，BERT输出其最终的隐藏表示。
        3.  预测: 训练两个新的向量，起始向量 S 和结束向量 E。
            *   词元 i 作为答案开始的得分：S⋅Ti​ (点积)。
            *   词元 j 作为答案结束的得分：E⋅Tj​。
            *   通过Softmax将这些得分转换为概率。
            *   候选答案span (i,j) 的得分是 S⋅Ti​+E⋅Tj​。选择得分最高的且 j≥i 的span作为答案。
    *   **扩展到SQuAD v2.0 (如BERT论文所述):**
        1.  **将\[CLS\]视为可能的答案:**
            *   允许答案的开始和结束位置都指向特殊的 `[CLS]` 词元。如果模型预测答案的开始和结束都是 `[CLS]`，则意味着没有答案。
        2.  **计算无答案得分 (snull​):**
            *   让 C 表示 `[CLS]` 词元的最终隐藏表示。
            *   无答案span的得分 snull​\=S⋅C+E⋅C。
        3.  **计算最佳非空答案得分 (si,j∗​):**
            *   遍历段落P中所有可能的span (i,j) (其中 j≥i)，计算它们的得分 S⋅Ti​+E⋅Tj​。
            *   si,j∗​\=maxj≥i​(S⋅Ti​+E⋅Tj​)。
        4.  **最终预测:**
            *   引入一个可调的阈值 τ。
            *   如果 si,j∗​≥snull​+τ，则预测最佳的非空答案span (i,j)。
            *   否则（即最佳非空答案的得分不够高，不足以超过无答案得分加上一个偏差量），则预测“无答案”。
            *   阈值 τ 在开发集上进行调整，以最大化针对SQuAD v2.0的官方评估指标（通常是F1分数，同时考虑了找到正确答案和正确判断无答案的能力）。
    *   **工作流程:**
        1.  **训练:** 模型在SQuAD v2.0数据集上进行微调。对于有答案的问题，损失函数促使模型预测正确的start/end logits。对于无答案的问题，损失函数促使模型预测 `[CLS]` 作为start/end，或者使 snull​ 相对于其他 si,j​ 更高。
        2.  **预测:** 对新问题和段落，计算所有可能的span得分以及 snull​。使用学习到的阈值 τ 来决定是输出一个答案span还是声明无答案。
*   **Q4.2: 为QA和Span任务设计BERT预训练目标 (SpanBERT)**
    
    *   **原始BERT预训练的局限性:**
        *   **Masked Language Model (MLM):** 随机mask掉15%的 _单个词元_ (subword tokens)，然后预测这些被mask的词元。这主要学习词元级别的表示。
        *   **Next Sentence Prediction (NSP):** 判断两个句子A和B是否是原文中连续的。这学习句子级别的关系。
        *   对于需要理解和预测文本 _片段 (spans)_ 的任务（如QA中的答案span，共指消解中的指代span），MLM和NSP没有直接对span进行建模。
    *   **SpanBERT提出的新预训练目标:**
        1.  **Span Masking (替换MLM中的随机单token masking):**
            *   **如何mask:** 不是mask单个词元，而是随机选择一个连续的文本 _span_ (一段词元) 进行mask。
            *   **Span长度:** Span的长度从一个几何分布中采样，这意味着更倾向于采样较短的span，但也会采样较长的span。
            *   **整词masking:** 确保被mask的span对应完整的词，而不是词的一部分 (subword)。
            *   **预算:** 总的mask比例仍然保持在15%左右。
        2.  **Span Boundary Objective (SBO - 新增目标):**
            *   **动机:** 答案span通常由其边界词（开始和结束词）决定。SBO的目标是让模型学习从span的边界信息来预测span内部的内容。
            *   **如何工作:**
                *   对于一个被mask的span (xs​,...,xe​)（其中 xs​ 是开始词元，xe​ 是结束词元）。
                *   对于这个span内部的 _每一个_ 被mask的词元 xi​ (s≤i≤e)：
                    *   模型只能使用该span _外部_ 的边界词元的表示，即 xs−1​ (左边界词的BERT输出) 和 xe+1​ (右边界词的BERT输出)，以及 xi​ 相对于左边界 xs−1​ 的 _相对位置嵌入_ pi−s+1​。
                    *   将这三者 (xs−1​ 的表示, xe+1​ 的表示, pi−s+1​) 输入到一个小的两层前馈网络 f，得到 xi​ 的预测表示 yi​\=f(H(xs−1​),H(xe+1​),Pi−s+1​) (其中H是BERT的输出编码，P是位置嵌入)。
                    *   用 yi​ 来预测原始的被mask词元 xi​（通过一个输出层和softmax，类似MLM的预测头）。
                *   损失函数是所有被mask的span内部所有词元的预测损失之和（交叉熵损失）。
        3.  **NSP:** SpanBERT移除了NSP任务，发现它对下游任务性能帮助不大，有时甚至有害。
    *   **为什么这些目标对QA和Span任务更好:**
        *   **Span Masking:** 迫使模型学习连续文本片段的表示，而不是孤立的词元。这更接近于答案span的特性。
        *   **SBO:** 直接训练模型根据边界信息来理解和重构span内部的内容。这对于需要精确确定span边界的任务（如QA）非常有益。模型学会了将span的语义信息编码到其边界表示中。
    *   **通用设计思路 (为特定下游任务设计预训练目标):**
        1.  **分析下游任务的特性:** 确定下游任务成功的关键是什么（例如，是词元理解、句子关系、片段理解、长距离依赖等）。
        2.  **设计与特性匹配的预训练任务:**
            *   **Masking策略:** 如何选择mask的内容（单个token、span、实体等）？
            *   **预测目标:** 预测被mask的内容本身？预测其属性？预测其与其他部分的关系？
            *   **输入表示:** 在预测时，允许模型看到哪些上下文信息？
        3.  **评估有效性:** 在相关的下游任务上评估新的预训练模型，看性能是否有提升。

希望这些详细的讲解能帮助你更好地理解这些题目和解决方案！

信息来源和相关内容

---

翻译为中文并用中文讲解1.1 To design such a task-oriented dialogue system, we need to first perform natural language understanding tasks, including domain classification, intent detection, and slot filling. Explain in this CRB system, what is the domain classification problem, what is the intent detection problem, and what is the slot filling problem. You can design 5 reasonable slots to indicate this CRB example. \[9 marks\]

  

1.2 What are the next-step tasks/modules after the natural language understanding tasks to finish this closed-loop system? Explain how these next-step tasks/modules work using this CRB example. \[9 marks\]

  

1.3 Can you provide at least two metrics to evaluate this CRB system performance? Explain their meanings. \[4 marks\]

  

2.1 What is the distributional hypothesis in word representations? What is the key problem that the contextualized word representation (like ELMo) solves, compared with context-independent word representations (like word2vec or GloVe)? \[4 marks\]

  

 2.2 If we select the skip-gram model as our training strategy on a larger corpus, and then start the training process. However, we found that the training speed is very slow. Why does this problem occur and how to improve the efficiency of training? Give at least one of these possible solutions and explain it. \[6 marks\]

  

 2.3 After obtaining a well-trained language model, we incorporate it with a multi-layer recurrent neural network (RNN) to address the named entity recognition task. A major challenge of the target dataset is that the text length of different cases in the dataset varies greatly, ranging from several words to hundreds of words. Is this network a good choice for the task? Give your answer and explanation. If not a good choice, how to improve the model? \[5 marks\]

  

  

3\. Read the following program code snippet to define a sentiment classification model in

PyTorch. Does this code snippet work for the sentiment classification task? If not, please

clarify the bugs. \[5 marks\]

import torch.nn as nn

from transformers import BertModel

class SentimentClassificationBERT(nn.Module):

def \_\_init\_\_（self,bert\_version,num\_class):

super().\_\_init\_\_()

self.bert=BertModel.from\_pretrained(bert\_version)

hidden\_size =self.bert.config.hidden\_size

self.linear\_decoder =nn.Linear(hidden\_size,num\_class)

def forward(self，input\_ids,attention\_mask):

sequence\_output,pooled\_output =self.bert(input\_ids=input\_ids,

attention\_mask=attention\_mask)

logits = self.linear\_decoder(sequence\_output)

return logits

---

显示思路

好的，我们来逐个翻译并详细讲解这些问题。

问题翻译与讲解
-------

### **1\. 任务导向对话系统 (CRB 系统 - Conference Room Booking / 会议室预订系统)**

假设我们要为一个会议室预订（CRB）系统设计一个面向任务的对话系统。

#### **1.1 问题翻译与讲解**

**英文原文:** 1.1 To design such a task-oriented dialogue system, we need to first perform natural language understanding tasks, including domain classification, intent detection, and slot filling. Explain in this CRB system, what is the domain classification problem, what is the intent detection problem, and what is the slot filling problem. You can design 5 reasonable slots to indicate this CRB example. \[9 marks\]

**中文翻译:** 1.1 为了设计这样一个面向任务的对话系统，我们首先需要执行自然语言理解（NLU）任务，包括领域分类（domain classification）、意图检测（intent detection）和槽位填充（slot filling）。请结合这个CRB（会议室预订）系统解释：什么是领域分类问题，什么是意图检测问题，以及什么是槽位填充问题。你可以设计5个合理的槽位来举例说明这个CRB系统。

**中文讲解:**

在一个面向任务的对话系统中，自然语言理解（NLU）模块负责将用户的自然语言输入转化为机器可理解的结构化表示。对于一个会议室预订（CRB）系统，这三个NLU核心任务具体如下：

1.  **领域分类 (Domain Classification):**
    
    *   **问题定义:** 判断用户的查询或指令属于哪个预定义的业务领域或主题。一个复杂的对话系统可能处理多个领域（如预订会议室、查询天气、播放音乐等）。领域分类的目标是将用户输入分配到正确的领域，以便后续由该领域的特定逻辑来处理。
    *   **在CRB系统中的例子:**
        *   如果用户说：“我想订一个会议室”，系统需要识别出这个请求属于“会议室预订”领域，而不是“餐厅预订”或“机票查询”领域。
        *   即使系统只处理会议室预订，也可能需要区分是“预订”、“取消预订”还是“查询可用性”等子领域（尽管这些更偏向意图）。但在更广义的系统中，区分“会议室相关操作”和其他潜在操作（如“系统帮助”）就是领域分类。
2.  **意图检测 (Intent Detection):**
    
    *   **问题定义:** 在确定了领域之后，需要进一步识别用户在该领域下的具体意图或目的。用户想要完成什么操作？
    *   **在CRB系统中的例子 (假设领域已确定为“会议室预订”):**
        *   用户说：“帮我订一个明天的会议室。” -> **意图:** `预订会议室 (book_conference_room)`
        *   用户说：“我想看看今天下午有哪些会议室可以用？” -> **意图:** `查询可用会议室 (find_available_rooms)`
        *   用户说：“取消我订的301会议室。” -> **意图:** `取消预订 (cancel_booking)`
        *   用户说：“我昨天订的会议室状态怎么样了？” -> **意图:** `查询预订状态 (check_booking_status)`
3.  **槽位填充 (Slot Filling):**
    
    *   **问题定义:** 一旦识别了用户的意图，系统需要从用户的话语中提取完成该意图所必需的关键信息片段，这些信息片段被称为“槽位”（slots）。槽位是预先定义好的参数，对应于意图的变量。
        
    *   **在CRB系统中的例子 (假设意图是 `预订会议室`):** 为了成功预订会议室，系统可能需要以下信息（槽位）：
        
        1.  `日期 (date)`: 用户希望预订会议室的日期。例如：“明天”、“5月12号”。
        2.  `时间 (time)`: 用户希望预订会议室的具体时间段。例如：“下午2点到4点”、“上午10点”。
        3.  `时长 (duration)`: 会议室使用的时长。例如：“2小时”、“1个小时”。
        4.  `参会人数 (number_of_participants)`: 大约有多少人参加会议，以便选择合适大小的会议室。例如：“10个人”、“大约5人”。
        5.  `会议室偏好 (room_preference)`: 用户是否对会议室有特定要求，如设备（投影仪）、楼层或特定房间号。例如：“需要有投影仪”、“301会议室”。
    *   **槽位填充过程举例:** 用户说：“我想预订明天下午3点供5人使用的会议室，需要有投影仪。” NLU模块会进行槽位填充，提取出：
        
        *   `日期`: "明天"
        *   `时间`: "下午3点"
        *   `参会人数`: "5人"
        *   `会议室偏好`: "有投影仪" (可能还需要进一步澄清 `时长`)

这些NLU任务的输出（领域、意图、槽位及其值）会传递给对话管理模块，以决定系统的下一步行动。

#### **1.2 问题翻译与讲解**

**英文原文:** 1.2 What are the next-step tasks/modules after the natural language understanding tasks to finish this closed-loop system? Explain how these next-step tasks/modules work using this CRB example. \[9 marks\]

**中文翻译:** 1.2 在自然语言理解任务之后，为了完成这个闭环系统，接下来的步骤/模块是什么？请结合这个CRB例子解释这些接下来的步骤/模块是如何工作的。

**中文讲解:**

在一个面向任务的对话系统中，自然语言理解（NLU）只是第一步。为了构成一个闭环系统（能够与用户持续交互直至完成任务），NLU之后通常包括以下核心模块/任务：

1.  **对话状态跟踪 (Dialogue State Tracking, DST):**
    
    *   **工作原理:** DST模块负责在多轮对话中维护和更新当前对话的状态。对话状态通常包括到目前为止用户表达的意图、已填充的槽位值以及系统的历史行为等。它整合NLU的输出和对话历史，形成对当前对话情况的全面理解。
    *   **在CRB系统中的例子:**
        *   **用户第一轮:** "我想订个会议室。"
            *   NLU输出: 意图=`book_conference_room`。
            *   DST更新状态: `intent=book_conference_room`, `date=null`, `time=null`, `participants=null`...
        *   **系统回应 (由对话策略决定):** "好的，请问您想预订哪一天？"
        *   **用户第二轮:** "明天。"
            *   NLU输出: 槽位=`date:明天`。
            *   DST更新状态: `intent=book_conference_room`, `date=明天`, `time=null`, `participants=null`... DST会持续跟踪哪些槽位已被填充，哪些还需要询问用户。
2.  **对话策略 (Dialogue Policy / Dialogue Management, DM):**
    
    *   **工作原理:** 对话策略模块根据DST提供的当前对话状态，决定系统下一步应该执行什么动作 (system action)。这个动作可以是向用户提问以获取更多信息、向用户确认信息、调用后端API执行任务，或者给出最终答复。对话策略可以是基于规则的，也可以是基于强化学习等统计模型训练得到的。
    *   **在CRB系统中的例子:**
        *   **当前状态 (来自DST):** `intent=book_conference_room`, `date=明天`, `time=下午2点`, `participants=null`。
        *   **对话策略决定动作:** 由于 `participants` 槽位为空，策略可能会决定生成一个询问人数的系统动作，如 `request(participants)`。
        *   **当前状态 (来自DST):** `intent=book_conference_room`, `date=明天`, `time=下午2点`, `participants=10人` (所有必要槽位已填充)。
        *   **对话策略决定动作:** 策略可能会决定调用后端API查询符合条件的会议室，如 `api_call(find_room, date='明天', time='下午2点', participants=10)`。如果找到，可能会生成一个确认动作 `confirm_booking(room_id='301', ...)`。
3.  **自然语言生成 (Natural Language Generation, NLG):**
    
    *   **工作原理:** NLG模块负责将对话策略决定的抽象系统动作转换成用户可以理解的自然语言文本。例如，将系统动作 `request(time)` 转换为句子 "请问您想预订几点呢？"。NLG的目标是生成清晰、自然、符合语境的回复。
    *   **在CRB系统中的例子:**
        *   **系统动作 (来自DM):** `request(duration)`
        *   **NLG输出:** "您需要使用多久呢？" 或 "会议时长是多久？"
        *   **系统动作 (来自DM):** `inform_ результат(booking_successful, room='301', date='明天', time='下午2点')`
        *   **NLG输出:** "好的，已成功为您预订明天下午2点的301会议室。"
4.  **后端/外部API交互 (Backend/External API Interaction):** (虽然不完全是对话系统的“模块”，但对任务完成至关重要)
    
    *   **工作原理:** 对于很多任务导向系统，需要与外部数据库或服务进行交互以获取信息或执行操作。例如，在CRB系统中，需要查询会议室数据库的可用性、进行预订、取消预订等。对话策略模块会决定何时调用这些API，并处理API返回的结果。
    *   **在CRB系统中的例子:**
        *   当所有预订所需信息收集完毕后，DM会触发调用一个 `book_room_api(date, time, duration, participants, room_preference)`。
        *   API会返回预订成功（及会议室详情）或失败（及原因，如无可用会议室）。
        *   DM再根据API返回结果决定下一步的系统动作。

**闭环流程总结:** 用户输入 -> NLU (领域、意图、槽位) -> DST (更新对话状态) -> DM (根据状态决定系统动作) -> \[可选：后端API交互\] -> NLG (将系统动作转为自然语言) -> 系统回复给用户。这个循环不断进行，直到任务完成或用户结束对话。

#### **1.3 问题翻译与讲解**

**英文原文:** 1.3 Can you provide at least two metrics to evaluate this CRB system performance? Explain their meanings. \[4 marks\]

**中文翻译:** 1.3 你能提供至少两个用于评估此CRB系统性能的指标吗？请解释它们的含义。

**中文讲解:**

评估一个面向任务的对话系统（如CRB系统）的性能可以从多个层面进行，以下是至少两个常用的评估指标：

1.  **任务完成率 (Task Completion Rate / Success Rate):**
    
    *   **含义:** 这是评估面向任务对话系统最核心的指标之一。它衡量的是系统在多大程度上能够成功帮助用户完成其预期的任务。
    *   **计算方法:** (成功完成任务的对话数) / (总对话数)。
    *   **在CRB系统中的例子:** 如果用户想要预订一个会议室，对话结束后，会议室确实被成功预订（或者系统明确告知无法预订并给出合理原因，这也可能被视为一种形式的“成功”处理），则认为该对话任务完成。如果用户中途放弃，或者系统未能理解用户意图，或者预订失败但未给出清晰解释，则任务未完成。
    *   **重要性:** 直接反映了系统的实用性和有效性。高任务完成率意味着系统能够很好地满足用户需求。
2.  **对话轮数 (Number of Turns / Dialogue Length):**
    
    *   **含义:** 指完成一个任务平均需要多少轮用户与系统之间的交互。一轮通常指用户的一次输入和系统的一次回复。
    *   **计算方法:** (所有对话的总轮数) / (总对话数)。通常我们希望在成功完成任务的前提下，对话轮数越少越好。
    *   **在CRB系统中的例子:**
        *   **高效系统:** 用户说“订明天下午2点10人会议室”，系统直接确认或预订成功，可能只需要2-3轮。
        *   **低效系统:** 系统可能需要多次澄清，例如：“哪一天？”、“几点？”、“多少人？”，导致对话轮数增加。
    *   **重要性:** 反映了对话的效率。轮数过长可能意味着系统理解能力差、策略不佳，导致用户体验下降。但需要注意的是，并非轮数越少就一定越好，有时为了确保准确性，适当的澄清是必要的。此指标常与任务完成率结合分析。

**其他可选指标:**

*   **用户满意度 (User Satisfaction):** 通过问卷调查或评分（如1-5星）来收集用户对交互体验的主观评价。
*   **NLU准确率 (NLU Accuracy):** 单独评估NLU模块的性能，包括意图识别准确率、槽位填充的F1值等。这对于诊断系统瓶颈很重要。
*   **响应时间 (Response Time):** 系统对用户输入做出响应所需的时间。过长的延迟会影响用户体验。
*   **语义错误率 (Semantic Error Rate):** 系统在理解或执行层面发生的错误（例如，错误地预订了时间或会议室）。

选择哪些指标取决于评估的具体目标和系统的特点。任务完成率和对话轮数是比较通用的核心指标。

### **2\. 词表示 (Word Representations)**

#### **2.1 问题翻译与讲解**

**英文原文:** 2.1 What is the distributional hypothesis in word representations? What is the key problem that the contextualized word representation (like ELMo) solves, compared with context-independent word representations (like word2vec or GloVe)? \[4 marks\]

**中文翻译:** 2.1 在词表示中，什么是分布假说（distributional hypothesis）？与上下文无关的词表示（如word2vec或GloVe）相比，上下文相关的词表示（如ELMo）解决了什么关键问题？

**中文讲解:**

1.  **分布假说 (Distributional Hypothesis):**
    
    *   **含义:** 分布假说的核心思想是“一个词的意义由其经常共同出现的词（即其上下文）所决定”(You shall know a word by the company it keeps - J.R. Firth)。换句话গাঁও，如果两个词经常出现在相似的上下文中，那么这两个词的语义就可能相似。
    *   **在词表示中的应用:** 这个假说是现代词嵌入方法（如word2vec, GloVe, FastText, ELMo, BERT等）的理论基础。这些方法通过分析大规模语料库中词语的共现统计信息，将词语映射到低维稠密的向量空间中。在这样的向量空间里，语义相近的词（因为它们有相似的上下文分布）其对应的向量在空间中的距离也更近。
2.  **上下文相关的词表示 (如ELMo) 解决了上下文无关词表示 (如word2vec, GloVe) 的关键问题:**
    
    *   **上下文无关词表示 (Context-Independent Word Representations) 的问题:**
        
        *   像word2vec和GloVe这类模型为词汇表中的每一个词生成一个 _固定_ 的词向量。这意味着无论一个词出现在什么句子中，它的词向量表示都是相同的。
        *   **关键问题：无法处理一词多义 (Polysemy)。** 自然语言中很多词有多种含义，具体含义取决于其所处的上下文。例如，“bank”可以指银行，也可以指河岸。上下文无关的词表示会将“bank”的所有含义混合到一个单一的向量中，无法区分其在不同语境下的具体意义。
        *   例如，在句子 "I went to the bank to deposit money" 和 "The boat was on the river bank" 中，"bank" 的词向量在word2vec中是完全一样的。
    *   **上下文相关的词表示 (Contextualized Word Representations - 如ELMo) 解决的关键问题:**
        
        *   ELMo (Embeddings from Language Models) 以及后续的Transformer模型（如BERT, GPT）能够为同一个词在不同的上下文中生成不同的词向量表示。
        *   **解决方案：动态的、上下文感知的词向量。** ELMo使用一个双向长短期记忆网络 (biLSTM) 预训练的语言模型。当要获取一个词在特定句子中的表示时，整个句子会输入到这个预训练好的biLSTM中，该词的表示是其在biLSTM中所有层的前向和后向隐藏状态的函数。因此，词的表示会根据其左右上下文动态调整。
        *   **解决了什么：**
            *   **一词多义问题:** 对于“bank”这个词，ELMo在 "I went to the bank to deposit money" 中生成的向量会更接近“金融机构”相关的语义，而在 "The boat was on the river bank" 中生成的向量会更接近“地理特征”相关的语义。
            *   **更丰富的语义表达:** 上下文相关的表示能够捕捉到词义在不同语境下的细微差别和用法，从而提供更丰富、更精确的语义信息，这对于下游的NLP任务（如情感分析、问答、机器翻译等）非常有益。

总结来说，上下文相关的词表示（如ELMo）解决的核心问题是传统上下文无关词表示无法有效处理一词多义以及无法根据具体语境动态调整词义的问题。

#### **2.2 问题翻译与讲解**

**英文原文:** 2.2 If we select the skip-gram model as our training strategy on a larger corpus, and then start the training process. However, we found that the training speed is very slow. Why does this problem occur and how to improve the efficiency of training? Give at least one of these possible solutions and explain it. \[6 marks\]

**中文翻译:** 2.2 如果我们选择skip-gram模型作为在大型语料库上的训练策略，并开始训练过程。然而，我们发现训练速度非常慢。为什么会出现这个问题？以及如何提高训练效率？请给出至少一种可能的解决方案并解释它。

**中文讲解:**

**为什么Skip-gram在大型语料库上训练速度慢?**

Skip-gram模型的目标是根据中心词来预测其上下文词。对于每个训练样本（一个中心词和其上下文中的一个词），模型需要进行以下操作：

1.  计算中心词与词汇表中 _所有_ 词作为上下文词的得分（通常通过点积）。
2.  通过Softmax函数将这些得分转换为概率分布。
3.  计算损失（例如交叉熵损失），并更新模型参数（中心词的词向量和上下文词的词向量）。

**导致训练缓慢的主要原因：Softmax计算成本过高**

*   当词汇表非常大时（大型语料库通常意味着大词汇表，可能有几十万甚至上百万个词），在每一步为词汇表中的所有词计算Softmax概率的计算量非常巨大。Softmax函数的分母需要对词汇表中所有词的得分进行求和，这个操作的计算复杂度与词汇表大小 V 成正比。对于每个训练样本都要执行这样的计算，使得训练过程非常耗时。

**如何提高Skip-gram的训练效率？(至少一种解决方案)**

以下是几种常用的提高Skip-gram训练效率的方法：

1.  **负采样 (Negative Sampling):**
    
    *   **解释:** 负采样是近似Softmax的一种有效方法。它将多分类问题（预测正确的上下文词，类别数等于词汇表大小）转化为一系列二分类问题。
    *   **工作原理:**
        1.  对于每个训练样本（一个中心词 wc​ 和一个实际的上下文词 wo​，称为正样本），我们不再尝试预测 wo​ 在整个词汇表中的概率。
        2.  取而代之，我们从词汇表中随机抽取 k 个不正确的上下文词（称为负样本 wneg1​​,...,wnegk​​）。这些负样本通常根据其在语料库中的出现频率（例如，使用unigram分布的3/4次方进行采样，高频词更容易被选作负样本）进行采样。
        3.  模型的目标就变成：对于正样本 (wc​,wo​)，最大化其同时出现的概率（或者说，模型能够正确区分这是一个正样本）；对于每个负样本 (wc​,wnegi​​)，最小化其同时出现的概率（或者说，模型能够正确区分这是一个负样本）。
        4.  这通常通过为每个 (wc​,wo​) 和 (wc​,wnegi​​) 对应用一个sigmoid函数来输出它们是“真实上下文对”的概率，然后使用二元交叉熵损失进行优化。
    *   **效率提升的原因:** 我们不再需要为整个词汇表 V 计算概率和更新参数，而是只需要为1个正样本和 k 个负样本（k 通常远小于 V，例如5-20）计算和更新。这大大减少了每个训练步骤的计算量。
2.  **分层Softmax (Hierarchical Softmax):**
    
    *   **解释:** 分层Softmax是另一种近似Softmax的方法，它使用一个二叉树（通常是霍夫曼树，高频词路径短）来表示词汇表。
    *   **工作原理:**
        1.  词汇表中的每个词都对应于树的一个叶子节点。从根节点到每个叶子节点都有一条唯一的路径。
        2.  预测一个上下文词的概率，等价于预测从根节点到对应叶子节点的这条路径上一系列向左或向右选择的概率。
        3.  在树的每个非叶子节点上，都有一个向量表示，并训练一个二分类器（如逻辑回归）来决定是向左孩子走还是向右孩子走。
        4.  一个词的概率是其路径上所有分支选择概率的乘积。
    *   **效率提升的原因:** 计算一个词的概率不再需要遍历整个词汇表，而是只需要沿着树的路径进行计算。对于平衡二叉树，路径长度大约是 log2​V。霍夫曼树可以进一步优化，使得高频词的路径更短。这比原始Softmax的 O(V) 复杂度要低得多。

选择负采样或分层Softmax都可以显著提高Skip-gram模型在大型语料库上的训练速度。负采样在实践中通常更容易实现，并且对于低频词的效果也较好。

#### **2.3 问题翻译与讲解**

**英文原文:** 2.3 After obtaining a well-trained language model, we incorporate it with a multi-layer recurrent neural network (RNN) to address the named entity recognition task. A major challenge of the target dataset is that the text length of different cases in the dataset varies greatly, ranging from several words to hundreds of words. Is this network a good choice for the task? Give your answer and explanation. If not a good choice, how to improve the model? \[5 marks\]

**中文翻译:** 2.3 在获得一个训练良好的语言模型后，我们将其与一个多层循环神经网络（RNN）结合起来解决命名实体识别（NER）任务。目标数据集的一个主要挑战是，数据集中不同样本的文本长度差异很大，从几个词到几百个词不等。这个网络（指多层RNN结合语言模型）是解决该任务的好选择吗？给出你的答案和解释。如果不是好选择，如何改进模型？

**中文讲解:**

**这个网络（预训练语言模型 + 多层RNN）是解决该NER任务的好选择吗？**

**答案:** 对于处理文本长度差异巨大的NER任务，传统的“预训练语言模型 + 多层RNN”架构**可能不是最佳选择，但可以作为基线，并且通过一些改进可以使用。**

**解释:**

*   **优点:**
    
    *   **预训练语言模型 (LM):** 提供了强大的上下文词表示，能够捕捉丰富的语义信息，这对于NER任务非常有益。
    *   **RNN (如LSTM, GRU):** RNN的循环结构使其能够处理序列数据并理论上可以捕捉长距离依赖关系。多层RNN可以学习更复杂的特征表示。
*   **潜在问题/挑战 (尤其在文本长度差异大时):**
    
    1.  **梯度消失/爆炸 (Vanishing/Exploding Gradients):** 虽然LSTM和GRU在一定程度上缓解了这个问题，但对于非常长的序列（几百个词），RNN仍然可能难以有效地将信息从序列的早期传递到后期，或者从后期反向传播梯度到早期。这会影响模型学习长距离依赖的能力，而NER有时需要理解较远上下文中的线索。
    2.  **计算效率:** RNN是顺序处理的，即必须按时间步依次计算。对于非常长的序列，这会导致训练和推理速度较慢。如果一个批次中既有很短的序列也有很长的序列，为了进行批处理，短序列需要被大量填充 (padding)，这会浪费计算资源。
    3.  **对长序列的建模能力瓶颈:** 尽管RNN理论上可以处理任意长度的序列，但实践中其捕获超长距离依赖的能力会下降。

**如果不是好选择，如何改进模型？**

如果传统的RNN在该任务上表现不佳，可以考虑以下改进方法：

1.  **使用双向RNN (Bidirectional RNN, e.g., BiLSTM, BiGRU):**
    
    *   **改进:** 对于NER任务，一个词的标签不仅取决于其左侧上下文，也取决于其右侧上下文。双向RNN能够同时从前向和后向两个方向处理序列，为每个时间步提供更全面的上下文信息。这通常能显著提升NER性能。
    *   **如何工作:** 将前向RNN在每个时间步的隐藏状态与后向RNN在同一时间步的隐藏状态拼接起来，作为该时间步的最终表示，然后输入到输出层（如CRF层或Softmax层）进行标签预测。
2.  **引入注意力机制 (Attention Mechanism) 与RNN结合:**
    
    *   **改进:** 注意力机制可以帮助RNN在处理长序列时，更关注与当前预测相关的部分，而不是平等对待所有历史信息。
    *   **如何工作:** 在RNN的解码阶段（或者在每个时间步的表示上），可以引入一个自注意力机制，或者如果结合了编码器-解码器结构，可以使用标准的注意力机制，让模型动态地选择性地聚焦于输入序列的不同部分。
3.  **使用Transformer模型替代或补充RNN:**
    
    *   **改进:** Transformer模型（如BERT, RoBERTa等作为编码器）完全基于自注意力机制，而不是循环结构。它们在捕捉长距离依赖方面通常优于RNN，并且可以并行处理序列中的所有词元（在编码阶段），计算效率更高（对于长序列，自注意力的复杂度是 O(n2d)，而RNN是 O(nd2)，但在实际中Transformer因其并行性可能更快）。
    *   **如何工作:**
        *   **直接使用预训练Transformer作为编码器:** 将文本输入到预训练的Transformer模型（如BERT），获取每个词元的上下文表示。然后，在这些表示之上直接接一个线性层（和Softmax或CRF）进行NER标签预测。这是目前NER任务中最主流和效果最好的方法之一。
        *   **Transformer + CRF:** 在Transformer的输出之上增加一个条件随机场 (CRF) 层。CRF能够学习标签之间的依赖关系（例如，I-PERSON不能跟在B-LOCATION之后），有助于产生更合法的标签序列。
4.  **处理变长序列的策略:**
    
    *   **分块处理 (Chunking/Segmenting):** 对于非常长的文档，可以将其分割成固定长度或有重叠的块，对每个块分别进行NER，然后合并结果。
    *   **动态批处理 (Dynamic Batching) / 按长度排序输入:** 将长度相似的句子组织在同一个批次中，以减少padding的开销，提高训练效率。

**结论:** 虽然“预训练LM + 多层RNN”可以用于NER，但面对长度差异巨大的文本，其性能和效率可能受限。**目前更优的选择是直接使用基于Transformer的预训练模型（如BERT）作为编码器，并在其输出之上接一个分类层（或CRF层）来进行NER**。这种架构能更好地处理长距离依赖，并且在许多NER基准测试中都取得了SOTA（State-of-the-Art）的结果。

### **3\. PyTorch情感分类模型代码分析**

**英文原文:** 3. Read the following program code snippet to define a sentiment classification model in PyTorch. Does this code snippet work for the sentiment classification task? If not, please clarify the bugs. \[5 marks\]

Python

    import torch.nn as nn
    from transformers import BertModel
    
    class SentimentClassificationBERT(nn.Module):
        def __init__(self, bert_version, num_class):
            super().__init__()
            self.bert = BertModel.from_pretrained(bert_version)
            hidden_size = self.bert.config.hidden_size
            self.linear_decoder = nn.Linear(hidden_size, num_class)
    
        def forward(self, input_ids, attention_mask):
            sequence_output, pooled_output = self.bert(input_ids=input_ids,
                                                     attention_mask=attention_mask)
            logits = self.linear_decoder(sequence_output)
            return logits

**中文翻译:** 3. 阅读以下定义PyTorch情感分类模型的程序代码片段。这段代码能用于情感分类任务吗？如果不能，请指出其中的错误。

**中文讲解:**

**这段代码能用于情感分类任务吗？如果不能，请指出其中的错误。**

**答案:** 这段代码**基本上可以工作**，但存在一个**主要问题**，导致其可能不是情感分类任务的最佳或最常见实现方式，并且可能会产生维度不匹配的错误，具体取决于损失函数的期望输入。

**主要问题/潜在错误:**

1.  **分类器输入选择错误 (Incorrect input to `self.linear_decoder`):**
    
    *   **代码现状:** `logits = self.linear_decoder(sequence_output)`
        
    *   **问题描述:**
        
        *   `BertModel` 的输出包含两个部分：
            *   `sequence_output`: 形状为 `(batch_size, sequence_length, hidden_size)`。这是序列中每个词元（token）的最后一层隐藏状态表示。
            *   `pooled_output`: 形状为 `(batch_size, hidden_size)`。这通常是序列的第一个词元 `[CLS]` 的表示，并经过一个线性层和Tanh激活函数进一步处理，设计用于句子级别的分类任务。
        *   情感分类通常是一个句子（或文档）级别的分类任务，我们需要一个能够代表整个输入序列的固定大小的向量来进行分类。
        *   代码中将 `sequence_output`（每个词元的表示）直接输入到 `self.linear_decoder`。如果 `self.linear_decoder` 是期望一个代表整个句子的向量（通常是这样），那么这里就会有问题。`nn.Linear(hidden_size, num_class)` 期望的输入是 `(batch_size, hidden_size)` 或 `(N, *, hidden_size)` 然后它作用于最后一个维度。
        *   如果直接用 `sequence_output`，那么 `self.linear_decoder` 会对序列中的 _每个词元_ 都输出一个分类结果，得到形状为 `(batch_size, sequence_length, num_class)` 的 `logits`。这更像是序列标注任务（如NER）的输出，而不是句子情感分类。
    *   **如何修正 (常见做法):**
        
        *   **使用 `pooled_output`:** 对于句子级别的分类任务，通常使用 `pooled_output` 作为分类器的输入。Python
            
                logits = self.linear_decoder(pooled_output)
            这样 `logits` 的形状将是 `(batch_size, num_class)`，这正是情感分类任务期望的输出形状（为批次中的每个句子输出一个类别概率分布）。
        *   **取 `[CLS]` 词元的 `sequence_output`:** 另一种常见的做法是直接取 `sequence_output` 中对应于 `[CLS]` 词元（通常是序列的第一个词元）的表示作为整个序列的表示：Python
            
                cls_representation = sequence_output[:, 0, :]  # (batch_size, hidden_size)
                logits = self.linear_decoder(cls_representation)
            `pooled_output` 本质上就是基于 `[CLS]` 的表示经过额外处理得到的，所以两者效果相似，但直接用 `pooled_output` 更直接。
2.  **`BertModel` 的返回格式 (需要注意):**
    
    *   从 `transformers` 库的版本 v4.x.x 开始，`BertModel` 的 `forward` 方法默认返回一个 `BaseModelOutputWithPoolingAndCrossAttentions` 对象（或类似的ModelOutput对象），而不是直接的元组 `(sequence_output, pooled_output)`。
    *   为了向后兼容，仍然可以通过解包获取，如 `sequence_output, pooled_output = self.bert(...)`。但更推荐的写法是访问返回对象的属性：Python
        
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state # 或者 outputs[0]
            pooled_output = outputs.pooler_output       # 或者 outputs[1]
        
    *   当前代码的解包方式在很多情况下仍然有效，但这取决于 `transformers` 库的具体版本和 `BertModel` 的配置。如果 `return_tuple=True`（在某些旧版本中可能是默认或可配置的），则返回元组。如果 `return_dict=True` (新版本默认)，则返回ModelOutput对象。为了代码的健壮性和可读性，建议使用属性访问。

**总结:**

该代码片段的核心逻辑（加载预训练BERT模型，并在其上添加一个线性分类头）是正确的。**最主要的问题在于 `forward` 方法中选择了 `sequence_output` 而不是 `pooled_output` (或 `sequence_output`的 `[CLS]` 部分) 作为线性解码器的输入，这使得它更像是在为每个词元进行分类，而不是为整个序列进行情感分类。**

如果损失函数期望 `(batch_size, num_class)` 的输入（例如标准的 `nn.CrossEntropyLoss` 用于句子分类），那么当前代码的 `logits` 形状 `(batch_size, sequence_length, num_class)` 会导致维度不匹配错误。

**建议的修改后代码 (使用 `pooled_output`):**

Python

    import torch.nn as nn
    from transformers import BertModel
    
    class SentimentClassificationBERT(nn.Module):
        def __init__(self, bert_version, num_class):
            super().__init__()
            self.bert = BertModel.from_pretrained(bert_version)
            hidden_size = self.bert.config.hidden_size
            self.linear_decoder = nn.Linear(hidden_size, num_class)
    
        def forward(self, input_ids, attention_mask):
            # 推荐的属性访问方式
            outputs = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask)
            pooled_output = outputs.pooler_output # (batch_size, hidden_size)
            
            # 或者如果坚持元组解包 (假设版本兼容)
            # _, pooled_output = self.bert(input_ids=input_ids, # _ 用来接收 sequence_output
            #                              attention_mask=attention_mask,
            #                              return_dict=False) # 明确返回元组，如果需要的话
    
            logits = self.linear_decoder(pooled_output) # (batch_size, num_class)
            return logits

信息来源和相关内容

---

