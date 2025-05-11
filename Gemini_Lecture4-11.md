![PDF 图标](https://drive-thirdparty.googleusercontent.com/32/type/application/pdf)

L4\_Transfo...finetuning

PDF

请你读取该ppt，然后用中文讲解给我，你不要逐页进行讲解，而应当按照每个ppt的大纲以及所有涉及到的知识点按点来进行讲解

---

显示思路

好的，这份PPT主要介绍了Transformer模型以及相关的预训练和微调技术，特别是BERT和GPT模型。内容将按照以下大纲和知识点进行讲解：

### 核心大纲

1.  **注意力机制 (Attention)**
2.  **Transformer模型**
3.  **BERT (Bidirectional Encoder Representations from Transformers)**
4.  **GPT (Generative Pre-Training)**

* * *

### 1\. 注意力机制 (Attention)

**1.1. 上下文词嵌入存在的问题**  

*   传统的循环神经网络 (RNN) 和长短期记忆网络 (LSTM) 在处理序列信息时，存在长程依赖问题，即难以捕捉距离较远的词之间的关联。  
*   信息在序列传递过程中容易丢失。  
*   难以并行计算，效率较低。  
*   双向RNN/LSTM的特征融合和表示能力相对较弱（与Transformer相比）。  

**1.2. 为什么需要注意力机制？**  

*   **降低计算复杂度**：注意力机制允许模型直接关注输入序列中的相关部分，而不是平等对待所有部分。  
*   **并行化计算**：自注意力机制 (Self-attention) 使得序列中所有位置可以同时计算，提高了效率。  
*   **解决长程依赖问题**：自注意力机制通过恒定数量的操作连接序列中的所有位置，有效捕捉长距离依赖关系。  
*   **提高模型可解释性**：通过观察注意力权重，可以理解模型在做决策时更关注哪些输入部分。  

**1.3. 什么是注意力机制？**  

*   **类比人类大脑**：类似于人类在处理信息时，会更关注重要的部分。  
*   **查询-键-值 (Query-Key-Value) 模型**：将一个查询 (Query) 和一组键值对 (Key-Value pairs) 映射到一个输出。查询、键、值和输出都是向量。  
*   **加权求和**：输出是值的加权和，每个值对应的权重由查询与对应键的相似度函数计算得出。  

**1.4. 自然语言处理中的注意力：翻译时的对齐**  

*   **机器翻译**：将源语言句子翻译成目标语言句子。  
*   **对齐**：指源语言单词和目标语言单词之间的对应关系。  
*   在基于RNN的编码器-解码器模型中，注意力机制允许解码器在生成每个目标词时，关注源句子中与之最相关的部分，而不是仅仅依赖编码器生成的固定长度向量。 这样可以避免将所有输入信息压缩到一个向量中导致信息丢失，并允许模型自适应地选择关注点。  

**1.5. 注意力机制的正式描述**  

1.  **计算相似度/注意力分数**：给定一个查询向量 q 和一组键值对 {(ki​,vi​)}i\=1L​，计算查询 q 与每个键 ki​ 之间的相似度分数 si​\=similarity(q,ki​)。  
2.  **归一化注意力分数**：将相似度分数归一化，使其介于0和1之间，并且总和为1。这些归一化后的分数称为注意力分布 ai​。常用方法是Softmax操作：ai​\=softmax(si​)\=∑j\=1L​exp(sj​)exp(si​)​。  
3.  **计算注意力/上下文向量**：将注意力分布作为权重，对相应的值进行加权求和，得到注意力向量 z\=∑i\=1L​ai​vi​。  
    *   注意：键 (Keys) 和值 (Values) 不一定是相同的，在机器翻译等场景中它们可能是不同的。  

**1.6. 注意力分数的计算方法**  

*   任何相似度度量方法都可以使用，如余弦相似度和皮尔逊相关系数。  
*   神经网络中常用的方法有：
    *   **加性注意力 (Additive Attention)**：si​\=w3​tanh(w1T​q+w2T​ki​)。  
    *   **乘性注意力 (Multiplicative Attention)**：si​\=qTWki​，其中 W 是一个权重矩阵。  
    *   **点积注意力 (Dot-Product Attention)**：si​\=qTki​。  
    *   **缩放点积注意力 (Scaled Dot-Product Attention)**：si​\=dk​![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702
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
        M834 80h400000v40h-400000z"></path></svg>)​qTki​​，其中 dk​ 是键向量的维度。除以 dk​![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702
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
        M834 80h400000v40h-400000z"></path></svg>)​ 是为了防止点积结果过大导致梯度消失。  

**1.7. 缩放点积注意力的矩阵形式**  

*   当有一组查询 (Q)、键 (K) 和值 (V) 时，可以高效地以矩阵形式计算： Attention(Q,K,V)\=softmax(dk​![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702
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

**1.8. 自注意力机制 (Self-Attention)**  

*   **核心思想**：对于输入序列中的每个词，计算它与序列中其他所有词（包括自身）之间的注意力。  
*   **特点**：
    *   任意两个位置之间的路径长度为常数。  
    *   易于并行计算。  

**1.9. 自注意力中的查询、键和值**  

*   通过对原始输入词嵌入 xi​ 进行线性变换得到：
    *   查询 (Query): qi​\=xi​WQ  
    *   键 (Key): ki​\=xi​WK  
    *   值 (Value): vi​\=xi​WV 其中 WQ,WK,WV 是可学习的权重矩阵。  

**1.10. 多头自注意力机制 (Multi-Head Self-Attention)**  

*   **动机**：通过使用多组不同的 WQ,WK,WV 权重矩阵（称为不同的“头”），模型可以从不同的表示子空间学习信息，并关注不同位置的信息。  
*   **计算过程**：
    1.  将输入分别通过 h 个不同的线性变换得到 h 组查询、键和值。  
    2.  对每一组并行地执行注意力函数（如缩放点积注意力），得到 h 个输出向量。  
    3.  将这 h 个输出向量拼接起来。  
    4.  再次通过一个线性变换得到最终的输出。  
*   **数学表示**： MultiHead(Q,K,V)\=Concat(head1​,...,headh​)WO 其中 headi​\=Attention(QWiQ​,KWiK​,VWiV​) WiQ​,WiK​,WiV​ 和 WO 都是可学习的权重矩阵。  
*   **效果**：多头注意力允许模型同时关注来自不同位置的不同信息，例如一个头可能关注句法关系，另一个头可能关注语义关系。  

* * *

### 2\. Transformer模型

**2.1. 概述**  

*   由Google在论文 "Attention is All You Need" 中提出。  
*   核心技术是多头自注意力机制。  
*   旨在解决序列到序列 (sequence-to-sequence) 任务，并能轻松处理长程依赖。  

**2.2. 整体架构**  

*   主要由编码器 (Encoder) 和解码器 (Decoder) 组成。  
*   **编码器**：将输入符号序列 (x1​,...,xn​) 映射为连续表示序列 z\=(z1​,...,zn​)。  
*   **解码器**：给定 z，一次生成一个输出符号，直到生成完整的输出序列 (y1​,...,ym​)。  
*   **自回归 (Auto-regressive)**：解码器在生成当前词时，会利用先前已生成的词作为额外输入。  
*   编码器和解码器都由多层堆叠的自注意力和逐点全连接前馈网络 (point-wise, fully connected layers) 组成。  

**2.3. Transformer中的注意力**

*   **编码器中的注意力 (Encoder Attention)**  
    *   在自注意力层中，所有的键、值和查询都来自前一层的输出或原始输入嵌入。  
    *   编码器中的每个位置都可以关注编码器前一层的所有位置。  
*   **解码器中的注意力 (Decoder Attention)**  
    *   **屏蔽自注意力 (Masked Self-Attention)**：类似于编码器中的自注意力，但是为了保持自回归特性，会屏蔽掉当前位置之后的信息（即不允许关注未来的词）。 这是通过将未来位置的注意力分数设置为负无穷来实现的。  
    *   **编码器-解码器注意力 (Encoder-Decoder Attention)**：查询来自解码器中屏蔽自注意力层的输出，而键和值来自编码器最顶层的输出。 这使得解码器的每个位置都能关注输入序列的所有位置。  

**2.4. 其他重要组件**  

*   **前馈网络 (Feed-Forward Network, FFN)**：编码器和解码器的每一层都包含一个全连接前馈网络，它分别且相同地应用于每个位置。其计算公式为：FFN(x)\=max(0,xW1​+b1​)W2​+b2​。  
*   **位置编码 (Positional Encoding)**：由于Transformer不包含循环或卷积结构，它本身无法感知序列中词的顺序。因此，需要引入位置编码来为模型提供词的位置信息。  
    *   位置编码向量会与词嵌入向量相加。  
    *   常用的位置编码使用不同频率的正弦和余弦函数： PE(pos,2i)​\=sin(pos/100002i/dmodel​) PE(pos,2i+1)​\=cos(pos/100002i/dmodel​) 其中 `pos` 是词在序列中的位置，`i` 是编码向量的维度索引。 这种方式使得模型可以学习到相对位置信息。  
*   **残差连接 (Residual Connection)**：在每个子层（如自注意力层、前馈网络）的输出都应用了残差连接，然后再进行层归一化 (Layer Normalization)。这有助于缓解梯度消失问题，使得可以训练更深的网络。  
*   **词嵌入与权重共享**：输入词和输出词被转换为维度为 dmodel​ 的向量。两个嵌入层和解码器输出前的线性层共享权重矩阵。  

* * *

### 3\. BERT (Bidirectional Encoder Representations from Transformers)

**3.1. 概述**  

*   由Google提出，全称是“来自Transformer的双向编码器表示”。  
*   是一种从无标签文本中预训练的深度双向表示模型。  
*   其核心特点是在所有层中同时考虑左右上下文信息。  
*   预训练好的BERT模型可以通过增加一个额外的输出层进行微调，以适应各种下游任务。  
*   在多种自然语言处理任务上取得了显著的效果。  

**3.2. BERT架构**  

*   由多层双向Transformer编码器堆叠而成。  
*   论文中提供了两种尺寸的模型：
    *   **BERT\_BASE**：12层，隐藏层大小768，自注意力头12个，总参数1.1亿。  
    *   **BERT\_LARGE**：24层，隐藏层大小1024，自注意力头16个，总参数3.4亿。  

**3.3. 预训练 (Pre-training) + 微调 (Fine-tuning) 范式**  

*   除了输出层，预训练和微调阶段使用相同的架构。  
*   使用相同的预训练模型参数来初始化不同下游任务的模型。  
*   在微调期间，所有的参数都会进行调整。  

**3.4. BERT成功的关键因素**  

*   **创新的预训练任务**：使用两个自监督任务进行预训练，而不是传统的单向语言模型。  
    *   **掩码语言模型 (Masked Language Model, MLM)**  
    *   **下一句预测 (Next Sentence Prediction, NSP)**  
    *   训练损失是MLM似然和NSP似然的均值之和。  
*   **对各种任务的兼容性**：只需对BERT模型进行特定任务的微调即可达到最先进的性能。  

**3.5. BERT的输入/输出表示**  

*   对于给定的词元 (token)，其输入表示由相应的词元嵌入、片段嵌入 (segment embedding) 和位置嵌入 (position embedding) 相加构成。  
*   **\[CLS\]**：在每个输入样本前添加的特殊符号，其对应的最终隐藏层输出可用于分类任务。  
*   **\[SEP\]**：特殊的分隔符词元，例如用于分隔问题和答案，或两个句子。  
*   这种表示方式可以明确地表示单个句子或一对句子。  

**3.6. BERT的预训练数据**  

*   BERT的成功部分归功于其海量的训练数据。  
*   主要使用了两个语料库：
    *   BooksCorpus (8亿词)  
    *   英文维基百科 (25亿词)  

**3.7. 掩码语言模型 (Masked Language Model, MLM)**  

*   **动机**：实现双向预训练。传统的语言模型要么是单向的（从左到右或从右到左），要么是浅层拼接双向信息。
*   **核心思想**：随机屏蔽输入词元中的一部分，然后预测那些被屏蔽的词元。  
*   **细节**：  
    *   随机选择15%的词元位置进行预测。  
    *   如果第i个词元被选中，则：
        *   80%的概率用特殊词元 **\[MASK\]** 替换。  
        *   10%的概率用一个随机词元替换。  
        *   10%的概率保持原词元不变。  
    *   目标是基于被屏蔽词元第i个位置的最终隐藏向量，通过交叉熵损失来预测原始词元。  
    *   这种屏蔽策略是为了减少预训练和微调之间的不匹配，因为 **\[MASK\]** 词元在微调阶段不会出现。  

**3.8. 下一句预测 (Next Sentence Prediction, NSP)** (幻灯片第43页)  

*   **动机**：许多重要的下游任务（如问答、自然语言推断）需要理解句子之间的关系，而这并非语言模型本身能直接捕捉到的。
*   **核心思想**：预训练一个二分类模型，判断句子B是否是句子A的实际下一句。  
*   **构建训练样本**：
    *   50%的情况下，句子B确实是句子A的下一句（标记为IsNext）。
    *   50%的情况下，句子B是从语料库中随机选择的一个句子（标记为NotNext）。
*   **预测**：将\[CLS\]词元对应的最终隐藏层输出送入一个简单的分类器进行预测。

* * *

### 4\. GPT (Generative Pre-Training)

**4.1. GPT框架**  

*   **架构**：多层Transformer解码器 (decoder)。  
*   **自监督预训练**：类似于词嵌入 (如word2vec) 的自监督学习。  
    *   给定一系列词元U，最大化上下文条件概率：L1​(U)\=∑i​logP(ui​∣ui−k​,...,ui−1​;Θ)，其中k是窗口大小。  
    *   这是一个**单向 (unidirectional)** 的语言模型，即只根据左侧的上下文来预测当前词。  
*   **计算**：
    *   第一层输入：h0​\=UWe​+Wp​，其中 U 是词元嵌入，We​ 是词嵌入矩阵，Wp​ 是位置嵌入矩阵。  
    *   第l层输出：hl​\=transformer\_block(hl−1​)。  
    *   最终输出概率：P(u)\=softmax(hn​WeT​)。  

**4.2. 监督微调 (Supervised Fine-tuning)**  

*   保留预训练好的Transformer模型。  
*   替换掉预训练时用于预测下一个词的最终线性层 We​，换成针对特定任务的线性层 Wy​。  
*   给定有标签的数据 (X,y)，最大化 P(y∣X)。  
*   通常会在微调时加入一个辅助的训练目标（如语言模型损失），以提高泛化能力并加速收敛。  

**4.3. 任务相关的输入转换**  

*   GPT设计了一种通用的架构，通过对不同任务的输入进行转换，使其能够适应多种下游任务：
    *   **分类 (Classification)**：例如情感分析。将文本序列输入，通过\[CLS\]（或GPT中的特殊分隔符后的最后一个词元）的表示进行分类。  
    *   **蕴含 (Entailment)**：给定前提p和假设h，判断p是否蕴含h。将前提和假设用特殊分隔符连接后输入。  
    *   **相似度 (Similarity)**：判断两个句子是否语义等价。将两个句子用特殊分隔符连接后输入，有时会交换顺序再输入一次。  
    *   **多项选择 (Multiple Choice)**：例如故事完形填空或问答。将上下文和每个选项分别连接后输入，选择得分最高的选项。  

* * *

### 5\. 预训练模型对比：ELMo, GPT, BERT

 

| 特性 | ELMo (Embeddings from Language Models) | GPT (Generative Pre-Training) | BERT (Bidirectional Encoder Representations from Transformers) |
| :-- | :-- | :-- | :-- |
| **架构** | 双向LSTM语言模型 | 单向Transformer解码器 | 双向Transformer编码器 |
| **预训练** | 无监督语料库。学习词和支持下游任务的语言上下文特征。 | 无监督语料库。 | 无监督语料库。两个无监督任务：MLM和NSP。 |
| **下游任务** | 基于特征的任务和任务特定的模型。 | 基于模型的任务和任务无关的模型。每个特定任务需要判别式微调。 | 基于模型的任务和任务无关的模型。每个特定任务需要判别式微调。 |

 

这份PPT详细介绍了从注意力机制的基本原理到复杂的Transformer、BERT和GPT模型的演进和核心技术。希望这些讲解对您有所帮助！

信息来源和相关内容

---

![PDF 图标](https://drive-thirdparty.googleusercontent.com/32/type/application/pdf)

L5\_NLU tasks

PDF

非常棒，请你接着讲解该ppt

---

显示思路

Okay, let's dive into the content of the second PPT, "L5\_NLU tasks.pdf". This presentation focuses on Natural Language Understanding (NLU) tasks, with a special emphasis on Text Classification and Question Answering.

### 主题大纲

1.  **自然语言理解 (NLU) vs. 自然语言生成 (NLG)**
2.  **GLUE基准测试**
3.  **自然语言处理任务的难度**
4.  **NLU任务1：文本分类 (Text Classification)**
5.  **NLU任务2：问答 (Question Answering - QA)**

* * *

### 1\. 自然语言理解 (NLU) vs. 自然语言生成 (NLG)

*   **NLU (Natural Language Understanding)**：专注于理解文本的含义，并将其转化为结构化数据 。这类任务包括文本分类和问答（阅读理解） 。  
*   **NLG (Natural Language Generation)**：专注于从结构化数据生成人类可读的文本，以有效传递信息或进行交流 。NLG的任务实例有机器翻译和对话生成 。  
*   PPT中提供了一个表格，更详细地列举了不同类型的NLP任务，包括词语标记（如分词、命名实体识别）、句子解析（如依存句法分析）、文本分类（如情感分析）、文本对匹配（如自然语言推断）以及文本生成（如语言建模、机器翻译） 。  

* * *

### 2\. GLUE基准测试 (General Language Understanding Evaluation)

*   GLUE是由纽约大学、华盛顿大学等机构的学者提出的一个包含九项NLU任务的基准测试集合 。  
*   这些任务旨在评估模型在多种不同NLU能力上的表现，具体包括 ：  
    *   **CoLA (Corpus of Linguistic Acceptability)**：判断句子是否合乎语法。
    *   **SST-2 (Stanford Sentiment Treebank)**：判断电影评论的情感（正面、负面或中性）。
    *   **MRPC (Microsoft Research Paraphrase Corpus)**：判断句子B是否是句子A的释义。
    *   **STS-B (Semantic Textual Similarity Benchmark)**：判断句子A和B的相似程度。
    *   **QQP (Quora Question Pairs)**：判断两个问题是否相似。
    *   **MNLI-mm (MultiNLI Matched/Mismatched)**：判断句子A是否蕴含或矛盾于句子B。
    *   **QNLI (Question NLI)**：判断句子B是否包含句子A中问题的答案。
    *   **RTE (Recognizing Textual Entailment)**：判断句子A是否蕴含句子B。
    *   **WNLI (Winograd NLI)**：判断句子B中替代句子A中模糊代词的名词是否正确。 每个任务都有其特定的评估指标，如准确率（Accuracy）、F1分数、马修斯相关系数（Matthews correlation）、皮尔逊/斯皮尔曼相关系数（Pearson/Spearman correlation） 。  

* * *

### 3\. 自然语言处理任务的难度

PPT中将常见的NLP任务按照难度大致分为三类 ：  

*   **简单 (Easy)**：词性标注 (Part-of-Speech Tagging)、命名实体识别 (Named Entity Recognition)、垃圾邮件检测 (Spam Detection)。
*   **中等 (Medium)**：同义词库 (Thesaurus)、句法分析 (Syntactic Parsing)、词义消歧 (Word Sense Disambiguation)、情感分析 (Sentiment Analysis)、主题建模 (Topic Modeling)、信息检索 (Information Retrieval)。
*   **困难 (Hard)**：机器翻译 (Machine Translation)、文本生成 (Text Generation)、自动摘要 (Automatic Summarization)、问答 (Question Answering)、对话界面 (Conversational Interfaces)。

* * *

### 4\. NLU任务1：文本分类 (Text Classification)

**4.1. 定义与来源**

*   文本分类（也称文本定性）是NLP中的一个经典问题，旨在为文本单元（如句子、查询、段落、文档）分配标签或类别 。其数学形式化为生成一个假设（或分类器、模型）h:D→C，其中 D 是文本数据项的域，C 是一个有限的类别集合 。  
*   文本数据来源广泛，包括网页数据、电子邮件、聊天记录、社交媒体、工单、保险索赔、用户评论以及客服问答等 。  
*   常见的文本分类任务有：垃圾邮件检测、情感分析、新闻分类、用户意图分类和内容审核 。  

**4.2. 文本分类的类型**  

*   **二元分类 (Binary classification)**：每个项目恰好属于两个类别中的一个（例如，将邮件分为“垃圾邮件”或“合法邮件”） 。  
*   **单标签多类分类 (Single-Label Multi-Class, SLMC)**：每个项目恰好属于多个类别中的一个（例如，将新闻文章分配到“国内新闻”、“国际新闻”、“娱乐”、“生活方式”、“体育”之一） 。  
*   **多标签多类分类 (Multi-Label Multi-Class, MLMC)**：每个项目可能属于零个、一个或多个类别（例如，将计算机科学文章分配到ACM分类系统中的类别）。这可以被看作是n个独立的二元分类问题 。  
*   **序数分类 (Ordinal Classification, OC)**：与SLMC类似，但类别之间存在顺序关系（例如，将产品评论分配到“灾难性的”、“差的”、“一般的”、“好的”、“优秀的”之一） 。  

**4.3. 硬分类 vs. 软分类**  

*   **硬分类 (Hard classification)**：确定一个项目属于哪个类别。结果是明确的类别 。  
*   **软分类 (Soft classification)**：预测每个项目-类别对的(d,c)得分，该得分表示d属于c的概率/证据强度/置信度 。例如，概率分类器输出后验概率 。当得分不是概率时，可以通过S型函数（如逻辑函数、softmax操作）将其转换为概率 。  

**4.4. 具体的文本分类任务示例**

*   **垃圾邮件检测 (Spam Detection)**：通过检测未经请求和不需要的电子邮件，防止垃圾邮件进入用户收件箱，从而改善用户体验 。可以是二元分类（垃圾邮件/非垃圾邮件）或多类别分类（如主要、社交、推广等不同标签） 。  
*   **情感分析 (Sentiment Analysis)**：分析文本数据中人们的观点（如产品评论、电影评论或推文，对股市的情绪），并提取其极性和观点 。可以是二元（正面/负面）或多类别（细粒度标签或多级强度，如评论星级，或“开心”、“悲伤”、“惊讶”、“愤怒”等情绪） 。  
*   **新闻分类 (News Categorization)**：新闻内容是最重要的信息来源之一。新闻分类系统通过识别新兴新闻主题或根据用户兴趣推荐相关新闻，帮助用户实时获取感兴趣的信息 。例如，将新闻文章分配到“国内新闻”、“国际新闻”、“娱乐”、“生活方式”、“体育”等类别，或判断其是否“热门” 。  
*   **用户意图分类 (User Intent Classification)**：分析文本并将其分类为不同的意图，这有助于理解客户查询背后的意图，自动化流程并获得有价值的洞察 。例如，意图可以是“购买”、“降级”、“取消订阅”、“请求演示”等 。  
*   **内容审核 (Content Moderation)**：在允许用户发表评论的网站上，版主使用审核系统来筛选与有用或信息性贡献无关的、淫秽的、非法的、有害的或侮辱性的内容 。  
*   **主题分析 (Topic Analysis)**：旨在识别文本的主题（例如，判断一篇产品评论是关于“客户支持”还是“易用性”） 。  

**4.5. 文本分类的方法**

*   **通用流程**：可以分为“特征提取 + 分类器”的传统方法，或直接使用端到端模型 。  
*   **预处理 (Preprocessing)**：包括移除（或保留）标点符号、移除URL、移除停用词、转为小写、分词、词干提取 (stemming)、词形还原 (lemmatization) 等步骤 。  
*   **特征提取 (Feature Extraction)**：
    *   通常使用密集向量（嵌入），如通过子词嵌入、word2vec、GloVe或预训练的上下文语言模型（如ELMo, BERT）获得 。  
    *   在按主题分类时，典型的选择是使用训练集中出现的词集作为特征（unigram模型，也称“词袋模型”） 。  
    *   在按作者/标题分类时，会使用平均词长、平均句长、标点符号频率、虚拟语气从句频率等特征 。  
    *   其他特征还包括句法信息、语用信息等 。  
*   **特征选择 (Feature Selection)**：目标是识别最具区分性的特征，以便丢弃其他特征 。  
    *   “过滤式”方法通过度量每个特征的区分能力（如使用互信息）并保留得分最高的特征 。  
    *   矩阵分解技术（如PCA, SVD, LSA）可用于合成新的特征，以替代那些存在歧义和多义性的原始特征 。  
*   **分类器选择 (Classifier Selection)**：支持向量机 (SVMs)、提升决策树桩 (Boosted decision stumps)、逻辑回归 (Logistic regression)、朴素贝叶斯方法 (Naive Bayesian methods)、懒惰学习方法 (如k-NN) 以及基于神经网络的方法（通常支持端到端分类） 。  
*   **端到端神经文本分类器 (End-to-end Neural Text Classifiers)**：例如TextRNN ，以及像BERT、GPT这样的预训练模型 。  

**4.6. 标注 (Labeling)**

*   文本分类通常需要大量标注数据，而标注成本很高 。  
*   **主动学习 (Active Learning)**：由系统建议需要标注的样本，算法可以交互式地查询用户以获取新数据点的标签 。  
*   **自监督学习 (Self-supervised Learning)**：通过构建正/负样本来进行学习，可以分为生成式/预测式方法和对比式方法 。  

**4.7. 评估 (Evaluation)**

*   常用的评估指标包括精确率 (Precision)、召回率 (Recall) 和 F1分数 (F1-score) 。  

**4.8. 文本分类的网络选择**

*   选择一个预训练语言模型 (PLM)，如BERT或GPT 。  
*   **领域自适应 (Domain Adaptation)**：使用领域内数据通过持续预训练来调整通用的PLM 。对于拥有大量未标记文本的领域（如生物医学），从头开始预训练语言模型也是一个好选择 。  
*   **任务特定模型设计 (Task-specific Model Design)**：在PLM之上添加一个或多个任务特定的层来生成目标任务的最终输出 。  
*   **任务特定微调 (Task-specific Fine-tuning)**：任务特定层可以单独训练（固定PLM参数），也可以与PLM一起训练，有时多任务训练也是一个不错的选择 。  
*   **模型压缩 (Model Compression)**：PLM的部署成本很高，通常需要通过知识蒸馏等方法进行压缩，以满足实际应用中的延迟和容量限制 。  

**4.9. 数据集与性能**

PPT中列举了多种用于文本分类的数据集及其性能表现：

*   **情感分析数据集**：Yelp (Yelp-5, Yelp-2), IMDb, Movie Review (MR), Stanford Sentiment Treebank (SST-1, SST-2), Multi-Perspective Question Answering (MPQA), Amazon (二分类和五分类) 。并展示了在这些数据集上的性能排行榜图表 。  
*   **新闻分类数据集**：AG News, 20 Newsgroups, Sogou News, Reuters news (及其衍生数据集如R8, R52, RCV1) 。并展示了在这些数据集上的性能排行榜图表 。  
*   **主题分类数据集**：Ohsumed, DBpedia, EUR-Lex, Web Of Science (WOS-46985, WOS-11967, WOS-5736), PubMed 。并展示了在这些数据集上的性能排行榜图表 。  

* * *

### 5\. NLU任务2：问答 (Question Answering - QA)

**5.1. 基本概念**

*   **核心思想**：从各种材料（文档、对话、在线搜索等）中提取信息，并给出简短扼要的答案以满足用户的信息需求 。  
*   **问题类型**：
    *   **事实型 (Factoid)**：最简单、最常见的问题，答案是简单的事实（例如，“三氧化二汞的化学符号是什么？” “哪支NFL球队代表AFC参加了第50届超级碗？”） 。  
    *   **数学型 (Mathematical)**：例如，“2+3=?” 。  

**5.2. 事实型问答 (Factoid QA)**  

*   **基于信息检索的问答 (Information-Retrieval (IR) based QA / Open-domain QA)**：
    1.  给定用户问题，使用信息检索技术从大量文本（如网页、PubMed等科学论文集）中找到相关段落 。  
    2.  然后，神经阅读理解算法阅读这些检索到的段落，并直接从文本片段中提取答案 。  
*   **基于知识库的问答 (Knowledge-based QA)**：
    1.  构建查询的语义表示（逻辑查询） 。  
    2.  使用这些语义表示来查询事实数据库 。例如，问题“哪些州与德克萨斯州接壤？”可以转换为逻辑查询 λx.state(x)∧borders(x,Texas) 。  

**5.3. 基于信息检索的问答流程**

通常包含两个主要步骤：信息检索 (IR) 和阅读理解 (RC) 。首先通过IR系统从大规模文本集中检索出与问题相关的文档或段落，然后由RC模型读取这些内容并抽取出答案 。  

**5.4. 基于知识库的问答流程**

将自然语言问题转换为逻辑形式的表达式，然后用这个逻辑表达式去查询知识库以获得答案 。例如，问题“基尔·斯塔默多大了？”可能被转换为 `XX.age(X)` 且 `X = PERS(Keir Starmer)`，然后在知识库中查找得到答案 。  

**5.5. Text2SQL**

*   这是一种将自然语言问题转换为SQL查询的技术，通常使用序列到序列 (Seq2Seq) 模型来实现 。  
*   应用场景包括预订餐厅、播放音乐等 。PPT中给出了一个例子，将问题“安娜·纳利克最初演唱的歌曲总数是多少？”转换为SQL查询 `SELECT COUNT Song choice WHERE Original artist = anna christine nalick` 。  

**5.6. 阅读理解 (Reading Comprehension - RC)**

*   **定义**：阅读和理解非结构化文本，并回答相关问题的能力 。  
*   **问题形式**：输入是上下文（一段文本）和问题，输出是答案 。答案可以是：  
    *   **抽取式 (Extractive)**：答案是上下文中的一个子字符串（更侧重理解） 。  
    *   **生成式/抽象式 (Abstractive)**：答案是自由形式的文本（更侧重生成） 。  
*   **与问答的关系**：问答是一个任务，而阅读理解是解决问答任务的一种可能方法。其他方法还包括基于知识库的信息检索、关键词检测机制等 。  
*   **RC需要语言理解能力**：例如，指代消解（理解“她”指的是谁）、推理（推断“特色菜”就是“鲶鱼”）等 。  

**5.7. SQuAD数据集 (Stanford Question Answering Dataset)**

*   **SQuAD 1.1**：
    *   **动机**：针对先前数据集存在的问题（如高质量人工数据库规模小，完形填空式问题不自然等） 。  
    *   **特点**：由众包工人在维基百科文章上提出问题，包含107,785个“问题-上下文-答案”三元组，覆盖536篇文章。答案是对应阅读段落中的文本片段（抽取式问答）。通常每个问题有3个由不同众包工人提供的标准答案 。  
    *   **优势**：人工编写和策划，比CNN/DM等数据集噪声小；非完形填空形式，更接近真实的语言理解 。PPT中给出了SQuAD 1.1的例子 。  
*   **SQuAD 2.0**：
    *   在SQuAD 1.1的10万个问题的基础上，增加了超过5万个由众包工人对抗性编写的、看起来与可回答问题相似但实际无法回答的问题 。  
    *   要求系统不仅能在可能的情况下回答问题，还要能判断段落是否支持答案，并在不支持时拒绝回答 。PPT中给出了一个SQuAD 2.0中“无答案”问题的例子 。  

**5.8. BiDAF模型 (Bidirectional Attention Flow for Machine Comprehension)**  

*   **动机**：更好地将注意力机制融入问答任务 。  
*   **核心特点** ：  
    *   **双向注意力**：查询到上下文 (Query-to-Context, Q2C) 和上下文到查询 (Context-to-Query, C2Q) 的双向注意力流。
    *   包含字符级、词级和上下文嵌入。
    *   注意力加权后的向量与原始嵌入一起传递到后续层。
*   **架构**：字符嵌入层、词嵌入层、上下文嵌入层（使用Bi-LSTM处理查询和上下文的字符及词嵌入）、注意力流层（生成查询感知的上下文特征向量和上下文感知的查询向量）、建模层（使用两层Bi-LSTM进一步提炼表示）、输出层（线性层后接Softmax得到答案开始和结束位置的概率分布） 。  
*   **注意力细节**：计算上下文嵌入H和查询嵌入U之间的相似度矩阵S。Q2C关注上下文中哪些词元与查询更相关，C2Q关注查询中哪些词元与上下文更相关 。  
*   **输出流**：经过Bi-LSTM和拼接操作后，最终通过两个独立的线性层和Softmax层分别预测答案的开始和结束概率 。  
*   **性能指标**：训练时使用正确开始/结束索引的对数似然；测试时选择使开始概率p1(i)与结束概率p2(j)之积最大的(i,j)对（i<j）。评估指标包括精确匹配率 (Exact Match, EM) 和 F1分数 。  
*   PPT展示了BiDAF在SQuAD上的结果、消融研究结果以及在CNN/Daily Mail数据集上的表现 。  
*   **BiDAF总结**：多粒度嵌入、双向注意力、无需早期汇总的查询感知上下文表示，在当时达到了SOTA（State-of-the-Art，即当时最佳）性能 。  

**5.9. 当前的SOTA模型**

*   目前，在问答任务上表现最佳的模型主要是基于预训练语言模型，如BERT 。  
*   通过微调BERT模型（例如，引入两个向量S和E来生成每个词元作为答案开始和结束的概率），可以在SQuAD等数据集上取得优异成绩。对于SQuAD 2.0中无法回答的问题，开始和结束指针都指向\[CLS\]符号 。PPT展示了BERT出现后SQuAD排行榜的变化 。  

**5.10. 问答领域的未来改进方向**

*   **SQuAD数据集的局限性** ：  
    *   仅支持片段抽取式答案（没有是/否、计数、隐含的“为什么”等类型的问题）。
    *   问题是看着段落构建的，并非源于真实的信息需求（问题存在偏见）。
    *   问题和答案片段之间通常有较大的词汇和句法匹配度，几乎没有多事实/多句子推理。
*   **未来需求** ：  
    *   **可解释性 (Explainability)**：模型不仅要给出答案，还要能解释为什么。
    *   **知识驱动的问答 (Knowledge grounded QA)**：结合外部知识库进行更深入的理解和推理。
    *   **多事实/多句子推理 (Multi-fact/sentence inference)**：处理需要综合多个信息点才能回答的问题。
*   PPT还提到了“Factoid”一词的特定含义：指因其呈现方式或在印刷品中反复出现而被认为是真实但实际并不可靠的信息片段 。并举例说明了ChatGPT可能出现的错误 。  

* * *

### 6\. 问答任务总结

 

*   问答是一项重要的任务，它依赖于语言理解的多个组成部分。
*   数据集对于阅读理解至关重要，但创建大规模且能真正检验语言理解能力的数据集非常困难。
*   目前模型在较简单的数据集上表现良好，但在更具挑战性的数据集上仍有提升空间。
*   注意力机制和预训练技术对提升问答性能非常有帮助。

希望这份讲解能够帮助您更好地理解这份关于NLU任务的PPT！

信息来源和相关内容

---

![PDF 图标](https://drive-thirdparty.googleusercontent.com/32/type/application/pdf)

L6\_NLG tasks

PDF

非常棒，请你接着讲解该ppt（用中文讲解给我，部分地方中英双语）

---

显示思路

好的，我们继续讲解这份关于自然语言生成 (Natural Language Generation, NLG) 任务的PPT。

### 本讲座大纲 (Outline)

1.  **自然语言生成任务 (Natural language generation tasks)**
2.  **机器翻译 (Machine translation)**
    *   什么是机器翻译？ (What is machine translation?)
    *   神经机器翻译 (Neural machine translation)
    *   评估 (Evaluation)
3.  **对话系统 (Dialogue system)**
    *   什么是对话系统？ (What is a dialogue system?)
    *   面向任务的对话系统 (Task-oriented dialogue systems)
    *   闲聊型对话系统 (Chitchat dialogue systems)
    *   性能评估 (Performance evaluation)

* * *

### 1\. 自然语言生成任务 (Natural Language Generation Tasks)

*   **定义 (Definition)**：NLG 专注于生成类似人类的文本，以有效地传递信息或进行交流 (NLG focuses on generating human-like text that conveys information or communicates effectively)。
*   **具体任务 (Generation tasks)**：
    *   机器翻译 (Machine translation)
    *   复述/改写 (Paraphrasing (rewriting))
    *   报告生成和长文本生成 (Report generation & long text generation)
    *   摘要 (Summarization)
    *   对话生成 (Dialog generation)

* * *

### 2\. 机器翻译 (Machine Translation - MT)

**2.1. 什么是机器翻译？ (What is machine translation (MT)?)**

*   将一个句子x从一种语言（源语言 - source language）翻译成另一种语言（目标语言 - target language）的句子y。
    *   例如 (e.g.)，源语言 (x): "L'homme est né libre, et partout il est dans les fers" (法语 - French)
    *   目标语言 (y): "Man is born free, but everywhere he is in chains" (英语 - English) - 卢梭 (Rousseau)

**2.2. 机器翻译发展时间线 (Timeline of machine translation)**

*   **1950年代**: 基于规则的机器翻译 (Rule-based MT)
    *   基于词典的机器翻译 (Dictionary-based MT)
    *   基于转换的机器翻译 (Transfer-based MT)
    *   中间语言机器翻译 (Interlingual MT)
*   **1980年代-1990年代**: 基于实例的机器翻译 (Example-based MT)
*   **1990年代 - 2010年代**: 统计机器翻译 (Statistical Machine Translation, SMT)
*   **当前趋势 (Current trend)**: 神经机器翻译 (Neural Machine Translation, NMT) (约2015年至今)

**2.3. 统计机器翻译模型 (Statistical MT model)**

*   核心问题：如何学习翻译模型 P(y∣x)？（PPT中为 P(x∣y), 但通常目标是 P(y∣x)，即给定源句x生成目标句y的概率）。
*   **数据需求 (Data Requirement)**：需要大量的平行语料数据 (parallel data)，例如人工翻译的德语句子和英语句子对。
*   **引入隐变量 (Latent Variable)**：为了从平行语料中学习 P(y∣x)（或 P(x∣y)），引入一个隐变量 a，即 P(x,a∣y)。
    *   a 指的是**对齐 (alignment)**：源句子x和目标句子y之间词级别的对应关系。
    *   语言之间的类型差异 (Typological differences) 会导致复杂的对齐。
    *   有些词可能没有对应词（例如，"spurious" word，即目标语中多余的词）。
*   **对齐的细节 (Alignment details)**：
    *   **多对一 (Many-to-one Alignment)**：源语言中的多个词对应目标语言中的一个词。
    *   **一对多 (One-to-many Alignment)**：源语言中的一个词（称为"fertile word"，能产词）对应目标语言中的多个词。
    *   **多对多 (Many-to-many Alignment)**：通常在短语级别 (phrase level) 出现，源语言中的多个词对应目标语言中的多个词。

**2.4. 神经机器翻译 (Neural Machine Translation - NMT)**

*   **NMT的成功 (NMT: the biggest success story of NLP deep learning)**：
    *   NMT从2014年一个边缘的研究活动发展成为2016年的主流标准方法。
    *   2014年：首篇seq2seq论文发表。
    *   2016年：谷歌翻译从SMT切换到NMT。
    *   由数百名工程师多年构建的SMT系统，被少数工程师在几个月内训练出的NMT系统所超越。
*   **Seq2seq模型 (Seq2seq model)**：
    *   是一种编码器-解码器 (encoder-decoder) 神经网络架构，用于将序列从一个域（如英语句子）转换到另一个域。它包含两个RNN：编码器RNN (encoder RNNs) 和解码器RNN (decoder RNNs)。
    *   **编码器 (Encoder)**：读取源语言句子，并将其编码成一个固定长度的向量（上下文向量 S 或 C）。
    *   **解码器 (Decoder)**：以编码器生成的上下文向量为初始状态，逐词生成目标语言句子。解码器的输出通过Softmax层得到词汇表中每个词的概率。
*   **Seq2seq作为条件语言模型 (Seq2seq as a conditional language model)**：
    *   Seq2seq模型是条件语言模型的一个例子。
    *   **“语言模型” (Language model)**：因为解码器在预测目标句子y的下一个词。
    *   **“条件” (Conditional)**：因为其预测还以源句子x为条件。
    *   NMT直接计算 P(y∣x)\=P(y1​∣x)P(y2​∣y1​,x)...P(yT​∣y1​,...,yT−1​,x)。
    *   这个概率是基于编码器RNN输出的隐藏状态C（上下文向量）。
*   **训练Seq2seq NMT系统 (Train a seq2seq NMT system)**：
    *   Seq2seq作为一个单一系统进行优化。
    *   反向传播 (Backpropagation) 进行“端到端 (end-to-end)”训练。
    *   损失函数 (Loss function) J 通常是每个时间步 t 的负对数概率之和的平均值：J\=T1​∑t\=1T​Jt​，其中 Jt​\=−logP(yt​∣y1​,...,yt−1​,x)。
*   **解码策略 (Decoding Strategies)**：
    *   **贪心解码 (Greedy decoding)**：在解码器的每一步都选择概率最高的词。问题在于一旦做出决定就无法撤销 (no way to undo decisions)。
    *   **穷举搜索解码 (Exhaustive search decoding)**：目标是找到一个翻译 y 使得 P(y∣x)\=∏t\=1T​P(yt​∣y1​,...,yt−1​,x) 最大化。问题是计算所有可能的序列 y 的复杂度为 O(VT)（V是词汇表大小，T是序列长度），这在实际中是不可行的。
    *   **集束搜索解码 (Beam search decoding)**：
        *   在解码的每一步，保留 k 个最可能的部翻译（称为 k 个假设 - hypotheses）。k 称为集束大小 (beam size)，实践中通常取5到10。
        *   一个假设 (y1​,...,yt​) 的得分 (score) 是其对数概率：score(y1​,...,yt​)\=logPLM​(y1​,...,yt​∣x)\=∑i\=1t​logPLM​(yi​∣y1​,...,yi−1​,x)。得分都是负数，越接近0越好。
        *   由于较长的假设得分自然更低，通常会按长度进行归一化：t1​∑i\=1t​logPLM​(yi​∣y1​,...,yi−1​,x)。
        *   PPT中给出了一个 k\=2 的集束搜索示例。
    *   **集束搜索的停止标准 (Stopping criteria for beam search)**：
        *   当一个假设生成 `<END>` 标记时，该假设完成，将其放在一边，继续探索其他假设。
        *   通常集束搜索会持续直到：
            *   达到预设的最大时间步 T。
            *   或者，已经获得了至少 n 个完成的假设（n是预设的截止数量）。
*   **有无注意力的Seq2seq对比 (Seq2seq with vs. without attention)**：
    *   **无注意力 (Without attention)**：解码器在生成每个词时都依赖于编码器输出的同一个上下文向量C。
    *   **有注意力 (With attention)**：解码器在生成每个词时，会根据当前解码状态动态地计算一个上下文向量 ct​，这个向量是编码器各个时间步输出的加权和，权重由注意力机制决定。这意味着解码器在每一步都会关注源句子的不同部分。
*   **基于Transformer的机器翻译 (Transformer-based MT)**：
    *   采用编码器-解码器结构 (Encoder-decoder structure)。
    *   核心是多头自注意力机制 (multi-head self-attention)。
    *   解码过程类似于seq2seq模型，也可以使用集束搜索等策略。

**2.5. 机器翻译模型评估 - BLEU (MT model evaluation - BLEU)**

*   **BLEU (Bilingual Evaluation Understudy)**：双语评估替补。
*   将机器生成的翻译（候选句 - candidate sentence）与一个或多个人工翻译（参考句 - reference sentences）进行比较，并基于以下几点计算相似度得分：
    *   **n-gram精度 (pn​) (n-gram precision)**：通常计算1-gram, 2-gram, 3-gram和4-gram的精度。
        *   计算方法：统计候选句中n-gram在所有参考句中出现次数的最小值之和，然后除以候选句中n-gram的总数。PPT中详细展示了1-gram到4-gram精度的计算过程。
    *   **简短惩罚因子 (Brevity Penalty, BP)**：用于惩罚过短的系统翻译。
        *   BP\={1e(1−r/c)​if c\>rif c≤r​，其中 c 是候选句长度， r 是参考句长度（当有多个参考句时，通常选择与候选句长度最接近的参考句长度）。
    *   **BLEU得分计算公式 (BLEU formula)**： BLEU\=BP⋅exp(∑n\=1N​wn​logpn​)，其中 wn​ 是n-gram精度的权重，通常取平均值（如N=4时，wn​\=0.25）。
*   **BLEU得分解释 (BLEU score interpretation)**：
    *   PPT中提供了一个BLEU得分范围及其对应的翻译质量描述（例如，<10几乎无用，30-40可理解到良好，>60质量通常优于人工）。
    *   **重要提示**：强烈不建议跨不同语料库和语言比较BLEU分数 (Comparing BLEU scores across different corpora and languages is strongly discouraged!)。
*   **BLEU的局限性 (BLEU is useful but imperfect)**：
    *   一个句子可以有很多种有效的翻译方式。
    *   因此，一个好的翻译可能因为与人工翻译的n-gram重叠度低而得到较低的BLEU分数。

**2.6. NMT性能 (NMT performance)**

*   PPT中展示了WMT2014数据集上英语-法语、法语-英语、英语-德语、德语-英语等多个语言对的机器翻译BLEU得分排行榜随时间的变化趋势，显示了NMT模型（特别是基于Transformer和预训练模型如FLAN）的显著进步。

**2.7. 机器翻译面临的挑战 (Remaining challenges in MT)**

*   **未登录词 (Out-of-vocabulary words)**：处理词汇表中未出现过的词。
*   **在长文本中保持上下文 (Maintaining context over longer text)**。
*   **低资源语言对 (Low-resource language pairs)**：缺乏大量平行语料的语言。
*   **常识运用仍然困难 (Using common sense is still hard)**。
*   **习语难以翻译 (Idioms are difficult to translate)**。

* * *

### 3\. 对话系统 (Dialogue Systems)

**3.1. 什么是对话系统？ (What is a dialogue system?)**

*   对话系统是一个旨在与人类进行对话的计算机系统 (A dialogue system is a computer system intended to converse with a human)。
*   它可以使用文本、语音、图形、触觉、手势等多种模式进行输入和输出。
*   本讲座主要关注基于自然语言的、用于不同目的（如获取知识、预订票务、随意聊天等）的对话系统。
*   常见的例子包括 Apple Siri, Google Now/Assistant, Microsoft Cortana, Amazon Alexa/Echo 等。
*   对话系统分为**口语对话系统 (Spoken dialogue systems)**（如智能音箱）和**基于文本的聊天机器人 (Text-based chatbot)**。

**3.2. 对话任务的类别 (Categories of dialog tasks)**

*   **面向任务的对话系统 (Task-oriented)**：
    *   可以是开放领域 (open-domain) 或封闭领域 (close-domain)。
    *   旨在识别用户的任务并执行相应的操作以完成目标 (aim at recognize the task of the user and execute corresponding tasks to accomplish the goal)。
    *   例如：预订餐厅、预订电影票、查询账户余额等。
*   **闲聊型对话系统 (Chitchat)**：
    *   通常是开放领域的 (open-domain)。
    *   旨在以对话的方式回应用户输入，进行社交互动 (aim at respond to the user input in a conversational manner, e.g., making socially engaging conversations)。

**3.3. 面向任务的对话系统 (Task-oriented dialogue systems)**

*   **典型流程 (Typical pipeline)**： `ASR (optional) -> SLU (NLU) -> DST -> DM (<-> knowledge base) -> NLG -> TTS (optional)`
    *   **ASR (Automatic Speech Recognition)**：自动语音识别（可选，用于口语输入）。
    *   **SLU (Spoken Language Understanding) / NLU (Natural Language Understanding)**：口语/自然语言理解。
    *   **DST (Dialogue State Tracking)**：对话状态跟踪。
    *   **DM (Dialogue Management) / DLP (Dialogue Policy Learning)**：对话管理/对话策略学习，与知识库 (Knowledge Base, KB) 交互。
    *   **NLG (Natural Language Generation)**：自然语言生成。
    *   **TTS (Text to Speech)**：文本到语音转换（可选，用于语音输出）。
*   **NLU在对话系统中的具体任务 (NLU in dialogue systems - Pipelined tasks)**：
    *   **领域分类 (Domain classification)**：根据用户输入判断其属于哪个预定义的领域（如餐厅、出租车、电影）。需要预定义的领域本体 (domain ontology)。
    *   **意图检测 (Intent detection)**：识别用户在特定领域内的具体意图（如在餐厅领域中，意图可能是查找餐厅、查找价格、查找类型等）。需要预定义的模式 (schema)。
    *   **槽位填充 (Slot filling)**：从用户输入中提取完成任务所需的具体信息（如餐厅预订中的“评分=好”、“类型=台湾菜”）。通常使用BIO (Beginning, Inside, Outside) 标注方案进行序列标注。
*   **对话状态跟踪 (Dialogue State Tracking, DST)**：
    *   维护对话的当前状态，并根据观察进行更新。可以想象成一个状态机，包含状态、输入、转换和输出。类似于强化学习中的场景。
*   **对话策略学习/对话管理 (Dialogue Policy Learning (DLP) or Dialogue Management (DM) for agent action)**：
    *   决定系统下一步应该采取什么行动。
    *   如果信息足够，则生成输出话语（例如，“最近的一家在台北101”）。
    *   如果信息不足，则向用户请求信息（例如，“您当前的位置是哪里？”）。
    *   如果不确定，则向用户确认信息（例如，“您想要台湾菜吗？”）。
*   **NLG在对话系统中的作用 (NLG in dialogue systems)**：
    *   根据选择的对话动作生成自然语言或图形用户界面 (GUI) 进行交互。

**3.4. 实现示例 (Implementation examples)**

*   **分类任务 (Classification tasks - 领域分类和意图检测)**：
    *   主要被视为话语分类任务 (utterance classification task)。
    *   给定带标签的话语集合，训练模型来估计新话语的标签。
    *   常用模型：深度信念网络 (Deep belief nets)、深度凸网络 (Deep convex nets)、RNNs、CNNs、大型语言模型 (LLMs)。
*   **序列标注任务 (Sequence tagging tasks - 例如，槽位填充的BIO标注)**：
    *   给定带标签的词序列集合，目标是为新的词序列估计标签。
    *   常用模型：RNNs、注意力机制 (attentions)。
*   **对话状态跟踪 (Dialogue state tracking)**：
    *   维护对话状态的信念，并根据观察进行更新。
    *   示例：神经信念追踪器 (neural belief tracker)、基于CNN/RNN的追踪器。
*   **对话策略学习 (Dialogue policy learning, DPL)**：
    *   通常使用（深度）强化学习 ((deep) reinforcement learning)。
*   **自然语言生成 (NLG)**：
    *   Seq2seq模型。
    *   受控生成 (controlled generation)：基于模板 (template based)、基于句法树 (syntax tree based)。
*   **预训练模型的应用 (Pretrained models)**：
    *   BERT, GPT, DialoGPT等预训练模型可用于对话系统中的各种任务。
    *   例如DialogBERT，通过掩码上下文回归 (Masked context regression) 和分布式话语顺序排序 (Distributed utterance order ranking) 等任务进行预训练。

**3.5. 闲聊型对话系统 (Chitchat dialogue systems)**

*   从离线的对话语料库中学习生成对话 (Learns to generate dialogues from offline dialogue corpora)。
*   通常不涉及明确的状态、动作、意图、槽位等概念 (No state, action, intent, slot, etc.)。
*   可以整合外部信息/特征，如情感、主题、知识背景等。
*   **示例模型**：
    *   Seq2seq模型（例如，Serban et al., 2016, 使用生成式分层神经网络模型构建端到端对话系统）。
    *   带有隐变量的分层seq2seq模型 (A hierarchical seq2seq model with Gaussian latent variable for generating dialogues)，隐变量可以代表主题或情感等（Serban et al., 2017）。
    *   基于知识背景的神经对话模型 (A knowledge-grounded neural conversation model) (Ghazvininejad et al., 2017)。

**3.6. 性能评估 (Performance evaluation)**

*   **理想情况**：人工评估 (human evaluation)，但这对于训练来说不可行，即使对于评估来说成本也很高。
*   **启发式指标 (Heuristics)**：
    *   困惑度 (Perplexity)
    *   BLEU分数
    *   对话长度/多样性 (Dialogue length/diversity)
    *   流畅性 (Fluency)
    *   参与度 (Engagingness)
    *   一致性 (Consistency)
*   对话系统的评估仍然是一个开放性问题 (Still an open question)。
*   PPT中展示了一个对seq2seq模型 (NCM) 和CleverBot (CB) 进行人工评估的例子，评委根据偏好选择哪个机器人更好，或者判断为平局。

**3.7. 个性化对话代理：PERSONACHAT (Personalizing dialogue agents: PERSONACHAT)**

*   Zhang et al., 2018 的工作，旨在让对话代理具有一致的个性。
*   构建了一个包含明确角色信息的数据集 (Dataset with consistent personalities)。
    *   1155个角色 (personas)
    *   10,981个对话 (dialogues)
    *   164,356句话语 (utterances)
    *   每个对话包含3-5个角色描述句，6-8轮聊天。
*   **评估指标 (Evaluation Metrics)**：
    *   困惑度 (Perplexity)
    *   在20个候选话语中的Hit@1准确率 (Hit@1 accuracy among 20 candidate utterances)
    *   F1分数 (F1 score)
    *   人工评估 (Human evaluation)
*   **模型类型 (Models)**：
    *   **排序模型 (Ranking models)**：从训练集中选择回应。
        *   基于tf-idf词袋的IR基线 (tf-idf BoW based IR baseline)
        *   StarSpace Embedding
        *   Ranking Profile Memory Network
        *   Key-Value (KV) Profile Memory Network
    *   **生成模型 (Generative models)**：逐词生成回应。
        *   Seq2Seq
        *   Generative Profile Memory Network
*   PPT展示了排序模型和生成模型在PERSONACHAT上的评估结果，以及人工评估的结果。

* * *

### 4\. 总结 (Summary of L6)

*   本讲座主要介绍了两种NLG任务：机器翻译和对话系统。
*   **面向任务的对话系统 (Task-oriented dialogue systems)** 的典型流程和关键技术。
*   **闲聊型对话系统 (Chitchat dialogue systems)** 的特点和常用模型。
*   **性能评估 (Performance evaluation)** 的方法和挑战。
*   **未在讲座中详细讨论的挑战 (Challenges (not covered in the lecture))**：
    *   知识准确性（事实型）(Knowledge accuracy (Factoid))
    *   伦理 (Ethics)
    *   偏见 (Bias)
    *   可解释性 (Explainability)
    *   隐私和数据安全 (Privacy and data security)

这份PPT对自然语言生成中的核心任务进行了概述，特别是机器翻译和对话系统，并讨论了它们的发展、关键技术和评估方法。希望这次的讲解对您有所帮助！

信息来源和相关内容

---

![PDF 图标](https://drive-thirdparty.googleusercontent.com/32/type/application/pdf)

L7\_LLM

PDF

非常棒，请你接着讲解该ppt

---

显示思路

好的，我们来解读这份关于大型语言模型 (Large Language Models, LLMs) 的PPT。

### 本讲座大纲 (Outline)

1.  **大型语言模型定义 (Large language model definition)**
2.  **规模法则和涌现能力 (Scaling law and emergent abilities)** 3.. **对齐 (Alignment)**

* * *

### 1\. 大型语言模型定义 (Large Language Model Definition)

*   **ChatGPT的崛起 (ChatGPT becomes the fastest-growing 'app' of all time)**：
    
    *   ChatGPT (Chat Generative Pre-trained Transformer) 是由OpenAI开发并于2022年11月发布的人工智能聊天机器人。
    *   它能够以更自然的方式与人类交流。
    *   **示例 (Examples)**：解释量子计算、为10岁孩子生日提供创意、编写HTTP请求的Javascript代码。
    *   **能力 (Capabilities)**：记住用户之前的对话内容、允许用户进行后续更正、被训练来拒绝不当请求。
    *   **局限性 (Limitations)**：可能偶尔产生不正确的信息、可能偶尔产生有害指令或带偏见的内容、对2021年之后的世界和事件知识有限。
*   **LLM的强大文本处理能力 (LLMs have powerful textual processing abilities)**：
    
    *   LLM能够编写和调试计算机程序、模仿名人CEO的风格撰写商业提案、创作音乐、电视剧本、童话和学生论文、回答测试问题（有时在某些测试中表现超过人类平均水平）、写诗和歌词、模拟Linux系统、模拟整个聊天室、玩井字游戏和模拟ATM机。
    *   PPT中展示了一张包含从2019年到2023年众多LLM发展的时间线图，例如GPT系列、T5、PaLM、LaMDA、BLOOM、LLaMA等，并标注了部分可公开获取的模型。
*   **LLM示例 (Example of LLM)**：
    
    *   **聊天机器人 (Chatbot)**：展示了ChatGPT进行故事创作和讲英语笑话的能力。
    *   **编写代码和解决数学问题 (Write code and solve math problem)**：展示了ChatGPT编写Python冒泡排序代码和解决鸡兔同笼数学问题的能力。
*   **GPT模型的演进 (The evolution of GPT models)**：
    
    *   **GPT-1 (2018.06)**：发布第一个版本。
    *   **GPT-2 (2019.02)**：发布第二个版本。
    *   **GPT-3 (2020.05)**：预印本论文发表，API于2021年11月公开。
    *   **ChatGPT (2022.11)**：在OpenAI博客上宣布，API于2023年3月可用。
    *   **GPT-4 (2023.03)**：通过ChatGPT发布，API将很快公开。
    *   PPT中还展示了从GPT-3系列到GPT-3.5系列再到ChatGPT和GPT-4的详细演进路径，包括了基于代码的训练 (Training on code)、指令微调 (Instruction tuning) 和来自人类反馈的强化学习 (Reinforcement Learning from Human Feedback, RLHF) 等关键技术节点。例如，GPT-3初始模型 (Davinci) 经过代码训练得到Codex初始模型，再经过指令微调得到InstructGPT初始模型。GPT-3.5系列中的`text-davinci-003`和ChatGPT都是通过RLHF得到的。
*   **小型语言模型 (SLMs) 与大型语言模型 (LLMs) 的对比 (Comparisons between SLMs and LLMs)**：
    
    | 特性 (Aspect) | 小型语言模型 (Small Language Models) | 大型语言模型 (Large Language Models) | | :-------------------------------- | :-------------------------------------------------------------------- | :----------------------------------------------------------------------- | | **大小 (Size)** | 参数量可能少于1500万 (Can have less than 15 million parameters) | 参数量可达数千亿 (Can have hundreds of billions of parameters) | | **计算需求 (Computational Requirements)** | 可使用移动设备处理器 (Can use mobile device processors) | 可能需要数百个GPU处理器 (Can require hundreds of GPU processors) | | **性能 (Performance)** | 可处理简单任务 (Can handle simple tasks) | 可处理复杂多样的任务 (Can handle complex, diverse tasks) | | **部署 (Deployment)** | 在资源受限的环境中更容易部署 (Easier to deploy in resource-constrained environments) | 部署通常需要大量基础设施 (Deployment often requires substantial infrastructure) | | **训练 (Training)** | 可能需要一周时间训练 (Can be trained a week) | 训练可能需要数月时间 (Training can take months) |
    
*   **提示工程 (Prompt engineering)**：
    
    *   **提示 (Prompt)** 是一段插入到输入样本中的文本，以便将原始任务形式化为（掩码）语言建模问题 (A prompt is a piece of text inserted in the input examples, so that the original task can be formulated as a (masked) language modeling problem)。
    *   **ChatGPT + 提示 (ChatGPT + Prompt)**：展示了ChatGPT Playground中各种预设的提示模板，用于执行问答、摘要、语法纠错、自然语言到API代码生成、文本到命令、文本分类、代码解释、关键词提取等多种任务。
*   **范式转变 (Paradigm shift)**：
    
    *   从**预训练-微调 (pretraining-finetuning)** 到**提示 (prompting)**。
    *   **预训练-微调**：预训练模型是可调的 (tunable)，并为下游任务进行适配。
    *   **提示**：预训练模型通常是冻结的 (frozen)，通过添加一些输入前缀 (input prefix) 来使用模型执行不同任务。
    *   PPT中对比了三种模式：
        *   **模型调整 (Model Tuning / Fine-Tuning)**：预训练模型可调。
        *   **提示调整 (Prompt Tuning)**：预训练模型冻结，但提示是可学习的“软提示 (soft prompt)”。
        *   **提示设计 (Prompt Design)**：预训练模型冻结，提示是人工设计的“工程化提示 (engineered prompt)”。

* * *

### 2\. 规模法则和涌现能力 (Scaling Law and Emergent Abilities)

*   **大型语言模型的时代 - 规模的福祉 (The era of large language models - The blessings of scale)**：
    *   PPT展示了一张图，显示了从1950年到2022年AI训练运行中估计使用的计算资源（以浮点运算次数衡量，对数尺度）。图表清晰地显示出近年来语言模型（如NPLM, BERT-Large, GPT系列, PaLM, LaMDA）的计算资源消耗急剧增加，远超早期的视觉和绘图模型。
*   **LLM有多大？(How large are large language models)**：
    *   大型模型的参数规模、训练数据量和计算成本都在指数级增长 (NLP model size and computation are increasing exponentially)。
    *   PPT展示了从2017年的Google Transformer (0.05B参数) 到2021年的Google Switch Transformer (1.6T参数) 和OpenAI GPT-3 (170B参数)、Megatron-Turing NLG (530B参数) 等模型的参数量增长趋势，形象地称之为“NLP的摩尔定律：模型大小每年增长10倍 (NLP's Moore's Law: Every year model size increases by 10x)”。
*   **训练数据达到万亿级别 (Training data reaches Trillions)**：
    *   为了验证他们的发现，研究者用与Gopher相同的计算预算训练了一个LLM (即Chinchilla)，其模型参数远少于Gopher (70B vs 280B)，但训练词元数量远多于Gopher (1.4T vs 300B)。
    *   表格列出了多个知名LLM的参数量和训练词元数，如LaMDA (137B参数, 168B词元), GPT-3 (175B参数, 300B词元), Chinchilla (70B参数, 1.4T词元)。
    *   一张图表比较了人类（以13岁计）、BERT、RoBERTa、GPT-3和Chinchilla在训练期间见过的词元数量，Chinchilla达到了1.4万亿。
*   **LLM的规模法则 (Scaling law of LLMs)** (基于Kaplan et al., 2020, "Scaling Laws for Neural Language Models")：
    *   **规模法则**是指模型性能的简单、可预测的规则。
    *   **经验观察**：扩大模型规模会导致困惑度 (perplexity) 的可靠提升。
    *   **性能度量 (Performance measures)**：交叉熵损失 (cross entropy loss, L)，通常在上下文中的词元上取平均。
    *   **资源度量 (Resource measures)**：
        *   **模型大小 (Model size, N)**：不包括词汇表和位置嵌入的参数数量。
        *   **数据集大小 (Dataset size, D)**：以词元 (tokens) 数量计。
        *   **计算能力 (Computing power, C)**：总的非嵌入训练计算量，估计为 C≈6NBS（B为批大小，S为训练步数）。单位通常为PF-days（PetaFLOP-days）。
    *   **主要发现 (Key Findings)**：
        *   **性能主要取决于规模，弱依赖于模型形状 (Performance depends strongly on scale, weakly on model shape)**：性能最主要受模型参数N、数据集大小D和训练计算量C这三个规模因素影响。在合理范围内，性能对其他架构超参数（如深度与宽度）的依赖性很弱。
        *   **平滑的幂律关系 (Smooth power laws)**：当不受其他两个因素瓶颈限制时，性能与N, D, C三个规模因素均呈现幂律关系，这种趋势跨越了超过六个数量级。目前未观察到这种趋势在高端出现偏离，尽管性能最终必须在达到零损失之前趋于平缓。PPT中展示了测试损失 (Test Loss) 分别随计算量 (Compute)、数据集大小 (Dataset Size) 和参数量 (Parameters) 变化的幂律曲线图。
        *   **过拟合的普遍性 (Universality of overfitting)**：只要同步扩大N和D，性能就会可预测地提高；但如果固定N或D中的任何一个而增加另一个，则会进入收益递减的状态。性能惩罚可预测地取决于比率 N0.74/D，这意味着模型大小每增加8倍，数据量大约只需要增加5倍即可避免惩罚。
        *   **训练的普遍性 (Universality of training)**：训练曲线遵循可预测的幂律，其参数大致独立于模型大小。通过外推训练曲线的早期部分，可以大致预测如果训练更长时间会达到的损失。
        *   **样本效率 (Sample efficiency)**：大模型比小模型具有更高的样本效率，可以用更少的优化步骤和更少的数据点达到相同的性能水平。
*   **涌现能力：零样本/少样本学习 (Emergent abilities: zero-shot/few-shot learning)**：
    *   以GPT-3 (175B参数) 为例，其模型规模和数据量再次大幅增加。
    *   仅仅通过训练语言模型，就能出人意料地获得解决其他任务（如文本生成、机器翻译、阅读理解）的能力。
    *   **零样本 (Zero-shot)** 可能过于困难，但**少样本学习 (Few-shot learning / In-context learning)** 表现良好。即在不进行梯度更新的情况下，通过在模型的输入中提供少量任务相关的示例，模型就能理解并执行新任务。
    *   PPT展示了在SuperGLUE基准上，GPT-3 175B模型进行少样本学习的性能随着上下文中示例数量 (K) 的增加而提升，并接近甚至超过了经过微调的SOTA模型和人类水平。
    *   **示例**：英语到法语的翻译。
        *   **零样本 (Zero-shot)**：直接给出 "Translate English to French: cheese => "
        *   **单样本 (One-shot)**：给出 "Translate English to French: sea otter => loutre de mer cheese => "
        *   **少样本 (Few-shot)**：给出多个翻译示例，然后是 "cheese => "
*   **涌现能力：思维链 (Emergent abilities: Chain-of-Thought, CoT)**：
    *   **标准提示 (Standard Prompting)**：直接给出问题，模型输出答案。对于复杂推理问题，可能得到错误答案。
    *   **思维链提示 (Chain-of-Thought Prompting)**：在提示中不仅给出问题和答案，还给出推导出答案的中间推理步骤。这引导模型在解决新问题时也生成类似的推理链，从而提高复杂推理任务的准确性。
    *   PPT对比了标准提示和CoT提示在解决数学应用题时的效果，CoT显著提升了模型的正确率。
    *   研究表明 (Wei et al., 2022)，对于足够大的模型（如PaLM 62B, 540B），CoT提示才能有效提升性能，而对于较小的模型（如LaMDA, GPT的早期版本）效果不明显，甚至可能有害。这体现了CoT也是一种涌现能力。

* * *

### 3\. 对齐 (Alignment)

*   **语言模型与用户意图不对齐 (Language modeling - assisting users - Language models are not aligned with user intent)**：
    *   原始的语言模型（如GPT-3）在被要求以特定方式解释概念时（例如，用几句话向6岁孩子解释登月），可能无法生成符合用户期望的、有用的回答，而是生成与提示风格相似的更多提示或不相关内容。
*   **指令微调 (Instruction finetuning)**：
    *   **方法**：收集大量（指令，输出）对的样本，涵盖多种任务，并用这些数据对语言模型进行微调。
    *   **效果**：经过指令微调的模型（如FLAN-T5）能够更好地理解和遵循指令，即使在未见过的任务上也能表现良好。
    *   PPT展示了FLAN-T5在指令微调前后对于一个消歧问答任务的不同表现。微调前，模型输出与问题不符；微调后，模型能按要求逐步思考并给出正确答案。
*   **InstructGPT：来自人类反馈的强化学习 (InstructGPT: Reinforcement Learning from Human Feedback, RLHF)**：
    *   这是一个三步过程，旨在使语言模型的行为与用户意图和偏好更一致：
        1.  **收集演示数据，训练监督策略 (Collect demonstration data, and train a supervised policy)**：
            *   从提示数据集中采样一个提示。
            *   人类标注员演示期望的输出行为。
            *   使用这些（提示，演示输出）数据对预训练的GPT-3模型进行有监督的微调 (Supervised Fine-Tuning, SFT)。
        2.  **收集比较数据，训练奖励模型 (Collect comparison data, and train a reward model, RM)**：
            *   从提示数据集中采样一个提示，并使用上一步训练的SFT模型生成多个输出。
            *   人类标注员对这些输出从最好到最差进行排序。
            *   使用这些比较数据训练一个奖励模型，该模型学习预测人类更偏好哪个输出。
        3.  **使用强化学习优化策略 (Optimize a policy against the reward model using reinforcement learning)**：
            *   从数据集中采样一个新的提示。
            *   当前的策略（即语言模型）生成一个输出。
            *   奖励模型对这个输出进行打分（计算奖励）。
            *   使用这个奖励通过强化学习算法（如PPO, Proximal Policy Optimization）更新策略。
*   **ChatGPT：用于对话代理的指令微调 + RLHF (ChatGPT: Instruction Finetuning + RLHF for dialog agents)**：
    *   ChatGPT的训练也使用了RLHF，方法与InstructGPT类似，但在数据收集设置上略有不同。
    *   初始模型通过有监督微调进行训练：人类AI训练员提供对话，他们在对话中扮演用户和AI助手的双方角色。训练员可以访问模型编写的建议来帮助他们撰写回复。
    *   这个新的对话数据集与InstructGPT数据集（已转换为对话格式）混合使用。
    *   OpenAI等公司对于ChatGPT的具体训练细节（包括数据、训练参数、模型大小）趋于保密，可能是为了保持竞争优势。

* * *

### 总结 (Summary of L7)

*   **大型语言模型定义 (Large language model definition)**：回顾了LLM的定义、能力、演进及与SLM的区别，以及提示工程的重要性。
*   **规模法则和涌现能力 (Scaling law and Emergent ability)**：讨论了模型规模（参数、数据、计算）与性能的幂律关系，以及LLM在达到一定规模后展现出的零样本/少样本学习和思维链等先前未预料到的能力。
*   **指令微调 (Instruction tuning)**：通过在大量（指令，输出）对上进行微调，使LLM能更好地遵循用户指令。
*   **来自人类反馈的强化学习 (RLHF)**：通过人类的偏好反馈来进一步优化LLM，使其输出更符合人类期望。
*   **未在讲座中讨论的挑战 (Challenges (not covered in the lecture))**：
    *   **幻觉 (Hallucination)**：模型可能一本正经地编造不实信息。
    *   **偏见和公平性 (Bias and fairness)**：模型可能从训练数据中学习并放大社会偏见。
    *   **可解释性 (Explainability)**：理解LLM为何做出特定决策仍然困难。
    *   **隐私和数据安全 (Privacy and data security)**：训练和使用LLM涉及大量数据，引发隐私担忧。

这份关于大型语言模型的PPT内容非常前沿和重要，涵盖了从基本定义到最新技术（如RLHF）以及未来挑战的广泛议题。

信息来源和相关内容

---

![PDF 图标](https://drive-thirdparty.googleusercontent.com/32/type/application/pdf)

L8\_prompts\_alignment

PDF

非常棒，请你接着讲解该ppt

---

显示思路

好的，我们来详细解读这份关于大型语言模型 (Large Language Models, LLMs) 的提示 (Prompting) 与对齐 (Alignment) 的PPT。

### 本讲座大纲 (Outline)

1.  **提示学习 (Prompt Learning)**
2.  **思维链提示 (Chain of Thought Prompting)**
3.  **对齐 (Alignment)**

* * *

### 1\. LLM 提示 (LLM Prompting)

**1.1. 动机 (Motivation)**

*   **语言模型规模的激增 (The language model scaling wars!)**：PPT展示了从ELMo (94M参数) 到Megatron-Turing NLG (530B参数) 等模型参数量的急剧增长趋势。例如，LLaMA (Meta) 拥有65B参数和1.4万亿训练词元，而PaLM (Google) 则高达540B参数和7800亿训练词元。
*   **预训练+微调范式的挑战 (Pretraining + finetuning becomes a paradigm for downstream tasks)**：对于拥有数千亿参数的大型模型，即使是进行少量迭代的微调也成本高昂 (For large models, fine-tuning hundreds of billions of parameters (even a few iterations) is also costly!)。
*   **核心问题 (Core question)**：如何降低这些预训练语言模型 (PLMs) 的“使用成本” (How to reduce the "usage cost" of these PLMs?)？
*   **实际挑战 (Practical challenges)**：大规模模型在共享和提供服务方面成本高昂 (large-scale models are costly to share and serve)。针对不同任务 (Task A, Task B, Task C) 分别微调出一个完整的11B参数模型副本，在存储和计算上都是巨大的负担。
*   **解决方案的曙光 (Solution's dawn)**：语言模型提示学习应运而生 (Language model prompting to the rescue!)。

**1.2. 提示学习 (Prompt Learning)**

*   **GPT-3的启示 (GPT3: Language Models are Few-Shot Learners)**：
    *   GPT-3能够通过利用自然语言提示 (natural-language prompts) 和任务演示 (task demonstrations) 作为上下文，在不更新底层模型参数的情况下，仅用少量样本就能很好地处理各种任务。
    *   GPT-3的巨大模型规模是其成功的重要因素，同时提示和演示的概念也为我们如何更好地使用语言模型提供了新的思路。

**1.3. 什么是提示？ (What is a prompt?)**

*   **定义 (Definition)**：提示是一段插入到输入样本中的文本，目的是将原始任务形式化为一个（掩码）语言建模问题 (A prompt is a piece of text inserted in the input examples, so that the original task can be formulated as a (masked) language modeling problem)。
*   **示例 (Example)**：要对电影评论 "No reason to watch" 进行情感分类，可以在句子后附加一个提示 "It was"，得到 "No reason to watch. It was \_\_\_."。我们自然期望语言模型生成 "terrible" 的概率高于 "great"。
*   **LLM提示的组成 (An LLM prompt example may include)**：
    *   **指令 (Instruction)**：例如，“现在你是情感分析专家。” (Now you are expert of sentiment and emotional analysis.)
    *   **上下文 (Context)**：例如，一段包含多个说话者的对话内容。
    *   **描述 (Descriptions)**：例如，对特定说话人语音特征的描述 (Target speech characteristics: low volume with low variation...)。
    *   **问题 (Questions)**：例如，“请根据上下文和音频特征从&lt;happy, sad, neutral, angry, excited, frustrated>中选择&lt;Speaker\_0: "yeah">的情感标签。” (Please select the emotional label of &lt; Speaker 0: "yeah"> from &lt;happy, sad, neutral, angry, excited, frustrated> based on both the context and audio features.)
    *   **输出类型/格式 (Output types)**：例如，“仅用一个标签回应。” (Respond with one label only:)

**1.4. 为什么选择提示学习？ (Why prompt learning?)**

*   **统一任务格式 (Unified task format)**：提示使得下游任务能够采用与预训练目标相同的格式，并且不需要引入新的参数 (Prompting makes it possible for downstream tasks to take the same format as the pre-training objectives and requires no new parameters)。
    *   只需要设计一个模板 (template)（例如，"It was"）和预期的文本回复（称为标签词 - label words，例如，"great" 代表正面标签，"terrible" 代表负面标签）。
*   **少样本学习的采样效率 (Sampling efficiency for few-shot case)**：对于一个新任务，如果只有少量训练样本（例如十几条），微调预训练模型和新的任务特定参数会很困难，但使用提示则过程要平滑得多。
*   PPT中用图示对比了：
    *   **(a) MLM预训练 (MLM pre-training)**：例如，`[CLS] it's a [MASK] movie in every regard, and [MASK] painful to watch. [SEP]`，模型预测`[MASK]`处的内容。
    *   **(b) 标准微调 (Fine-tuning)**：例如，`[CLS] No reason to watch. [SEP]`，模型通过一个分类头 (CLS head) 输出`label:negative`。
    *   **(c) 基于提示的带演示微调 (Prompt-based fine-tuning with demonstrations)**：例如，输入`[CLS] No reason to watch. It was [MASK]. [SEP]`，并给出正面和负面的演示样例 `A fun ride. It was great. [SEP] The drama discloses nothing. It was terrible. [SEP]`，模型通过MLM头预测`[MASK]`处的内容，并通过标签映射 (label mapping) 将 "great" 映射到 "positive"，将 "terrible" 映射到 "negative"。

**1.5. 提示的形态 (Prompt shape)**

*   **完形填空式提示 (Cloze prompts)**：填充文本字符串中的空白 (fill in the blanks of a textual string)。
    *   例如："Dante was born in \[MASK\]." 模型需要预测 `[MASK]` 的内容（如 "Florence"）。常用于ELMo/BERT等模型。
*   **前缀式提示 (Prefix prompts)**：续写一个字符串前缀 (continue a string prefix)。
    *   例如，在表格到文本 (table-to-text) 任务中，输入 `name Starbucks type coffee shop [SEP] Starbucks serves coffee`，模型需要续写。
    *   **前缀调整 (Prefix-tuning)** 是一种方法，它在输入前添加一系列连续的、任务特定的向量，同时保持语言模型参数冻结。

**1.6. 手动提示工程 (Manual prompt engineering)**

*   **方法 (Method)**：基于人类的直觉和内省创建直观的模板 (Create intuitive templates based on human introspection)。
    *   LAMA数据集为语言模型中的知识探索提供了手动创建的完形填空模板。
    *   Brown等人 (GPT-3论文) 创建了手工制作的前缀提示来处理各种任务，包括问答、翻译和常识推理探测任务。
*   **问题 (Issues with manual prompt engineering)**：
    *   创建和实验这些提示是一门艺术，需要时间和经验 (creating and experimenting with these prompts is an art that takes time and experience)。
    *   即使是经验丰富的提示设计者也可能无法手动发现最优的提示 (even experienced prompt designers may fail to manually discover optimal prompts)。

**1.7. 自动化提示搜索 (Automated prompt searching)**

*   **离散提示 (Discrete prompts / Hard prompts)**：在离散空间中描述的模板，通常对应于自然语言短语。
    *   **基于挖掘的方法 (Mining-based approach)** (例如，Jiang等人的MINE)：从文本中提取主体和客体之间的中间词或短语作为提示。例如，从 "Barack Obama was born in Hawaii" 提取出 "x was born in y"。
    *   **提示复述 (Prompt paraphrasing)** (例如，Jiang等人的方法)：通过将提示往返翻译到另一种语言再翻译回来，以增加词汇多样性，同时保持与原始提示的语义相似性。例如，"x shares a border with y" 可能复述为 "x has a common border with y" 或 "x adjoins y"。
    *   **基于梯度的搜索 (Gradient-based search)** (例如，Wallace等人的方法)：在实际词元上应用基于梯度的搜索，以找到能触发预训练LM生成期望目标预测的短序列。
    *   **提示生成 (Prompt generation)** (例如，Gao等人的方法)：引入seq2seq预训练模型T5到模板搜索过程中。通过指定在模板中插入模板词元的位置，并为T5提供训练样本来解码模板词元。PPT中展示了如何通过T5从带标签的样本（如 "A fun ride. &lt;X> great &lt;Y>"）生成最佳的模板（如 `<S1> A [MASK] one.`）。
*   **连续提示 (Continuous prompts / Soft prompts)**：
    *   与供人类理解的离散提示不同，连续提示不局限于人类可解释的自然语言。它们直接在模型的嵌入空间中进行提示 (Continuous prompts perform prompting directly in the embedding space of the model)。
    *   **移除的约束 (Constraints removed)**：
        *   放宽了模板词元的嵌入必须是自然语言词元嵌入的约束。
        *   移除了模板必须由预训练LM参数化的限制。
    *   **前缀调整 (Prefix tuning)** 是一种代表性方法，它在输入前添加一系列连续的、任务特定的向量，同时保持LM参数冻结。
    *   **源提示调整 (Source Prompt Tuning)** 和 **目标提示调整 (Target Prompt Tuning)**：可以先在一个或多个源任务（或无监督任务）上学习一个通用的源提示，然后用它来初始化特定目标任务的提示。
    *   **基于离散提示初始化的调整 (Tuning initialized with discrete prompts)**：
        *   **Autoprompt**：使用已创建或通过离散提示搜索方法发现的提示来初始化连续提示的搜索。例如，原始输入 "a real joy."，通过Autoprompt可能生成带有触发词 (trigger tokens) 的提示 "a real joy. atmosphere alot dialogue Clone totally \[MASK\]."。
    *   **硬-软提示混合调整 (Hard-soft prompt hybrid tuning)**：通过在嵌入的输入中插入可训练的变量来学习连续提示。
        *   **P-Tuning**：不是使用纯粹可学习的提示模板，而是在硬提示模板中插入一些可调整的嵌入。PPT中展示了从离散提示搜索（通过提示生成器）到P-Tuning（通过提示编码器引入伪提示 Pseudo Prompts，并进行反向传播）的示意图。

**1.8. 对提示的回答 (Answer to prompts)**

*   语言模型被用来填充未完成的槽位以获得最终输出。
*   **回答的形态 (Shape of an answer)**：
    *   **词元 (Tokens)**：预训练LM词汇表中的一个词元，或词汇表的一个子集。
    *   **片段 (Span)**：一个短的多词元片段，通常与完形填空提示一起使用。
    *   **句子 (Sentence)**：一个句子或一个文档，通常与前缀提示一起使用。

**1.9. 挑战 (Challenges)**

*   **超越分类和生成的任务 (Tasks beyond classification and generation)**：目前大多数关于提示学习的工作都围绕文本分类或基于生成的任务展开，许多其他类型的任务尚未充分讨论。
*   **结构化信息的提示 (Prompting with structured information)**：如何在提示学习中表达树、图、表或关系结构是一个主要挑战。
*   **模板和答案的纠缠 (Entanglement of template and answer)**：性能高度依赖于模板和答案映射。如何有效地搜索或学习模板和答案映射的最佳组合仍然是一个挑战。

* * *

### 2\. 思维链提示 (Chain of Thought Prompting - CoT)

*   **核心思想 (Core Idea)**：思维链是一系列通向最终输出的中间自然语言推理步骤。思维链提示旨在引发大型语言模型进行推理 (Chain-of-Thought Prompting Elicits Reasoning in Large Language Models)。
*   **对比标准提示 (Standard Prompting vs. CoT Prompting)**：
    *   **标准提示**：直接给出问题和少量示例（或无示例），模型直接输出答案。对于需要多步推理的问题，容易出错。
    *   **CoT提示**：在提示的示例中，不仅给出最终答案，还展示了得出该答案的详细思考过程（即思维链）。这引导模型在解决新问题时也尝试生成类似的推理步骤。
    *   PPT通过一个数学应用题（罗杰的网球）的例子清晰地展示了CoT如何帮助模型从错误答案（27个苹果）纠正到正确答案（9个苹果），因为它模仿了示例中的逐步推理。
*   **CoT对模型规模的依赖 (Dependency on Model Scale)**：研究表明，CoT提示的效果在非常大的语言模型（如PaLM 540B）上才显著，而对于较小模型可能效果不佳甚至有害。这表明CoT是一种**涌现能力 (emergent ability)**。
*   **自洽性改进思维链推理 (Self-consistency Improves Chain Of Thought Reasoning)**：
    *   从大型模型的解码器中采样一系列不同的推理路径。
    *   每条路径可能对应一个最终答案。
    *   通过对这些路径产生的答案进行投票（例如，选择出现次数最多的答案），可以提高最终答案的鲁棒性和准确性。这种策略基于一个直觉：如果多个不同的推理路径都能得到相同的答案，那么这个答案的可信度就更高。
*   **大型语言模型是零样本推理器 (Large Language Models are Zero-Shot Reasoners)**：
    *   发现仅通过添加一个固定的简单提示，如 **"让我们一步一步地思考" (Let's think step by step)**，就能促使大型模型进行逐步推理并生成结果，而不需要手动注释复杂的CoT示例。
    *   这可以分两步进行：
        1.  **推理提取 (Reasoning Extraction)**：给模型问题和 "Let's think step by step."，让模型生成推理过程。
        2.  **答案提取 (Answer Extraction)**：将问题、"Let's think step by step."以及上一步生成的推理过程拼接起来，再送入模型，让模型直接生成最终答案。
*   **自动思维链提示 (Automatic Chain of Thought Prompting in Large Language Models - Auto-CoT)**：
    *   **动机**：手动创建CoT示例成本高，且大型模型即使在有提示的情况下也可能生成错误的推理链。Auto-CoT旨在自动构建演示。
    *   **步骤 (Steps)**：
        1.  **问题聚类 (Question Clustering)**：使用句子嵌入（如Sentence-BERT）对问题进行表示，然后通过K-means等方法将问题聚类。
        2.  **演示采样 (Demonstration Sampling)**：从每个聚类中选择代表性问题。对于每个选定的问题，拼接上 "Let's think step by step."，然后送入大型模型以生成其推理过程 (rationale)。
        3.  **构建提示 (Constructing the Prompt)**：将生成的（问题，推理过程，答案）对作为演示，与目标测试问题一起构建最终的提示，引导模型为测试问题生成推理过程和答案。
*   **从易到难的提示 (Least-to-Most Prompting Enables Complex Reasoning)**：
    *   **核心思想**：将复杂问题分解为一系列更简单的子问题，然后逐步解决这些子问题。
    *   **阶段 (Stages)**：
        1.  **问题分解 (Decompose Question into Subquestions)**：首先让语言模型将原始复杂问题分解为若干子问题。
        2.  **顺序解决子问题 (Sequentially Solve Subquestions)**：然后依次让语言模型解决这些子问题，后一个子问题的解决可以利用前面子问题的答案。
    *   PPT用一个水滑梯问题为例，展示了如何先分解出“每次行程需要多长时间？”这个子问题，解决后再解决原始问题“在关闭前她能滑多少次？”。
*   **思维树、思维图等 (Tree of Thoughts, Graph of Thoughts......)**：
    *   CoT是线性推理，而更高级的推理可能需要探索多个推理路径，形成树状或图状结构。Tree of Thoughts (ToT) 允许模型在每个思考步骤中探索多个分支，并使用启发式方法评估这些分支，从而进行更复杂的规划和问题解决。

**1.10. 未来发展 (Future developments of Prompt Learning)**

*   **商业领域 (Commercial field)**：广泛应用于自然语言处理、推荐系统等，通过提示学习更有效地处理用户反馈，提供个性化服务。
*   **社会领域 (Social field)**：用于情感分析、舆情监控等，更准确地分析公众情绪和观点，为政策制定和社会管理提供参考。
*   **科学领域 (Scientific field)**：广泛应用于各种预测和分类任务。
    *   **医学领域 (Medical field)**：预测疾病发展趋势和治疗效果，辅助医生制定更科学的治疗方案。
    *   **环境科学领域 (Environmental science field)**：预测气候变化、自然灾害等复杂系统的行为，为环境保护和灾害防治提供决策支持。
*   **总结 (In short)**：提示学习作为一种新的机器学习方法，利用上下文提示来引导模型进行预测，具有许多优势和广泛的应用前景。尽管目前面临一些挑战和局限性，但随着技术的不断发展和完善，提示学习有望成为未来机器学习领域的重要方向之一。

* * *

### 3\. 对齐 (Alignment)

*   **强化学习回顾 (Reinforcement Learning Review)**：
    
    *   **基本概念 (Basic Concepts)**：智能体 (Agent) 在环境 (Environment) 中，通过观察状态 (S\_t)，执行动作 (a\_t)，环境会反馈奖励 (r\_t) 和下一个状态 (S\_t+1)。
    *   **RL的特点 (Characteristics of RL)**：没有监督者，只有奖励信号；反馈是延迟的；时间序列性重要（数据非独立同分布）；智能体的动作影响后续接收的数据；目标是最大化累积奖励。
    *   **RL的例子 (Examples of RL)**：直升机特技飞行、人形机器人行走、玩Atari游戏、击败围棋世界冠军。
*   **用于LLM对齐的强化学习 (RL for LLM Alignment)**：
    
    *   **挑战 (Challenge)**：LLM可能产生有害的、有偏见的或不相关的内容。
    *   **目标 (Goal)**：确保LLM产生的输出与人类的价值观和意图对齐 (Ensure LLMs produce outputs aligned with human values and intentions)。
    *   **解决方案 (Solution)**：使用强化学习 (RL) 根据反馈来微调模型。
*   **RLHF回顾 (RLHF Review)**：再次强调了之前讲过的三步RLHF流程：
    
    1.  收集演示数据，训练有监督策略 (SFT)。
    2.  收集比较数据，训练奖励模型 (RM)。
    3.  使用强化学习（如PPO）针对奖励模型优化策略。
*   **RL中的策略 (Policy in RL)**：
    
    *   策略是智能体的行为方式，是从状态到动作的映射 (f:s\_trightarrowa\_t)。
    *   **确定性策略 (Deterministic policy)**：a\=Pi(s)。
    *   **随机性策略 (Stochastic policy)**：pi(a∣s)\=P\[A\_t\=a∣S\_t\=s\]。
*   **最优策略 (The Optimal Policy)**：
    
    *   目标是找到最大化期望累积奖励（通常是折扣累积奖励）的最优策略 pi∗：pi∗\=argmax\_piE\[sum\_tge0gammatr\_t∣pi\]。
*   **策略梯度 (Policy Gradient)**：
    
    *   **Q-Learning的问题 (Problem with Q-Learning)**：Q函数可能非常复杂，难以学习每个(状态, 动作)对的精确值。
    *   **直接学习策略 (Directly learning policy)**：策略本身可能比Q函数简单得多。可以直接学习一个参数化的策略 Pi\=pi\_theta,thetainmathbbRm。
    *   **目标 (Goal)**：找到最优参数 $\\\\theta^\\\* = \\\\arg\\\\max\\\_{\\\\theta} J(\\\\theta)$，其中 J(theta)\=E\[sum\_tge0gammatr\_t∣pi\_theta\]。
    *   **方法 (Method)**：在策略参数上进行梯度上升 (Gradient ascent on policy parameters)。
    *   LLM对齐中将要介绍的RL方法基本都源于策略梯度。
*   **三种用于LLM对齐的RL方法 (Three RL Methods for LLM Alignment)**：
    
    *   **PPO (Proximal Policy Optimization)**：一种通过优化策略来训练智能体的RL算法。
        *   **核心思想 (Key Idea)**：在训练过程中平衡样本效率和稳定性。
        *   **核心机制 (Core Mechanism)**：使用一个裁剪的目标函数 (clipped objective) 来防止过大的策略更新，确保学习过程更平滑。
        *   **裁剪目标函数 (Clipped Objective)**：LCLIP(theta)\=hatE∗t\[min(r\_t(theta)hatA∗t,textclip(r\_t(theta),1−epsilon,1+epsilon)hatA∗t)\]，其中 r\_t(theta)\=fracpi∗theta(a\_t∣s\_t)pi∗theta∗old(a\_t∣s\_t) 是概率比率，hatA\_t 是优势函数 (Advantage Function) A(s,a)\=Q(s,a)−V(s)。
        *   **PPO算法细节 (Algorithm Details)**：迭代地收集轨迹、计算梯度、更新策略参数。
        *   **PPO在LLM对齐中的应用 (PPO in LLM Alignment)**：通常在SFT和奖励建模之后，使用PPO最大化奖励模型给出的奖励。
        *   **优点 (Strength)**：在RLHF中被证明有效。
        *   **缺点 (Weakness)**：计算成本高且实现复杂。
    *   **DPO (Direct Preference Optimization)**：一种无需复杂RL即可使用人类偏好对齐LLM的方法。
        *   **核心思想 (Key Idea)**：直接使用偏好数据优化策略，而不是学习一个奖励函数。
        *   **与PPO对比 (DPO vs PPO)**：DPO跳过了奖励建模，直接使用成对的偏好数据优化策略，因此更高效且可扩展。
        *   **工作原理 (How It Works)**：收集模型输出的成对比较（偏好的 vs. 不偏好的），然后使用基于偏好的监督损失函数直接优化LLM。
        *   **优点 (Strength)**：简单（无需奖励建模）、高效（训练快，计算资源少）、可扩展性好。
        *   **缺点 (Weakness)**：依赖高质量的偏好数据、可能难以泛化到未见过场景、人类偏好可能引入偏见。
    *   **GRPO (Generalized Reward Preference Optimization / Group Relative Policy Optimization)**：一种专门设计用于增强LLM推理能力的混合RL算法。 (PPT中作者标注为Shao et al., 2025，但根据上下文及常见命名，Group Relative Policy Optimization更贴切，也与PPO对比的图示相符)
        *   **设计目标 (Design Goal)**：继承PPO的鲁棒性，同时避免昂贵的计算。
        *   **与PPO对比 (GRPO vs PPO)**：GRPO没有独立的价值网络 (value network) 作为Critic。它对同一问题或状态从旧策略中采样多个输出，并将这些输出的平均奖励作为基线 (baseline)。
        *   **优点 (Strength)**：高效（消除了价值网络，降低了内存和计算成本）、稳定（基于组的优势估计和KL散度正则化使训练更稳定）、可扩展性好（更适合DeepSeek-V2和V3等大规模模型）。
        *   **算法细节 (Algorithm Details)**：为每个问题采样多个输出，使用奖励模型评分，在组内归一化奖励，使用相对优势和KL正则化优化策略，并迭代更新。
*   **PPO, DPO, GRPO的比较总结 (Comparison and Summary)**：PPT末尾预留了这三种方法的比较总结空间。
    

* * *

这份PPT深入探讨了如何通过提示工程与大型语言模型进行有效交互，以及如何通过强化学习等对齐技术使LLM的行为更符合人类的期望和价值观。内容涵盖了从基本概念到前沿算法的多个层面。

信息来源和相关内容

---

![PDF 图标](https://drive-thirdparty.googleusercontent.com/32/type/application/pdf)

L9\_LLMAgents

PDF

非常棒，请你接着讲解该ppt

---

显示思路

好的，我们来详细讲解这份关于大型语言模型智能体 (LLM Agents) 的PPT。

### 本讲座大纲 (Outline)

1.  **LLM智能体介绍 (LLM agents introduction)**
2.  **推理 (Reasoning)**
3.  **工具学习 (Tool learning)**
4.  **知识整合-RAG (Knowledge incorporation-RAG)**

* * *

### 1\. LLM智能体介绍 (LLM Agents Introduction)

**1.1. 语言模型的局限性 (Language models are powerful, but they still suffer from)**

尽管大型语言模型非常强大，但它们仍然存在一些问题：

*   **幻觉 (Hallucination)**：可能生成不真实或虚构的信息。
*   **知识不准确 (Inaccuracy of knowledge)**：知识库可能过时或包含错误信息。
*   **缺乏可解释性 (Lack of interpretability)**：难以理解模型为何做出特定决策。
*   **不一致性 (Inconsistency)**：对于相似的输入可能产生矛盾的输出。
*   **可扩展性有限 (Limited scalability)**：在处理超大规模或非常专业的知识时可能受限。
*   **能力受限 (Restricted capabilities)**：无法直接与外部世界交互或执行物理操作。

**核心问题 (Core Question)**：LLM能否利用内部推理/规划和外部工具/知识，不仅扩展自身能力，还能使我们的NLP系统更鲁棒、可扩展和可解释？ (Can LLMs utilize internal reasoning/planning and external tools/knowledge to not only expand their capacities but also to make our NLP systems more robust, scalable, and interpretable?)

**1.2. 智能体示例 (Agent examples)**

PPT中展示了多种智能体的图片，例如：

*   自动驾驶汽车 (Autonomous car)
*   AlphaGo (下围棋的AI)
*   ChatGPT (聊天机器人)
*   OpenAI Five (玩Dota 2的AI)
*   Amazon Alexa (语音助手)
*   能玩魔方的机器人手 (Robotic hand solving Rubik's cube)

这些例子说明了智能体在不同领域的应用。

**1.3. 什么是智能体？ (What are agents?)**

*   **定义 (Definition)**：一个与某种“环境 (environment)”交互的“智能 (intelligent)”系统。
    *   智能体通过执行**动作 (Action)** 来影响环境，并通过**观察 (Observation)** 来感知环境。
*   **环境类型 (Types of Environments)**：
    *   **物理环境 (Physical environments)**：机器人、自动驾驶汽车等。
    *   **数字环境 (Digital environments)**：用于Atari游戏的DQN、Siri、AlphaGo等。
    *   **人类作为环境 (Humans as environments)**：聊天机器人。
*   “智能体”的定义取决于如何定义“智能”和“环境”，这些定义会随着时间而演变。
*   **思考题 (Exercise question)**：你会如何定义“智能”？ (how would you define "intelligent"?)

**1.4. 什么是LLM智能体？ (What are LLM agents?)**

*   LLM智能体是一种特殊的智能体，它使用大型语言模型作为其核心组件，通常负责思考、规划或理解。
*   智能体与环境的交互通常也通过语言 (L) 进行，即动作 (Action) 和观察 (Observation) 都是语言形式。
*   **智能体的层级 (Levels of Agents)**：
    *   **Level 1: 文本智能体 (Text agent)**：使用文本进行交互，但不一定依赖LLM进行复杂推理。
        *   例子 (Examples)：ELIZA (早期的聊天机器人), LSTM-DQN (用于游戏的基于LSTM的深度Q网络)。
    *   **Level 2: LLM智能体 (LLM agent)**：使用LLM来行动 (Uses LLM to act)。
        *   例子 (Examples)：SayCan (让机器人在物理世界中执行指令), Language Planner。
    *   **Level 3: 推理智能体 (Reasoning agent)**：使用LLM进行推理以指导行动 (Uses LLM to reason to act)。
        *   例子 (Examples)：ReAct, AutoGPT。
    *   **本讲座的重点 (The key focus of the field and the talk)** 主要集中在Level 2和Level 3的LLM智能体。

**1.5. LLM智能体示例：SayCan (Example: SayCan)**

*   SayCan (由PaLM驱动的助手机器人) 可以执行复杂的任务，它利用**思维链提示 (chain of thought prompting)** 来解读指令，并评估单个技能对完成高级请求的贡献程度。
*   **工作流程 (How PaLM-SayCan Works)**：
    1.  用户给出指令，例如：“我刚锻炼完，能给我拿点喝的和零食来恢复一下吗？” ("I just worked out. Can you bring me a drink and a snack to recover?")
    2.  PaLM-SayCan解读任务：“用户需要一杯饮料和一个零食。我会拿一个水瓶和一个苹果。” ("The user has asked for a drink and a snack. I will bring a water bottle and an apple.")
    3.  机器人执行任务（例如，从厨房台子上拿起水瓶和苹果）。
    4.  系统可以展示PaLM-SayCan背后一步一步的解决方案。

* * *

### 2\. 推理 (Reasoning)

*   **回顾思维链 (LLM reasoning: recall of CoT)**：
    *   再次强调了标准提示 (Standard Prompting) 和思维链提示 (Chain-of-Thought Prompting) 的区别。思维链通过展示中间推理步骤，帮助LLM解决需要多步思考的问题。
    *   PPT中展示了MathCoder的架构图，说明了如何通过监督微调 (Supervised Fine-Tuning, SFT) 和从GPT-4等模型进行推理插值 (Problem Interpolation) 来训练数学问题解决模型。
*   **LLM推理：解决数学问题 (LLM reasoning: solving math problems)**：
    *   **数据集 (Datasets)**：
        *   **GSM8K**：包含8500个高质量、语言多样的小学数学应用题。
        *   **MATH**：包含12500个具有挑战性的竞赛数学问题。
    *   PPT展示了GSM8K和MATH数据集中问题的具体例子及其解题过程，这些解题过程通常包含详细的思考步骤，适合用于训练或评估模型的推理能力。

* * *

### 3\. 工具学习 (Tool Learning)

**3.1. 人类+工具使用的动机 (Human + tool use: motivations)**

*   人类自身有局限性：时间有限、记忆有限、会感到疲倦、有情绪波动 (As humans, we have limited time and memory, feel tired, and have emotions)。
*   **人类+工具的优势 (Human + tool use advantages)**：
    *   增强的可扩展性 (Enhanced scalability)
    *   改进的一致性 (Improved consistency)
    *   更强的可解释性 (Greater interpretability)
    *   更高的能力和生产力 (Higher capacity and productivity)
    *   PPT配图：驾驶辅助系统、汽车工厂的自动化机器人。

**3.2. LLM+工具使用的动机 (LLMs + tool use: motivations)**

*   LLM也存在类似人类的局限性 (Just like humans, LLMs suffer from the similar limitations)。
*   **LLM+工具的优势 (LLMs + tool use advantages)**：同样可以带来可扩展性、一致性、可解释性、能力和生产力的提升。
*   PPT配图：各种编程语言/库 (SQL, pandas, scikit-learn, PyTorch, Hugging Face, matplotlib) 和Google搜索框，象征LLM可以调用的工具。

**3.3. LLM+工具使用：可执行语言接地的视角 (LLMs + tool use in perspective of executable language grounding)**

*   **核心思想 (Core Idea)**：将语言模型的能力与可执行动作联系起来 (Ground language models into executable actions)。
*   **具体做法 (Method)**：将自然语言指令映射到可以在各种环境中执行的代码或动作 (Mapping natural language instructions into code or actions executable within various environments)。
    *   环境示例：数据库 (databases)、Web应用程序 (web applications)、机器人物理世界 (robotic physical world)。
*   **组合 (Combination)**：LM (负责规划和推理 - planning and reasoning) + 动作 (actions)。
*   **应用示例 (Application Examples)**：
    *   **数据分析 (Data analysis)**：用户用自然语言提问“创建一个按星期显示转化率的条形图”，LLM生成代码调用数据分析工具完成。
    *   **Web/App交互 (Web/Apps)**：如图中展示的法语学习App。
    *   **机器人物理世界 (Robotic physical world)**：用户说“把积木堆在空碗上”，LLM生成策略代码 (Policy Code)，调用感知API (Perception APIs)（如`detect_objects`）和控制API (Control APIs)（如`pick_place`）来完成任务。
*   **输入 (Inputs)**：
    *   语言 (Language)：用户的问题/请求。
    *   工具包 (Toolkit)：代码、搜索引擎API、自定义函数、专家模型等。
    *   环境 (Environment)：数据库、IDE、网页/应用、视觉和机器人物理世界等。
*   **输出 (Outputs)**：
    *   可在相应环境中执行的、接地的推理代码/动作序列。
    *   决定选择什么工具、何时以及如何使用选定的工具。

**3.4. LLM+工具使用在可执行语言接地中的示例 (Example of LLMs + tool use in executable language grounding)**

*   **BINDER框架 (BINDER framework)** (Cheng et al., ICLR 2023)：一个无需训练的神经符号框架，用于将任务输入映射到一个可执行的BINDER程序。
    *   该程序允许将对GPT-3 Codex的API调用绑定到SQL/Python中。
    *   然后通过SQL/Python解释器 + GPT-3 Codex执行该程序以得出答案。
*   **示例场景 (Example Scenario)**：
    *   给定一个包含衬衫信息（品牌、产地、销量、价格、细节描述）的表格和非结构化的知识（如100%纯棉等）。
    *   问题：“哪款在北美生产且不含化学物质的衬衫销量最好？” (Which is the best-selling shirt made in North America and with no chemicals?)
    *   **对比不同方法**：
        *   **LLM + 无工具 (End-to-End GPT-3 Codex)**：直接将输入喂给LLM。
            *   **优点**：通用 (General)。
            *   **缺点**：不可解释且不稳定 (Uninterpretable & not robust)，可扩展性和鲁棒性差。即使使用思维链 (Chain-of-Thought)，虽然有所改进，但答案的蕴含关系仍然不可靠。
        *   **LLM + 代码 (Semantic Parsing/Code Generation)**：LLM生成SQL查询语句。
            *   **优点**：可解释、可扩展、鲁棒。
            *   **缺点**：生成的SQL可能因无法处理“无化学物质”、“北美制造”等模糊概念而出错（例如，直接在SQL中写 `Shirt NOT LIKE '%chemicals%'` 可能不准确），能力有限 (Limited PL grammar coverage)。
        *   **LLM + 代码和NLP专家函数API (BINDER)**：LLM生成混合了SQL和对Codex的特定任务API调用（如`codex("No chemicals?"; Shirt)='yes'`）的BINDER程序。
            *   **优点**：可解释、鲁棒、覆盖范围广、能力强 (Interpretable, robust, wide coverage, capable)。能够正确处理模糊概念并结合结构化查询。
    *   **BINDER工作流程**：
        1.  BINDER首先提示Codex将问题解析为一个BINDER程序，Codex需要决定：
            *   输入中哪些部分可以转换成目标编程语言（如图中的灰色SQL子句）。
            *   需要哪些相应的任务API调用（如图中的蓝色`codex(...)`子句）来提示Codex解决其他部分。
            *   在BINDER程序中何处插入这些API调用。
        2.  接下来，BINDER再次提示Codex为这些任务API调用生成答案（基于生成的任务提示）。
        3.  然后将生成的结果整合回编程语言表达式中。
        4.  最后执行该表达式以得出最终答案。
*   **程序辅助语言模型 (Program-Aided Language models, PAL)** (Gao et al.)：
    *   使用LLM阅读自然语言问题并生成程序作为中间推理步骤，但将解题步骤交由运行时（如Python解释器）执行。
    *   **对比思维链 (CoT vs. PAL)**：
        *   **CoT**：生成自由形式文本的中间推理步骤。
        *   **PAL**：生成包含代码的中间步骤，将运行推理步骤的角色从语言模型转移到代码解释器。
    *   PPT展示了数学应用题的例子，PAL生成的Python代码能更精确地进行计算。
    *   **思想程序提示 (Program of Thoughts Prompting, POT)** (Chen et al.)：与PAL类似，旨在将计算从推理中解耦，用于数值推理任务。PPT通过斐波那契数列和利率计算的例子对比了CoT和POT。
*   **LLM + API（搜索/浏览器）用于信息收集 (LLM + APIs to search/browser for gathering information)**：
    *   **WebGPT** (Nakano et al., OpenAI)：通过微调GPT-3，使其能够使用基于文本的网页浏览环境来回答长篇问题，允许模型搜索和导航网页。
    *   定义了一系列浏览器操作命令，如 `Search <query>`, `Clicked on link <link ID>`, `Find in page: <text>`, `Quote: <text>` 等。
    *   实验结果表明，WebGPT在回答问题的真实性和信息量上均优于直接使用GPT-3。
*   **LLM + 网页/应用或个性化函数 (LLM + webs/apps or personalized functions)**：
    *   **ReAct框架 (ReAct: Synergizing Reasoning and Acting in Language Models)** (Yao et al., Google Brain)：
        *   促使LLM以交错的方式生成推理轨迹 (reasoning traces) 和任务特定动作 (task-specific actions)。
        *   这种方式允许两者之间产生更大的协同作用：推理轨迹帮助模型引导、跟踪和更新行动计划，并处理异常；而动作使其能够与外部源（如知识库或环境）交互并收集额外信息。
        *   PPT展示了ReAct在HotpotQA（问答任务）和AlfWorld（基于文本的虚拟环境交互任务）中的应用示例。在AlfWorld中，模型会生成如 "Think: 我首先需要找到一个胡椒瓶...它更可能出现在柜子(1-6)或台面(1-3)上。" 这样的思考步骤，然后执行 "Act: 前往柜子1" 这样的动作。
*   **LLM + 专家模型API (LLM + APIs to expert models)**：
    *   **HuggingGPT** (Shen et al., Microsoft)：一个由LLM驱动的智能体，利用LLM（如ChatGPT）连接机器学习社区（如Hugging Face）中的各种AI模型来解决AI任务。
    *   **工作流程 (Workflow)**：
        1.  **任务规划 (Task Planning)**：当收到用户请求时，使用ChatGPT进行任务规划。
        2.  **模型选择 (Model Selection)**：根据Hugging Face中可用的模型功能描述选择合适的模型。
        3.  **任务执行 (Task Execution)**：使用选定的AI模型执行每个子任务。
        4.  **响应生成 (Response Generation)**：根据执行结果总结响应。
    *   PPT示例：用户要求描述图片并计算图片中物体的数量。HuggingGPT可能会规划出图像分类、物体检测和图像描述等子任务，并调用Hugging Face上相应的模型来执行，最后综合结果生成答案。
*   **LLM + 代码、机械臂、专家模型 (LLM + code, robotic arm, expert models: Code as Policies)**：
    *   使用LLM根据自然语言命令编写机器人策略代码。
    *   结合感知API和控制API，使机器人能够理解指令并与物理世界交互。

* * *

### 4\. 工具学习的方法 (Methods for tool learning)

*   **LLM提示进行工具使用 (LLM prompting for tool use)**：
    *   重申了PAL的思想，即生成自然语言和编程语言交织的中间步骤。
    *   展示了BINDER如何通过精心设计的提示（包含任务/工具描述、环境描述、用户问题）引导GPT-3 Codex生成包含API调用的SQL代码。
*   **LLM+工具用于问答和决策：ReAct (LLM + tool use for QA and decision making: ReAct)**：
    *   再次强调ReAct通过交错生成推理轨迹和动作，实现“推理以行动 (reason to act)”和“行动以推理 (act to reason)”的协同。
*   **LLM微调/预训练进行工具使用 (LLM finetuning/pretraining for tool use)**：
    *   **TALM (Tool Augmented Language Models)** (Parisi et al.)：
        *   结合了纯文本方法来增强语言模型使用不可微工具的能力。
        *   采用迭代的“自我博弈 (self-play)”技术，从少量工具使用演示开始引导性能提升。
        *   **算法流程**：迭代地微调LM，然后使用微调后的LM为任务输入采样工具查询、调用外部工具获取结果，如果结果正确则将（任务输入、工具查询、工具结果、任务输出）加入工具使用数据集，用于下一轮微调。
    *   **Toolformer** (Schaison et al., Meta AI)：
        *   训练模型来决定调用哪些API、何时调用、传递什么参数以及如何最好地将结果整合到未来的词元预测中。
        *   这是以**自监督 (self-supervised)** 的方式完成的，每个API只需要少量的演示。
        *   **数据集构建流程**：
            
            1.  从LM数据集中采样API调用。
            2.  执行API调用。
            3.  过滤掉那些没有减少后续词元预测损失的API调用。
            4.  将所有剩余的有效API调用（及其结果）插入到原始文本中，形成新的训练数据。
            
            *   PPT示例中，模型学会了在文本中插入 `[QA("Who is the publisher of The New England Journal of Medicine?") → Massachusetts Medical Society]` 这样的API调用及结果。

* * *

### 5\. 其他近期相关工作 (Other recent works / related work)

*   **LLM作为工具制造者 (LLM as a tool maker)** (Cai et al., Google DeepMind)：
    *   在有大量问题解决请求的情况下，直接使用强大的LLM解决所有实例可能成本高昂。而轻量级模型成本效益高，但通常难以处理复杂任务。
    *   **思路 (Idea)**：使用少量实例，让一个强大的模型（工具制造者 - Tool Maker）生成可重用的工具（实现为Python函数）。然后，一个成本效益高的工具使用者模型 (Tool User) 可以重用这些工具来解决后续请求中的相似实例。
*   **LLM+P：赋予LLM最优规划能力 (LLM+P: Empowering Large Language Models with Optimal Planning Proficiency)** (Liu et al.)：
    *   LLM在解决长远机器人规划问题上并不可靠。相比之下，经典规划器一旦问题以格式化方式给出，就能使用高效的搜索算法快速识别正确甚至最优的计划。
    *   **LLM+P方法**：首先将语言描述转换为规划领域定义语言 (Planning Domain Definition Language, PDDL) 文件，然后利用经典规划器快速找到解决方案，最后将找到的解决方案翻译回自然语言。
*   **代码生成相关工作 (Code generation related work)**：
    *   **CODET (CODE GENERATION WITH GENERATED TESTS)** (Chen et al., Microsoft)：代码解决方案和测试用例都由预训练语言模型生成。然后通过双重执行一致性 (dual execution agreement) 来选择最佳代码解决方案。
    *   **SELF-DEBUGGING (TEACHING LARGE LANGUAGE MODELS TO SELFDEBUG)** (Chen et al., Google DeepMind)：通过少量演示教会大型语言模型调试其预测的程序。模型能够通过调查执行结果并用自然语言解释生成的代码来识别其错误。

* * *

### 6\. 整合外部知识 - RAG (Incorporating external knowledge - RAG)

*   **RAG (Retrieval Augmented Generation)**：检索增强生成。
*   **核心组件 (Core Components)**：
    *   **向量数据库 (Vector Database)**：存储知识的嵌入表示。
        *   **嵌入 (Embedding)**：将文本块转换为向量。
        *   **索引 (Indexing)**：构建向量索引以便高效检索。
        *   **查询/检索 (Querying (Retrieve))**：根据用户输入查询相关的文本块。
    *   **后处理 (Post-process)**：对检索到的信息进行处理。
    *   **生成器 (Generator)**：通常是一个LLM（如GPT, DeepSeek），结合原始查询和检索到的上下文信息来生成最终答案。
*   **工作流程 (Workflow)**：
    1.  用户输入问题 (Input Query)。
    2.  将问题嵌入并在向量数据库中检索相关文档块 (Relevant Documents / Chunks)。
    3.  将检索到的文档块与原始问题（通常还有特定提示）结合起来 (Combine Context and Prompts)。
    4.  将组合后的内容输入给LLM生成器，生成最终答案 (Generation -> Answer)。
*   **对比 (Comparison)**：
    *   **无RAG (without RAG)**：LLM可能因为知识截止或缺乏特定信息而无法回答或回答不准确（例如，关于OpenAI CEO Sam Altman近期事件的问题）。
    *   **有RAG (with RAG)**：通过从外部知识源检索最新或相关信息，LLM可以生成更准确、更全面的答案。PPT中示例显示，有RAG的模型能够就Sam Altman事件给出更深入的评论。

* * *

### 7\. 挑战与未来工作 (Challenges and future work)

*   **复杂性 (Complexity)**：如何处理更复杂的领域专业工具或未见过的工具？
*   **交互性 (Interactivity)**：如何超越单轮交互？
*   **评估 (Evaluation)**：如何评估存在多种可能解决方案的任务？如何进行实时交互式评估？
*   **效率 (Efficiency)**：能否使用更小的模型实现类似功能？
*   **可靠性 (Reliability)**：如何让智能体知道何时应该拒绝回答 (know when to abstain)、了解自身能力边界 (know its capacity)、以及如何有效地记忆和查询工具？
*   **其他 (Others)**：
    *   更好的工具API设计或工具制造 (Better tool API design/tool making)？
    *   个性化 (Personalization)？

* * *

这份PPT系统地介绍了LLM智能体的概念、动机、关键技术（如推理、工具学习、知识整合）以及面临的挑战。通过赋予LLM使用工具和外部知识的能力，可以显著扩展其应用范围并提高其解决复杂问题的能力。

信息来源和相关内容

---

![PDF 图标](https://drive-thirdparty.googleusercontent.com/32/type/application/pdf)

L10\_Effici...ing of LLM

PDF

非常棒，请你接着讲解该ppt

---

显示思路

好的，我们来开始讲解这份关于“大型语言模型的高效训练 (Efficient Training of LLMs)”的PPT。

### 本讲座大纲 (Outline)

1.  **背景 (Background)**
2.  **模型并行 (Model Parallelism)**
3.  **数据并行 (Data Parallelism)**
4.  **参数高效微调 (Parameter-Efficient Fine-Tuning, PEFT)**

* * *

### 1\. 背景 (Background)

**1.1. 模型权重 (Model Weights)**

*   **定义 (Definition)**：权重是模型在训练过程中学习到的参数 (Weights are the parameters of the model that are learned during training)。
*   **更新机制 (Update Mechanism)**：优化器根据梯度调整权重，以最小化损失函数 (They are adjusted by the optimizer based on the gradients to minimize the loss function)。
*   **Transformer中的权重 (Weights in Transformers)**：在Transformer模型中，权重包括自注意力 (self-attention) 机制中的参数、逐位置前馈网络 (position-wise feed-forward networks) 中的参数以及层归一化 (layer normalization) 中的参数。
*   PPT中的图示展示了Transformer中自注意力机制的计算流程，其中 WqT​,WkT​,WvT​ 就是模型权重矩阵，它们将输入 (n个词元，每个词元维度为d) 转换为查询 (Q)、键 (K) 和值 (V) 矩阵。

**1.2. 内存都去哪儿了？ (Where did all the memory go?)**

*   **模型权重的存储 (Storage of Model Weights)**：模型权重在fp16 (半精度浮点数)下占用2字节 (16位)，在fp32 (单精度浮点数)下占用4字节 (32位)。
*   **示例 (Example)**：一个拥有15亿 (1.5B) 参数的GPT-2模型，在16位精度下，其模型权重就需要3GB的内存 (Consider a 1.5B parameter GPT-2 model requires 3GB of memory for its model weights in 16-bit precision)。
*   **内存不足问题 (Out-of-Memory, OOM)**：然而，这样一个模型通常无法在具有32GB显存的单个GPU上使用朴素的TensorFlow或PyTorch进行训练，因为会出现内存不足的问题。这引出了一个问题：内存都去哪儿了？

**1.3. 混合精度训练 (Mixed Precision Training)**

*   **目的 (Purpose)**：为了能够利用GPU的高吞吐量张量核心单元 (tensor core units)，模型权重通常以fp16格式存储。
*   **过程 (Process)**：在混合精度训练期间，前向传播 (forward propagation) 和反向传播 (backward propagation) 都使用fp16进行计算。
*   **fp32的保留 (Retention of fp32)**：然而，为了在反向传播结束时有效地计算和应用更新，会保留一份fp32精度的参数副本以及所有其他的优化器状态 (optimizer states)。

**1.4. 优化器状态 (Optimizer States)**

*   **定义 (Definition)**：优化器状态是优化器在训练过程中存储的内部数据 (Optimizer state refers to the internal data stored by the optimizer during the training process)。这些数据用于高效地更新模型的权重。
*   **Adam优化器示例 (Adam Optimizer Example)**：Adam优化器需要为每个参数维护一个或两个优化器状态（例如，梯度的时间平均动量 (time averaged momentum) 和方差 (variance)）。
*   **内存消耗 (Memory Consumption)**：随着模型规模的增长，优化器状态所消耗的内存可能成为内存消耗的主导因素。

**1.5. 内存消耗计算 (Memory Consumption Calculation)**

*   假设一个模型有 Ψ 个参数。
*   在fp16训练期间：
    *   模型权重 (model weights) 需要 2Ψ 字节。
    *   对应的梯度 (gradients) 需要 2Ψ 字节。
*   Adam优化器的状态 (fp32精度)：
    *   参数的fp32副本 (fp32 copy of parameters)：4Ψ 字节。
    *   动量的fp32副本 (fp32 copy of momentum)：4Ψ 字节。
    *   方差的fp32副本 (fp32 copy of variance)：4Ψ 字节。
    *   总计优化器状态内存：4Ψ+4Ψ+4Ψ\=12Ψ 字节。
*   **总内存消耗 (Total Consumption)**：2Ψ(weights)+2Ψ(gradients)+12Ψ(optimizer states)\=16Ψ 字节。

**1.6. 内存都去哪儿了？（续） (Where did all the memory go? (continued))**

*   对于1.5B参数的GPT-2模型，在16位精度训练下，模型权重、梯度和优化器状态总共需要 16×1.5B\=24GB 的内存。
*   尽管24GB小于32GB的显存，但仍然会出现OOM问题。PPT在此处设问：为什么？ (why OOM?) 引出下一个内存消耗大户。

**1.7. 激活值 (Activation / Intermediate Results)**

*   **定义 (Definition)**：为了执行反向传播，在前向传播过程中存储的中间结果 (The intermediate results stored from forward pass in order to perform backward pass)。
*   PPT通过高亮Transformer自注意力计算图中的不同部分（Q, K, QKT, A, Z）来示意这些都是需要存储的激活值。
*   **激活内存的规模 (Scale of Activation Memory)**：基于Transformer的模型的激活内存与以下因素成正比： `Transformer层数 × 隐藏层维度 × 序列长度 × 批量大小` `(transformer layers × hidden dimensions × sequence length × batch size)`
*   **具体例子 (Concrete Example)**：1.5B参数的GPT-2模型，使用1K的序列长度和32的批量大小进行训练，大约需要60GB的激活内存。

这解释了为什么即使模型权重、梯度和优化器状态加起来小于GPU显存，仍然可能发生OOM，因为激活值占据了大量内存。

* * *

### 2\. 模型并行 (Model Parallelism, MP)

**2.1. 朴素模型并行 (Naïve Model Parallelism / Vertical Partition)**

*   **思想 (Idea)**：将模型的不同层组垂直地分布到多个GPU上 (Vertical partition, spreads groups of model layers across multiple GPUs)。
*   **示例 (Example)**：将编码器 (Encoders) 放置在GPU 0上，解码器 (Decoders) 放置在GPU 1上。
    *   当数据在编码器内部的层之间传递时，这与单GPU模型相同。
    *   但是，当数据需要从编码器传递到解码器时（即从GPU 0到GPU 1），会引入通信开销 (communication overhead)。
*   **优点 (Advantage)**：
    *   能够将非常大的模型装载到有限的硬件上 (fit very large models onto limited hardware)。
*   **缺点 (Disadvantage)**：
    *   **GPU空闲（气泡）(GPU idle (Bubbles))**：由于流水线的依赖关系，某些GPU在等待其他GPU完成计算时会处于空闲状态。PPT用一个时间流图展示了4个GPU（GPU0-GPU3）在顺序执行前向传播 (F0-F3) 和反向传播 (B3-B0) 时的空闲间隙。

**2.2. GPipe**

*   **思想 (Idea)**：将一个批次的训练样本分割成更小的**微批次 (micro-batches)**，然后在不同的计算单元（Cells，每个Cell可以放置在一个GPU上）上流水线式地执行这些微批次。
*   **优点 (Advantage)**：
    *   减少了气泡，更充分地利用了GPU (Less bubble, make the most use of GPUs)。
*   **缺点 (Disadvantage)**：
    *   需要通过实验来找到能够最高效利用GPU的微批次大小 (Need to experiment to find the size of micro-batch that leads to the highest efficient utilization of the GPUs)。
*   PPT中的图示展示了4个GPU处理4个微批次（F\_i,j 表示第i个GPU处理第j个微批次的前向传播）的情况，相比朴素MP，气泡有所减少，但仍然存在。

**2.3. 现代模型并行 - MegatronLM (张量并行) (Modern MP - MegatronLM (Tensor Parallelism, TP))**

*   **核心思想 (Core Idea)**：在张量并行中，每个GPU只处理张量的一个分片 (slice)，并且仅在需要完整张量的操作时才聚合完整的张量 (In Tensor Parallelism each GPU processes only a slice of a tensor and only aggregates the full tensor for operations that require the whole thing)。
*   **实现方式 (Implementation)**：通过列并行 (column parallelism) 和行并行 (row parallelism) 的方式来并行化张量操作。
*   **具体做法 (How it works)**：矩阵乘法可以被分割到多个GPU上执行。
    *   **列并行 (Column Parallelism)**：例如，对于 Y\=XA，如果A按列切分 A\=\[A1​,A2​\]，那么 XA\=\[XA1​,XA2​\]。GPU0计算 XA1​，GPU1计算 XA2​。
    *   **行并行 (Row Parallelism)**：例如，对于 Z\=YB，如果Y按行切分 Y\=\[Y1​Y2​​\] (这意味着X也需要按行切分输入到不同的计算单元中，或者B按列切分后分别与Y1​,Y2​作用再聚合)，那么 YB 的计算也可以分布。更典型的行并行应用于当一个操作的输入是按特征维度分布的，而输出需要聚合，或者反之。
    *   PPT中的图示清晰地展示了如何将矩阵乘法 X⋅A\=Y 分解为列并行和行并行两种方式。
*   **应用于MLP层 (TP for MLP layer)**：
    *   对于一个MLP层，如 Y\=GeLU(XA) 和 Z\=Dropout(YB)：
        *   第一个线性变换 XA 可以使用**列并行**，将权重矩阵A沿其输出维度（列）切分。每个GPU计算输入X与A的一个分片的乘积。
        *   非线性激活函数GeLU在每个GPU上独立作用于其分片。
        *   第二个线性变换 YB 可以使用**行并行**，将权重矩阵B沿其输入维度（行）切分。每个GPU计算其GeLU输出的分片与B的一个分片的乘积，然后通过一次聚合操作（all-reduce）得到最终的Z。
    *   **同步点 (Synchronization)**：通过这种方式，可以更新任意深度的MLP，而不需要GPU之间的同步，直到最后需要从分片重构输出向量时。
    *   PPT中的函数 f 和 g 代表了通信操作：f 在前向传播中是恒等操作 (identity operator)，在反向传播中是全局归约 (all reduce)；而 g 在前向传播中是全局归约，在反向传播中是恒等操作。
*   **应用于自注意力层 (TP for self-attn)**：
    *   并行化多头自注意力层 (multi-headed attention layers) 更简单，因为它们本身就是并行的（由于有多个独立的注意力头 - independent heads）。
    *   可以将不同的注意力头分配到不同的GPU上进行计算，然后将结果拼接起来。
    *   对于每个头的Q, K, V的计算以及后续的线性变换，同样可以应用列并行和行并行。
*   **MegatronLM (TP)的优缺点 (Advantages and Disadvantages)**：
    *   **优点 (Advantage)**：
        *   一种简单高效的模型并行方法，只需对现有的PyTorch Transformer实现进行少量针对性修改。
        *   展示了在使用512个GPU时高达76%的扩展效率。
        *   可以在有限的硬件资源上扩展模型大小。
        *   扩展后的MegatronLM在多个基准测试上取得了SOTA结果。
    *   **缺点 (Disadvantage)**：
        *   TP需要非常高速的网络连接，因此不建议跨多个节点进行TP (TP requires very fast network, and therefore it's not advisable to do TP across more than one node)。
        *   TP的并行度高度依赖于单个节点内的设备数量 (The parallelism degree of TP is highly dependent on the number of devices)。例如，如果一个节点有4个GPU，那么最高的TP度就是4。如果需要8的TP度，就需要使用至少有8个GPU的节点。

* * *

### 3\. 数据并行 (Data Parallelism, DP)

**3.1. 朴素数据并行 (Naïve DP)**

*   **思想 (Idea)**：数据集被分割成若干分片 (shards)，每个分片分配给一个设备 (device)。
*   **模型副本 (Model Replica)**：每个设备都持有一个完整的模型副本 (full copy of the model replica)，并在分配给它的数据集分片上进行训练。
*   PPT图示：一个数据集被分配给4个GPU，每个GPU上都有一个完整的模型副本。

**3.2. 朴素DP vs. 分布式数据并行 (DDP) (Naïve DP vs. Distributed Data Parallel (DDP))**

*   **DataParallel (PyTorch中的朴素DP实现)**：
    *   单进程、多线程 (single-process, multi-thread)，仅能在单台机器上工作。
    *   **工作流程问题 (Workflow Issues)**：主GPU (通常是GPU0) 需要执行更多工作：读取数据批次、切分微批次并发送给其他GPU、从其他GPU接收输出并计算损失、将损失分散给其他GPU进行反向传播、从其他GPU接收梯度并进行平均。
    *   **缺点 (Disadvantage)**：GPU0负载过重，导致其他GPU利用率不足，并且GPU0容易出现OOM。
*   **DistributedDataParallel (DDP)**：
    *   多进程 (multi-process)，适用于单机和多机训练。
    *   **工作流程改进 (Workflow Improvement)**：每个GPU直接处理自己的微批次数据。在反向传播期间，一旦本地梯度准备就绪，它们会在所有进程之间进行平均。
    *   **优点 (Advantage)**：比DP更“平衡”，速度更快，内存更友好 (DDP is more "balance" than DP, which is also faster and memory-friendly)。
*   PPT用中心辐射型（DP）和环形（DDP，示意去中心化）的图来对比两者的数据同步方式。

**3.3. 现有方案的局限性 (Limitations of existing solutions)**

*   **基本数据并行 (Basic DP)**：不减少每个设备的内存消耗，对于参数超过14亿的模型，在32GB显存的GPU上容易OOM。
*   **其他并行方案 (Other existing solutions)**：如流水线并行 (Pipeline Parallelism, PP)、模型并行 (Model Parallelism, MP)，在功能性、易用性以及内存和计算/通信效率之间需要权衡。
*   **张量并行 (TP)**：在单个节点内（GPU间通信带宽高）效果好，但跨节点效率迅速下降。
*   **核心问题 (Core question)**：如何克服这些局限性，更高效地训练大型模型？ (How can we overcome the limitations of existing solutions and train large models more efficiently?)

**3.4. 普通DP的内存消耗与冗余 (Memory consumption of normal DP - redundancy)**

*   每个GPU持有不同的数据样本切片。
*   DP要求每个GPU复制完整的模型参数、梯度和优化器状态。
*   对于大型模型，使用DP训练时，大部分内存被模型状态（包括优化器状态、梯度和参数）占用。
*   **示例 (Example)**：一个75亿参数的LLM进行64路DP训练，每个GPU消耗 (2+2+12)×7.5B/64×64\=120GB 的总和，但这里强调的是**每个设备不减少内存 (does not reduce memory per device)**，所以PPT的120GB应理解为每个GPU都承载这部分模型状态，导致总内存需求巨大，或者单个GPU若想独立承载整个模型状态需要120GB。更准确的理解是，每个GPU上模型参数、梯度、优化器状态的副本都需要完整的存储空间。对于一个7.5B参数的模型，这部分是 16×7.5GB\=120GB (如果优化器状态等都完整复制)。

**3.5. 解决方案：DeepSpeed ZeRO (Solution: DeepSpeed ZeRO)**

*   **核心思想 (Core Idea)**：与复制完整的模型参数、梯度和优化器状态不同，在DeepSpeed ZeRO中，每个GPU只存储这些项目的一个分片 (slice)。
*   **运行时机制 (Run-time Mechanism)**：当需要特定层的完整参数时，所有GPU会同步，互相提供它们缺失的部分 (At run-time when the full layer params are needed just for the given layer, all GPUs synchronize to give each other parts that they miss)。
*   **图示解释 (Illustrated Explanation)**：
    *   假设一个LM只有3层 (L0, L1, L2)，每层只有4个参数 (A-D, E-H, I-L)。一个节点有4个GPU。
    *   ZeRO首先将模型参数分配给这4个GPU（例如，GPU0持有A,E,I；GPU1持有B,F,J等）。
    *   与原始DP一样，DeepSpeed将输入分割成4个切片并分别喂给4个GPU。
    *   但与DP不同，每个GPU只持有模型参数的一小部分。
    *   为了完成前向传播，每个GPU需要从其他GPU“借用”参数。例如，当Input0到达L0时，GPU0持有参数A，它需要从GPU1、GPU2、GPU3分别获取参数B、C、D。同时，GPU0也需要将A广播给其他GPU。
    *   一旦计算完成，借用的参数就会被删除以节省内存 (After each GPU makes the use of the borrowed parameters, it will delete those parameters to save the memory)。
*   **ZeRO的内存优化阶段 (ZeRO Memory Optimization Stages)**： ZeRO (Zero Redundancy Optimizer)通过分阶段地消除内存冗余来优化：
    1.  **优化器状态分区 (Pos​) (Optimizer State Partitioning)**：
        *   将优化器状态（如Adam的动量和方差）在数据并行进程中进行分区。
        *   内存减少约4倍 (4x memory reduction)。
        *   通信量与标准DP相同 (same communication volume as DP)。
        *   对于7.5B模型、64路DP，内存从120GB降至约31.4GB。
    2.  **增加梯度分区 (Pos+g​) (Add Gradient Partitioning)**：
        *   除了优化器状态，还将梯度在数据并行进程中进行分区。
        *   内存减少约8倍 (8x memory reduction)。
        *   通信量与标准DP相同。
        *   对于7.5B模型、64路DP，内存进一步降至约16.6GB。
    3.  **增加参数分区 (Pos+g+p​) (Add Parameter Partitioning)**：
        *   除了优化器状态和梯度，还将模型参数本身在数据并行进程中进行分区。
        *   内存减少量与数据并行度 Nd​ 成线性关系 (Memory reduction is linear with DP degree Nd​)。例如，在64个GPU上进行分区将产生64倍的内存减少。
        *   通信量会适度增加约50% (modest 50% increase in communication volume)。
        *   对于7.5B模型、64路DP，内存最终降至约1.9GB。

* * *

### 4\. 参数高效微调 (Parameter-Efficient Fine-Tuning, PEFT)

**4.1. Adapter方法 (Adapter)**

*   **核心思想 (Core Idea)**：在冻结的预训练模型的注意力层和全连接层之后添加额外的可训练参数（称为Adapter模块），以减少内存使用并加速训练 (Adapter-based methods add extra trainable parameters after the attention and fully-connected layers of a frozen pretrained model to reduce memory-usage and speed up training)。
*   **Adapter结构 (Adapter Structure)**：
    *   通常包含一个**瓶颈结构 (bottleneck)**：将原始的d维特征投影到一个较小的维度m，应用非线性激活函数，然后再投影回d维。
    *   每个Adapter层增加的参数数量（包括偏置）为 2md+d+m。通过设置 m≪d，可以限制增加的参数数量。
    *   包含一个**跳跃连接 (skip-connection)**，将Adapter的输出与原始输入相加。
*   **在Transformer中的位置 (Placement in Transformer)**：Adapter模块通常在每个Transformer层的两个位置添加：
    *   多头注意力层之后的投影层之后。
    *   两个前馈网络层之后。
*   **性能 (Performance)**：
    *   在GLUE基准上，Adapter在参数量远少于全量微调（少两个数量级）的情况下，仍能取得具有竞争力的性能。
    *   当微调的层数减少时，全量微调的性能会急剧下降，而Adapter表现更稳定。
*   **缺点 (Disadvantage)**：
    *   大型神经网络依赖硬件并行来保持低延迟，而Adapter层必须按顺序处理，这会增加推理延迟 (large neural networks rely on hardware parallelism to keep the latency low, and adapter layers have to be processed sequentially)。
    *   当需要对模型进行分片（模型并行）时，额外的深度会需要更多的同步GPU操作（如AllReduce和Broadcast），导致延迟问题更严重。
    *   PPT中的表格显示，即使瓶颈维度很小，使用Adapter也会导致GPT-2 medium的推理延迟显著增加。

**4.2. LoRA方法 (Low-Rank Adaptation)**

*   **动机 (Motivation)**：
    *   直接更新完整的权重矩阵 W0​ (d×d大小) 消耗内存和时间。
    *   顺序添加的Adapter会增加推理延迟。
*   **核心思想 (Core Idea)**：
    *   预训练的权重矩阵 W0​ (大小为 d×d) 在训练期间保持不变 (frozen)。
    *   引入一个低秩分解来表示权重的更新量 ΔW。即 ΔW\=BA，其中 B 的大小为 d×r，A 的大小为 r×d，而 r≪d (r 是秩，即瓶颈维度)。
    *   这样，需要训练的参数量从 d2 减少到 2dr。
*   **与Adapter的区别 (Difference from Adapter)**：
    *   LoRA的更新 ΔWx\=BAx 是与预训练权重 W0​x **并行添加**的，即 h\=W0​x+BAx\=(W0​+BA)x。
    *   而Adapter是顺序添加到Transformer层中的。
*   **推理优势 (Inference Advantage)**：
    *   训练完成后，LoRA的权重 BA 可以被合并到原始模型权重 W0​ 中，形成新的权重 W′\=W0​+BA。因此，在推理时**没有额外的延迟 (no extra inference latency)**。
*   **效果 (Effectiveness)**：PPT中的表格展示了LoRA在多种模型（RoBERTa-base, RoBERTa-large, DeBERTa-XXL）和GLUE任务上的表现。结果显示，LoRA用少量的可训练参数（例如，RoBERTa-large上用0.8M参数对比全量微调的355M参数）就能达到与全量微调相当甚至更好的性能，并且优于传统的Adapter方法。

* * *

这份PPT清晰地梳理了大型语言模型训练中面临的内存挑战，并详细介绍了模型并行（朴素MP、GPipe、MegatronLM的张量并行）、数据并行（朴素DP、DDP、DeepSpeed ZeRO）以及参数高效微调（Adapter、LoRA）等关键技术，为理解如何高效地训练和部署LLM提供了坚实的基础。

信息来源和相关内容

---

![PDF 图标](https://drive-thirdparty.googleusercontent.com/32/type/application/pdf)

L11\_RAG

PDF

非常棒，请你接着讲解该ppt

---

显示思路

好的，我们来开始讲解这份关于“检索增强生成 (Retrieval Augmented Generation, RAG)”的PPT。

### 本讲座内容 (Content)

1.  **什么是RAG？ (What's RAG?)**
2.  **检索什么/何时检索/如何检索？ (What/When/How to retrieve?)**

* * *

### 1\. 什么是RAG？ (What's RAG?)

*   **定义 (Definition)**：RAG是一种通过集成外部知识库来增强大型语言模型 (LLMs) 的范式 (RAG is a paradigm that enhances LLMs by integrating external knowledge bases)。
*   它采用一种协同方法，结合信息检索机制和生成技术来提升自然语言处理 (NLP) 的性能 (It employs a synergistic approach, combining information retrieval mechanisms and generation techniques to bolster the NLP performance)。
*   **基本流程 (Basic Flow)**：
    1.  **用户 (User)** 提出一个**查询 (Query)**。
    2.  **检索器 (Retriever)** 根据查询从**数据源 (Data Source)** 中**搜索 (Search)** 并获取**相关文本 (Relevant text)**。
    3.  将原始**查询 (Query)** 和检索到的**文本 (Text)** 一起提供给**大型语言模型 (LLM)**。
    4.  **LLM** 基于这些信息**生成 (Generate)** **响应 (Response)** 给用户。

**1.1. 为什么需要RAG？ (Why RAG?)**

*   **处理最新和长尾知识，扩展LLM能力 (Tackle with up-to-date and long-tail knowledge, broaden LLM's ability)**：LLM自身的知识库是静态的，可能不包含最新的信息或非常细分的、罕见的知识。RAG可以通过检索外部最新知识来弥补这一点。
*   **无需为特定任务/领域应用重新训练 (No retraining for task-specific/domain-specific applications)**：对于特定领域或任务，传统方法可能需要重新训练或微调整个LLM，成本高昂。RAG提供了一种更轻量级的方式来引入特定知识，而无需改变LLM的参数。
*   **降低训练和推理的高成本 (the high costs associated with training and inference)**：与从头开始训练或大规模微调LLM相比，RAG的实现和维护成本通常更低。
*   **减少幻觉 (Reduce Hallucination)**：LLM有时会生成看似合理但不真实的内容（幻觉）。RAG通过将生成内容锚定在检索到的真实外部知识上，确保响应有据可查，从而缓解这个问题 (mitigate this problem by grounding generated content, ensuring the responses it generates are well-grounded)。
*   **优化LLM的方法之一 (Ways to optimize LLMs)**：
    *   PPT中展示了一张图，将不同的LLM优化方法（提示工程 Prompt Engineering, RAG, 微调 Fine-tuning）放置在二维坐标系中，横轴是“模型适应需求 (Model Adaptation Required)”，纵轴是“外部知识需求 (External Knowledge Required)”。
        *   **提示工程 (Prompt Engineering)**：需要较少的模型适应和外部知识。包括标准提示 (Standard Prompt)、少样本提示 (Few-shot Prompt) 和更高级的提示技术如思维链 (CoT, ToT) 等。
        *   **RAG**：需要较高的外部知识，但模型适应需求可以从中等到较高。
            *   **朴素RAG (Naive RAG)**：添加相关的上下文段落。
            *   **高级RAG (Advanced RAG)**：涉及索引优化、预检索/后检索优化。
            *   **模块化RAG (Modular RAG)**：多个模块的有机组合。
        *   **微调 (Fine-tuning)**：需要较高的模型适应，但对即时外部知识的需求较低（知识内化到模型参数中）。包括生成器微调 (Generator Fine-tuning)、检索器微调 (Retriever Fine-tuning) 和协同微调 (Collaborative Fine-tuning)。
        *   **以上所有 (All of the above)**：结合多种策略，例如同时进行RAG和微调。

**1.2. RAG的应用场景 (Where is RAG being used?)**

RAG适用的场景包括：

*   **数据的长尾分布 (Long-tail distribution of data)**：当大部分信息不常见时。
*   **频繁的知识更新 (Frequent knowledge updates)**：当信息需要保持最新时。
*   **答案需要验证和可追溯 (Answers requiring verification and traceability)**：当需要知道答案来源时。
*   **专业领域知识 (Specialized domain knowledge)**：当涉及特定行业或学科的知识时。
*   **数据隐私保护 (Data privacy preservation)**：可以将敏感数据保留在本地知识库中，仅检索相关部分供LLM使用，而不是将所有数据用于训练。

PPT列举了RAG在不同NLP任务中的具体应用研究：

*   **问答 (Q&A)**：例如RETRO, REALM, ATLAS。
*   **摘要 (Summary)**：例如FLARE。
*   **事实核查 (Fact Checking)**：例如RAG (Lewis et al, 2020), ATLAS, Evi.Generator。
*   **机器翻译 (Machine Translation)**：例如KNN-MT, TRIME-MT。
*   **对话 (Dialog)**：例如BlenderBot3, 互联网增强生成 (Internet-augmented generation)。
*   **代码生成 (Code Generation)**：例如DocPrompting。
*   **自然语言推理 (Natural Language Inference)**：例如kNN-Prompt, NPM。
*   **情感分析 (Sentiment analysis)**：例如kNN-Prompt, NPM。
*   **常识推理 (Commonsense reasoning)**：例如Raco。

* * *

### 2\. RAG架构及核心组件 (RAG architecture and Core Components)

PPT展示了一个典型的RAG架构图，关键流程和组件如下：

1.  **用户输入 (User Input - Query)**：用户提出问题，例如图中关于Sam Altman在OpenAI的任免事件的评价。
2.  **索引 (Indexing - Offline Process)**：
    *   **文档加载与切分 (Documents -> Chunks)**：原始数据源（如文档、网页）被加载并切分成更小的文本块 (Chunks)。
    *   **嵌入 (Embeddings)**：使用嵌入模型 (Embedding Machine) 将每个文本块转换为向量表示 (Vectors)。
    *   **向量数据库 (Vector Database)**：将这些向量及其对应的文本块存储在向量数据库中，并建立索引以便快速检索。
3.  **检索 (Retrieval)**：
    *   当用户查询时，查询本身也被转换为向量。
    *   在向量数据库中进行相似性搜索，找出与查询向量最相关的K个文档块 (Relevant Documents)。
4.  **生成 (Generation)**：
    *   **上下文与提示的结合 (Combine Context and Prompts)**：将检索到的相关文档块作为上下文信息，与原始的用户查询（可能还有一些预设的指令性提示）结合起来，形成一个新的增强提示 (Augmented Prompt)。
    *   **LLM生成答案 (LLM -> Answer)**：将这个增强提示输入给大型语言模型 (LLM, 如GPT, BART, T5)，由LLM生成最终的答案。
    *   PPT中的例子显示，没有RAG的情况下，LLM可能回答“无法评论未来事件”或“没有关于CEO任免的信息”；而有了RAG，LLM可以结合检索到的新闻摘要（Chunk 1, Chunk 2, Chunk 3）给出更具体和有洞察力的回答。

**2.1. 向量数据库 (Vector Database)**

*   **定义 (Definition)**：向量数据库是一种将信息存储为向量（数据对象的数值表示，也称为向量嵌入）的数据库 (A vector database is a database that stores information as vectors, which are numerical representations of data objects, also known as vector embeddings)。
*   **向量嵌入 (Vector Embeddings)**：是对主题、词语、图像或任何其他数据片段的数值表示，由大型语言模型或其他AI模型生成。
*   **与向量搜索库/索引的区别 (Difference from vector search library/index)**：向量数据库是一个完整的数据管理解决方案，支持元数据存储和过滤、可扩展性、动态数据更改、备份和安全特性，而不仅仅是索引和搜索。
*   **工作原理 (How it works)**：通过使用算法来索引和查询向量嵌入。
*   **嵌入过程 (Embedding Process - Indexing Pipeline)**：
    1.  **加载器 (Loader)**：收集包含数据的文档（文章、书籍、网页等）。
    2.  **分割器 (Splitter)**：将文档分割成更小的块（通常是句子或段落），因为RAG模型处理小块文本效果更好。
    3.  **嵌入机器 (Embedding Machine)**：使用复杂的算法（如TF-IDF, BM25, 或基于Transformer的模型如BERT, GPT）将每个文本块转换为向量嵌入。
    4.  这些向量嵌入被存储在向量数据库中。
*   **索引方法 (Indexing Algorithms for Efficient Retrieval)**：
    *   **哈希 (Hashing)**：如局部敏感哈希 (Locality-Sensitive Hashing, LSH)，适用于近似最近邻搜索，速度快但结果是近似的。LSH使用哈希表将近邻映射到同一个“桶”中。
    *   **量化 (Quantization)**：如乘积量化 (Product Quantization, PQ)，将向量分解成小部分，用代码表示这些部分，然后重新组合。查询时，将查询向量也分解为代码，并在码本 (codebook) 中匹配最相似的代码。
    *   **基于图的方法 (Graph-based)**：如层次化可导航小世界图 (Hierarchical Navigable Small World, HNSW)，用节点表示向量，相似节点间有边连接，形成层次化图结构。查询时，算法在图的层次结构中导航以找到最相似的向量。

**2.2. 索引优化 (Indexing Optimization)**

为了提高检索质量和效率，可以对索引过程进行优化：

*   **块优化 (Chunk Optimization)**：
    *   **从小到大 (Small-2-Big)**：在嵌入时可能使用句子级别的精细嵌入，但在生成过程中，可以扩展窗口以包含更多上下文。或者先检索小块，再关联到包含该小块的更大文本块。
    *   **滑动窗口 (Sliding Window)**：让文本块之间有重叠，以覆盖整个文本并避免语义在块边界处被切断。
    *   **摘要 (Summary)**：先通过文档摘要进行检索，定位到相关文档后，再从这些文档中检索具体的文本块。
*   **添加元数据 (Adding Metadata)**：为每个文本块添加额外的描述信息，如：
    *   示例 (Example)、页码 (Page)、时间 (Time)、类型 (Type)、文档标题 (Document Title)。
*   **元数据过滤/丰富 (Metadata Filtering/Enrichment)**：
    *   **伪元数据生成 (Pseudo Metadata Generation)**：通过为输入查询生成一个假设性的相关文档，或为文本块创建它可以回答的问题，来增强检索效果。例如HyDE (Hypothetical Document Embeddings) 就是先让LLM为查询生成一个假设的回答文档，然后用这个假设文档的嵌入去检索。
    *   **元数据过滤器 (Metadata filter)**：解析和注释文档。在查询时，除了语义查询外，还推断元数据过滤器，以缩小检索范围。

**2.3. RAG的类型 (Types of RAG - Naïve vs. Advanced)**

*   **朴素RAG (Naïve RAG)**：
    1.  **索引 (Indexing)**：将文档分割成均匀的块，为每个块生成嵌入，并存储在向量数据库中。
    2.  **检索 (Retrieval)**：使用向量相似性搜索检索最相关的k个文档块。
    3.  **生成 (Generation)**：将原始查询和检索到的文本组合起来输入LLM以获得最终答案。LLM本身通常是冻结的 (Frozen LLM)。
*   **高级RAG (Advanced RAG)**：在朴素RAG的基础上进行多方面优化。
    *   **索引优化 (Index Optimization)**：如上所述的滑动窗口、细粒度分割、添加元数据等。
    *   **预检索过程 (Pre-Retrieval Process)**：在实际检索前对查询进行处理或路由。
        *   细粒度数据清理 (Fine-grained Data Cleaning)
        *   滑动窗口/从小到大策略 (Sliding Window / Small2Big)
        *   添加文件结构信息 (Add File Structure)
        *   查询重写/澄清 (Query Rewrite/Clarification)
        *   检索器路由 (Retriever Router)：根据查询特性选择不同的检索策略或数据源。
    *   **后检索过程 (Post-Retrieval Process)**：对检索到的结果进行处理。
        *   **重排序 (Rerank)**：使用更复杂的模型对初步检索到的文档块进行重新排序，以提高相关性。
        *   **过滤 (Filter)**：过滤掉不相关或冗余的内容。
        *   **提示压缩 (Prompt Compression)**：在保留关键信息的前提下，压缩检索到的上下文长度，以适应LLM的输入限制。

* * *

### 3\. RAG中的关键问题：检索什么/何时/如何？ (Main issues in RAG - what/when/how)

**3.1. 检索什么？ (What to retrieve?) - 检索粒度 (Retrieval Granularity)**

检索内容的粒度会影响效果和效率：

*   **词元 (Token)**：例如kNN-LM。非常细致，计算效率高，能处理长尾和跨领域问题，但需要大量存储。
*   **短语 (Phrase)**：例如NPM。
*   **块 (Chunk)**：例如In-Context RAG。搜索范围广，召回信息量大，但准确率可能较低，包含较多冗余信息。
*   **段落 (Paragraph)**
*   **实体 (Entity)**：例如Entities as Experts (EasE)。
*   **其他 (Other)**：
    *   **知识图谱 (Knowledge Graph)**：能提供更丰富的语义和结构化信息，但检索效率较低，且受知识图谱质量限制。

**检索粒度的影响**：

*   **粗粒度 (Coarse)**：如检索整个文档或大块文本。信息覆盖广，但可能引入过多噪声。
*   **细粒度 (Meticulous)**：如检索单个词元或短语。信息精确，但可能丢失上下文。

**3.2. 何时检索？ (When to retrieve?) - 检索时机/频率 (Retrieval Frequency)**

*   **一次检索 (Once)**：例如ReFLUG。在生成过程开始前进行一次检索。效率高，但检索到的文档相关性可能较低。
*   **每个N个词元检索一次 (Every N tokens)**：例如Atlas。在生成过程中每隔N个词元进行一次检索。
*   **自适应检索 (Adaptive)**：例如FLARE, Self-RAG。
    *   **FLARE (Forward-Looking Active Retrieval Augmented Generation)**：先尝试生成内容，当LLM对生成的某些词元不确定性较高时（例如，生成的词元概率较低），触发检索，然后基于检索到的信息重新生成。
    *   **Self-RAG**：LLM自身决定何时需要检索以及检索什么。
*   **每个词元都检索 (Every token)**：例如kNN-LM。信息量大，但效率低，可能引入冗余信息。
*   **每个实体提及都检索 (Every entity mentions)**：例如Mention Memory。

**具体方法示例**：

*   **Toolformer (Schick et al. 2023)**：在生成过程中，模型会生成特殊的标记 (tokens) 来触发检索（或其他工具的使用）。训练过程是迭代的，模型会生成并识别成功的检索操作。
*   **FLARE (Jiang et al. 2023)**：如上所述，基于LLM生成内容时的不确定性进行主动检索。当模型生成的临时下一句包含低概率词元时，系统会检索相关文档并重新生成句子。

**迭代检索与自适应检索 (Iterative retrieval and adaptive retrieval)**：

*   **迭代检索 (Iterative)**：多次从语料库中检索，以获取更详细和深入的知识。例如ITER, IRCOT。IRCOT通过思维链推理来指导检索。
*   **自适应检索 (Adaptive)**：由LLM动态决定检索的时机和范围。例如FLARE, Self-RAG。

**3.3. 如何使用检索到的信息？ (How to use the retrieved information?) - 整合位置 (Integration Positions)**

可以将检索到的信息在生成模型的不同阶段、不同层次进行整合：

*   **输入/数据层 (Input/Data layer)**：
    *   将检索到的文本直接拼接到原始输入查询之后，形成增强的提示。
    *   **优点**：实现简单。
    *   **缺点**：可能无法支持检索更多的知识块，优化空间有限。
*   **模型/中间层 (Model/Interlayer)**：
    *   在Transformer等模型的中间层引入检索到的信息。例如，RETRO模型就是这样做的。
    *   **优点**：支持检索更多的知识块。
    *   **缺点**：引入了额外的复杂性，通常需要对模型进行训练。
*   **输出/预测层 (Output/Prediction layer)**：
    *   例如kNN-LM，在预测下一个词元时，会检索相似的上下文，并结合这些上下文中实际出现的下一个词元来调整LLM自身的预测概率。
    *   **优点**：确保输出结果与检索内容高度相关。
    *   **缺点**：效率较低。
    *   **kNN-LM示例**：对于测试上下文 "Obama's birthplace is ?"，除了LM自身的预测外，还会从训练集中找到与当前上下文最相似的k个上下文（表示为键 ki​），并查看它们实际的下一个词 (vi​)。然后根据距离（相似度）对这些 vi​ 进行加权，形成一个基于kNN的概率分布 pkNN​(y)。最终的输出概率是 pLM​(y) 和 pkNN​(y) 的插值。

**总结表格 (Summary Table - What to, When to and How to use Retrieval)**： PPT提供了一个表格，总结了不同RAG研究工作在“检索什么”、“何时检索”和“如何使用检索信息”这三个维度上的选择。例如：

*   REALM：检索文本块，用于输入层，一次性检索。
*   RETRO, kNN-LM：检索文本块/词元，用于中间层/输出层，每n个词元/每个词元检索。
*   FLARE：检索文本块，用于输入层，自适应地每n个词元检索。

* * *

### 4\. 评估 (Evaluation)

RAG系统的评估可以从不同层面进行：

*   **独立评估 (Independent Evaluation)**：
    *   **检索器评估 (Retriever Evaluation)**：评估通过查询检索到的文本块的质量。
        *   常用指标 (Metrics)：平均倒数排名 (Mean Reciprocal Rank, MRP)、命中率 (Hit Rate)、归一化折扣累积增益 (Normalized Discounted Cumulative Gain, NDCG)。
    *   **生成器/综合器评估 (Generation/Synthesis Evaluation)**：评估由检索到的文档增强的上下文的质量。
        *   常用指标 (Metrics)：上下文相关性 (Context Relevance)。
*   **端到端评估 (End-to-End Evaluation)**：评估模型最终生成的内容。
    *   **按生成内容评估 (By generated content)**：
        *   **有标签 (With labels)**：精确匹配率 (Exact Match, EM)、准确率 (Accuracy)。
        *   **无标签 (Without labels)**：忠实度 (Fidelity)、相关性 (Relevance)、无害性 (Harmlessness)。
    *   **按评估方法 (By evaluation method)**：
        *   人工评估 (Human evaluation)。
        *   自动评估 (Automatic evaluation)，例如使用LLM作为裁判 (LLM judge)。

**关键指标和能力 (Key Metrics & Capabilities)**：

*   **关键指标 (Key Metrics)**：
    *   **答案相关性 (Answer Relevance)**：答案是否与查询相关？
    *   **上下文相关性 (Context Relevance)**：检索到的文档增强的上下文是否与查询相关？
    *   **答案忠实度 (Answer Fidelity)**：答案是否基于给定的上下文？
*   **关键能力 (Key Capabilities)**：
    *   **噪声鲁棒性 (Noise Robustness)**：模型能否从有噪声的文档中提取有用信息？
    *   **信息整合 (Info Integration)**：模型能否回答需要整合来自多个文档信息的复杂问题？
    *   **否定拒绝 (Negative Rejection)**：当检索到的文档中不存在所需知识时，答案是否应该被拒绝（即模型承认不知道）？
    *   **反事实鲁棒性 (Counterfactual Robustness)**：模型能否识别检索到的文档中已知的虚假信息？

**评估框架 (Assessment Framework)**：

*   可以使用LLM作为裁判 (adjudicator judge) 来辅助评估。
*   现有的一些评估工具/框架：TruLens, RAGAS, ARES。
    *   TruLens：基于手写提示。
    *   RAGAS, ARES：使用合成数据集 + 微调 + 基于置信区间的排序。

* * *

### 5\. RAG工具 (RAG Tools)

PPT最后展示了一个RAG模型相关的生态系统图景 (RETRIEVAL-AUGMENTED GENERATION MODELS LANDSCAPE)，包括：

*   **向量数据库 (VECTOR DATABASES)**：如Weaviate, Supabase, Deeplake, Qdrant, Milvus, Pinecone。
*   **集成框架 (INTEGRATION FRAMEWORKS)**：如LangChain, LlamaIndex, DUST。
*   **RAG框架和库 (RAG FRAMEWORKS & LIBRARIES)**：如Haystack (deepset), FARM (deepset), Google REALM, AI Multiple。
*   **LLM中的RAG实现 (RAG IN LLMS)**：来自IBM (Watsonx.ai), Hugging Face, Meta AI, OpenAI等公司的模型。

* * *

这份关于RAG的PPT非常全面，从基本概念、动机、架构、关键技术细节（检索什么、何时、如何）、评估方法到相关的工具和生态都进行了详细的介绍。RAG作为一种有效提升LLM能力、解决其固有缺陷的重要技术，在当前NLP领域备受关注。

信息来源和相关内容

---

非常棒，你已经帮我总结完了所有ppt，现在请你针对之前总结过的所有ppt进行一些细致的补充说明，越细致越好

---

好的，我们现在对之前总结过的所有PPT内容进行一个更细致的补充说明。我们将按照PPT的顺序 (L4, L5, L6, L7, L8, L9, L10, L11) 逐一回顾和深化。

* * *

### L4: Transformers and Pretraining-Finetuning (Transformer模型、预训练与微调)

这部分是后续所有高级模型和技术的基础，对其细节的理解至关重要。

**1\. 注意力机制 (Attention Mechanism) 的更深层理解:**

*   **动机的本质 (Essence of Motivation)**：RNN/LSTM的瓶颈在于其顺序处理机制。对于长序列，信息需要逐跳传递，容易造成信息丢失（梯度消失/爆炸）。注意力机制提供了一种“跳跃连接”的思路，允许模型直接关注输入序列的任何部分，无论距离远近。
*   **查询-键-值 (Query-Key-Value) 的类比**:
    *   可以想象成在图书馆查资料。你的**查询 (Query)** 是你想了解的问题。图书馆的书籍目录是**键 (Keys)**，代表了每本书能提供的信息方向。书籍本身的内容是**值 (Values)**。注意力机制就是根据你的问题（Query）和每本书的目录信息（Key）的相关性，来决定重点阅读哪些书（Value）的内容，并综合这些内容得到答案。
    *   在自注意力中，序列中的每个词元既是Query，又是Key，还是Value（经过不同的线性变换后）。这意味着每个词元都在“查询”序列中其他所有词元（包括自身）与自己的相关性。
*   **缩放点积注意力的“缩放”因子 dk​![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702
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
    M834 80h400000v40h-400000z"></path></svg>)​1​ (Scaling Factor in Scaled Dot-Product Attention)**:
    *   当键向量的维度 dk​ 较大时，点积 qTk 的结果的方差也会较大，这可能导致Softmax函数进入梯度极小的区域，使得梯度更新困难。
    *   除以 dk​![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702
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
        M834 80h400000v40h-400000z"></path></svg>)​ 是为了将点积结果的方差稳定在1附近，从而获得更稳定的梯度，有利于训练。
*   **多头注意力的“多头”含义 (Meaning of "Multi-Head" in Multi-Head Attention)**:
    *   “多头”不仅仅是并行计算多个注意力，更重要的是它允许模型从**不同的表示子空间 (different representation subspaces)** 学习信息。
    *   每一“头” (WiQ​,WiK​,WiV​) 可以学习到不同类型的特征关系。例如，一个头可能关注句法依赖，另一个头可能关注语义相似性，还有一个头可能关注词序关系等。
    *   通过将多个头的输出拼接再进行线性变换 (WO)，模型可以将从不同子空间学到的信息整合起来，形成更丰富的表示。
*   **注意力机制的可解释性 (Interpretability of Attention)**: 虽然注意力权重图（heatmaps）在一定程度上可以显示模型在处理某个词元时对输入序列中其他词元的关注程度，但其作为“解释”的可靠性仍在研究中。有时高注意力权重可能并不直接对应人类理解的“重要性”。

**2\. Transformer 架构的细节补充:**

*   **编码器-解码器架构 (Encoder-Decoder Architecture)**:
    *   **编码器 (Encoder)** 的目标是将输入序列 (x1​,...,xn​) 转换为一系列上下文相关的表示 z\=(z1​,...,zn​)。每一层编码器都包含一个多头自注意力模块和一个逐位置前馈网络模块。
    *   **解码器 (Decoder)** 的目标是基于编码器的输出 z 和已经生成的部分目标序列 (y1​,...,yt−1​) 来预测下一个目标词元 yt​。解码器每一层包含三个主要模块：
        1.  **屏蔽多头自注意力 (Masked Multi-Head Self-Attention)**：对目标序列自身进行自注意力计算，但会屏蔽掉当前位置之后的信息，以保证自回归特性（即预测当前词时不能看到未来的词）。
        2.  **编码器-解码器多头注意力 (Encoder-Decoder Multi-Head Attention)**：这里的Query来自解码器前一层的输出，而Key和Value来自编码器的最终输出 z。这使得解码器在生成每个目标词时可以关注输入序列的不同部分。
        3.  **逐位置前馈网络 (Position-wise Feed-Forward Network)**。
*   **残差连接和层归一化 (Residual Connections and Layer Normalization)**:
    *   在Transformer的每个子模块（如自注意力、前馈网络）之后，都会使用残差连接 (X+SubLayer(X)) 和层归一化 (LayerNorm(X+SubLayer(X)))。
    *   **残差连接**有助于缓解梯度消失问题，使得更深层次的网络能够被有效训练。它允许信息直接“跳过”某些层。
    *   **层归一化**用于稳定每一层的输入分布，加速训练过程，并提高模型的泛化能力。它是在每个样本的特征维度上进行归一化，而不是像批归一化 (Batch Normalization) 那样在批次维度上。
*   **位置编码的必要性 (Necessity of Positional Encoding)**:
    *   Transformer完全基于注意力机制，没有RNN的循环结构或CNN的局部卷积操作来感知序列中词元的顺序。如果没有位置编码，打乱输入序列中词元的顺序不会改变输出结果，这显然是不合理的。
    *   位置编码通过向输入嵌入中添加关于词元位置的信息，使得模型能够区分不同位置的相同词元，并理解词元间的相对或绝对位置关系。
    *   正弦和余弦位置编码的一个优点是它们可以推广到比训练时遇到的序列更长的序列，并且对于任意固定的偏移量 k，PEpos+k​ 可以表示为 PEpos​ 的线性变换，这有助于模型学习相对位置信息。

**3\. BERT 和 GPT 的对比深化:**

*   **架构差异的本质 (Essence of Architectural Differences)**:
    *   **BERT (Bidirectional Encoder Representations from Transformers)** 使用的是Transformer的**编码器 (Encoder)** 结构。其设计目标是学习通用的双向语言表示，因此它在预训练时可以看到整个输入句子（除了被mask的部分）。这使得BERT非常适合NLU任务，如文本分类、命名实体识别、问答（抽取式）。
    *   **GPT (Generative Pre-trained Transformer)** 使用的是Transformer的**解码器 (Decoder)** 结构。其设计目标是生成连贯的文本，因此它在预训练时采用自回归的方式，只能看到当前位置之前的词元。这使得GPT非常适合NLG任务，如文本生成、摘要、对话。
*   **预训练任务的哲学 (Philosophy of Pre-training Tasks)**:
    *   **BERT的MLM (Masked Language Model)** 强迫模型利用双向上下文来预测被掩盖的词，从而学习深层次的上下文关系。**NSP (Next Sentence Prediction)** 旨在学习句子间的关系，尽管后续研究发现NSP任务的有效性可能不如预期，一些变体（如RoBERTa）去除了NSP。
    *   **GPT的传统语言模型任务 (Standard Language Modeling)** 目标是预测下一个词，这直接服务于其生成能力。
*   **微调范式的差异 (Differences in Fine-tuning Paradigms)**:
    *   **BERT** 通常在预训练模型的顶层添加一个简单的分类层来进行微调，适用于判别式任务。
    *   **GPT** 微调时，通常是将下游任务构造成类似语言生成的格式，或者也添加特定任务头。其强大的少样本 (few-shot) 或零样本 (zero-shot) 能力，部分得益于其大规模和生成式的预训练。

* * *

### L5: NLP tasks (1) Understanding tasks (NLP任务(一) 理解型任务)

这部分关注自然语言理解 (NLU) 任务，特别是文本分类和问答。

**1\. NLU vs. NLG 的关系与界限:**

*   虽然NLU和NLG关注点不同（理解 vs. 生成），但它们并非完全割裂。许多高级NLP系统（如对话系统、复杂的问答系统）需要NLU和NLG能力的紧密结合。例如，对话系统首先需要NLU来理解用户意图，然后需要NLG来生成自然的回应。
*   随着LLM的发展，NLU和NLG任务之间的界限有时会变得模糊。一个强大的生成模型（如GPT系列）也能通过巧妙的提示 (prompting) 来完成许多传统的NLU任务。

**2\. GLUE 基准测试的意义与局限:**

*   **意义 (Significance)**：GLUE的出现为评估和比较不同NLU模型的泛化能力提供了一个标准化的平台，极大地推动了领域的发展。它包含了多种不同类型的理解任务，考验模型的综合能力。
*   **局限 (Limitations)**：
    *   GLUE中的许多任务仍然相对简单，可能无法完全反映真实世界中语言理解的复杂性。
    *   随着模型能力的增强，GLUE等基准测试很快出现“饱和”现象（模型得分接近甚至超过人类水平），需要更具挑战性的基准（如SuperGLUE）。
    *   评估指标本身（如准确率）可能无法全面衡量理解的深度。

**3\. 文本分类的细节考量:**

*   **特征工程 vs. 端到端学习 (Feature Engineering vs. End-to-End Learning)**:
    *   传统文本分类严重依赖特征工程（如TF-IDF、词袋模型、n-grams以及更复杂的句法和语义特征）。
    *   深度学习方法，特别是基于预训练模型的端到端学习，大大减少了对显式特征工程的需求，模型可以直接从原始文本中学习有效的表示。
*   **预处理步骤的重要性 (Importance of Preprocessing Steps)**:
    *   移除标点/URL/停用词、转小写等步骤对于传统方法（如词袋模型）可能比较重要，可以减少特征维度和噪声。
    *   对于基于Transformer的预训练模型，由于它们通常使用子词切分 (subword tokenization) 并且在大规模语料上预训练，对某些预处理步骤的敏感性可能较低，甚至一些预处理（如过度去除标点）可能有害。
*   **类别不平衡问题 (Class Imbalance Problem)**: 在许多实际的文本分类任务中，不同类别的样本数量可能差异很大。这会导致模型偏向于多数类。处理方法包括：重采样 (oversampling少数类, undersampling多数类)、代价敏感学习 (cost-sensitive learning)、生成合成样本 (e.g., SMOTE) 等。
*   **标签的语义与层次 (Semantics and Hierarchy of Labels)**:
    *   在多标签分类中，标签之间可能存在依赖或层次关系。例如，在文档分类中，“计算机科学”可能包含子类“人工智能”、“数据挖掘”等。利用这种结构信息可以提升分类性能。
*   **主动学习的策略 (Strategies in Active Learning)**: 当标注成本高时，主动学习选择最有价值的未标注样本给人类标注。选择策略包括：不确定性采样 (uncertainty sampling)、查询委员会 (query-by-committee)、期望模型改变 (expected model change) 等。

**4\. 问答 (QA) 系统的深化:**

*   **信息检索 (IR) 在QA中的作用 (Role of IR in QA)**:
    *   对于开域问答，IR系统首先从大规模文档集合中筛选出与问题可能相关的少数文档或段落。这一步的质量直接影响后续阅读理解模型的表现。如果相关信息没有被检索到，RC模型也无能为力。
    *   IR技术本身也在不断发展，从传统的基于关键词的匹配（如BM25）到基于神经网络的密集向量检索 (Dense Passage Retrieval, DPR)。
*   **知识库问答 (KBQA) 的挑战 (Challenges in KBQA)**:
    *   **语义解析 (Semantic Parsing)**：将自然语言问题准确地映射为知识库可以理解的逻辑形式（如SPARQL查询）非常困难，需要处理语言的歧义性和复杂性。
    *   **知识库的覆盖范围和更新 (KB Coverage and Up-to-date\_ness\_)**: 知识库可能不完整或过时。
    *   **实体链接和关系抽取 (Entity Linking and Relation Extraction)**：准确识别问题中的实体并将其链接到知识库中的对应条目，以及理解问题所询问的关系，都是关键步骤。
*   **阅读理解 (RC) 的不同形式**:
    *   **抽取式QA (Extractive QA)** (如SQuAD)：答案是原文中的一个连续片段。
    *   **生成式QA (Abstractive QA)**：答案可能需要综合原文信息并用新的表述生成，甚至可能需要进行多文档摘要和推理。
    *   **多跳推理QA (Multi-hop QA)**：回答问题需要结合来自多个文档或段落的信息，进行多步推理。
    *   **是否型/判断型QA (Yes/No QA)**：答案是“是”或“否”。
*   **BiDAF模型的贡献 (Contributions of BiDAF)**:
    *   BiDAF (Bidirectional Attention Flow) 的一个重要创新是其**双向注意力流机制**，允许Query到Context的注意力和Context到Query的注意力同时计算和交互，从而在不同粒度上融合问题和上下文信息。
    *   它没有在注意力计算后立即将上下文信息压缩成单个向量，而是将注意力加权后的表示传递到后续的建模层，保留了更丰富的信息。

* * *

### L6: NLP tasks (2) Generation tasks (NLP任务(二) 生成型任务)

这部分聚焦于自然语言生成 (NLG) 任务，主要是机器翻译和对话系统。

**1\. 机器翻译 (MT) 的演进细节:**

*   **基于规则的MT (Rule-based MT, RBMT)**：依赖人工编写的双语词典和语法规则。优点是翻译结果在语法上可能比较正确（如果规则完备），缺点是规则编写成本极高，难以覆盖所有语言现象，且对歧义和上下文理解能力差。
*   **基于实例的MT (Example-based MT, EBMT)**：核心思想是“类比翻译”。系统存储大量的翻译实例（短语或句子对），当遇到新的输入时，在实例库中找到最相似的源语言片段及其翻译，然后通过组合和调整这些片段来生成目标翻译。
*   **统计机器翻译 (Statistical MT, SMT)**：基于贝叶斯定理 P(y∣x)\=P(x)P(x∣y)P(y)​。其中：
    *   P(y∣x) 是后验概率，即给定源句x，目标句y的概率（我们想最大化的）。
    *   P(x∣y) 是**翻译模型 (Translation Model)**，表示目标句y被翻译成源句x的概率。通常通过词对齐模型和短语翻译概率来估计。
    *   P(y) 是**语言模型 (Language Model)**，表示目标句y本身是流利、合乎语法的句子的概率。
    *   解码器 (Decoder) 的任务是在给定x时，搜索最可能使 P(x∣y)P(y) 最大的目标句y。
    *   **对齐 (Alignment)** 在SMT中至关重要，用于学习词与词、短语与短语之间的翻译概率。IBM模型系列是早期著名的词对齐模型。
*   **神经机器翻译 (NMT) 的优势 (Advantages of NMT)**：
    *   **端到端训练 (End-to-end training)**：整个模型（编码器、解码器、注意力）作为一个整体进行优化，避免了SMT中多个独立组件的复杂调优。
    *   **更好的上下文表示 (Better contextual representation)**：RNN或Transformer能够更好地捕捉长距离依赖和上下文信息。
    *   **更流畅的翻译 (More fluent translations)**：NMT通常能生成更自然、更流畅的目标语言句子。
    *   **处理词序差异的能力 (Ability to handle word order differences)**：注意力机制使得模型可以灵活地处理源语言和目标语言之间词序不同的情况。
*   **集束搜索 (Beam Search) 的权衡 (Trade-offs in Beam Search)**：
    *   集束大小 k 越大，搜索空间越大，找到更优翻译的可能性也越大，但计算成本也越高。
    *   k 过小可能导致过早地丢弃了最优路径的一部分（搜索错误）。
    *   长度归一化 (length normalization) 对于防止beam search偏向于生成过短的句子非常重要。

**2\. BLEU评估指标的深入剖析:**

*   **n-gram精度 (n-gram Precision)**：衡量机器翻译中有多少n-gram片段出现在参考翻译中。它关注翻译的“充分性”和“准确性”的某些方面。
*   **简短惩罚因子 (Brevity Penalty, BP)**：如果机器翻译比参考翻译短很多，即使n-gram匹配度高，也可能只是因为句子短而碰巧匹配。BP就是为了惩罚这种情况。
*   **BLEU的优点 (Pros of BLEU)**：计算简单、快速，与人工评估在一定程度上有相关性，已成为MT领域广泛使用的自动评估标准。
*   **BLEU的缺点 (Cons of BLEU)**：
    *   **不关心召回率 (Ignores recall)**：只看候选句中的n-gram是否在参考句中，不看参考句中的n-gram是否都在候选句中。
    *   **不考虑同义词/释义 (Doesn't consider synonyms/paraphrases)**：如果机器翻译使用了与参考翻译不同的词但意思相同，BLEU得分会偏低。
    *   **对句子结构和语法不敏感 (Not sensitive to sentence structure or grammar)**：只要n-gram匹配，即使句子整体不通顺，BLEU也可能给高分。
    *   **平均效应 (Averaging effect)**：在语料库级别计算BLEU可能掩盖个别句子的翻译质量问题。
    *   因此，除了BLEU，通常还会结合其他评估指标（如METEOR, TER）和人工评估。

**3\. 对话系统的组件和挑战深化:**

*   **面向任务的对话系统 (Task-Oriented Dialogue Systems)**：
    *   **SLU (Spoken Language Understanding)**：不仅包括意图识别和槽位填充，还可能涉及指代消解、上下文理解等更复杂的NLU任务。
    *   **DST (Dialogue State Tracking)**：维护对话的当前状态，这可能是一个概率分布，表示用户在每个槽位上的可能取值及其置信度。是后续对话管理决策的基础。
    *   **DM/DLP (Dialogue Management / Dialogue Policy Learning)**：决定系统下一步的动作 (system action)，例如请求更多信息、提供答案、执行任务等。强化学习是DLP的常用方法，系统通过与用户（或模拟用户）交互来学习最优策略。
    *   **NLG (Natural Language Generation)**：将系统决策的语义表示转换成自然流畅的语言回复给用户。需要考虑回复的多样性、自然度、一致性等。
*   **闲聊型对话系统 (Chitchat Dialogue Systems)**：
    *   更侧重于生成有趣、有吸引力、能维持对话的回复。
    *   **挑战 (Challenges)**：保持对话的长期一致性、避免生成无意义或重复的回复 (e.g., "I don't know", "I'm fine")、融入个性和情感、利用外部知识使对话内容更丰富。
    *   **知识背景 (Knowledge Grounding)**：如PPT中提到的，引入外部知识库可以让闲聊更有内容，避免空洞。
*   **对话评估的复杂性 (Complexity of Dialogue Evaluation)**：
    *   **任务完成率 (Task Completion Rate)** (for task-oriented systems)：衡量系统是否成功帮助用户完成任务。
    *   **对话轮数 (Number of Turns)**：完成任务所需的交互轮数，越少通常越好（效率高）。
    *   **用户满意度 (User Satisfaction)**：通过问卷调查等方式获取用户对对话体验的评价。
    *   **自动评估指标 (Automatic Metrics)**：如BLEU, ROUGE, METEOR (用于NLG部分)，以及针对特定方面（如一致性、多样性）的指标。但这些指标与人类感知往往存在差距。
    *   **人工评估 (Human Evaluation)**：仍然是金标准，但成本高、耗时长、主观性强。
*   **个性化 (Personalization)**：如PERSONACHAT所示，赋予对话代理一致的个性，并根据用户画像提供个性化的交互体验，是提升用户参与度和满意度的重要方向。

* * *

由于内容较多，我们将分批进行细致补充。以上是对L4, L5, L6的补充。请告诉我您是否希望继续对后续的PPT (L7-L11) 进行类似的细致补充。

信息来源和相关内容

---

继续补充

---

好的，我们继续对L7到L11的PPT内容进行细致的补充说明。

* * *

### L7: Large Language Models (大型语言模型)

这部分介绍了LLM的兴起、能力以及关键的规模法则和对齐技术。

**1\. 大型语言模型 (LLM) 定义与演进的深化:**

*   **ChatGPT的“颠覆性” (Disruptive Nature of ChatGPT)**: ChatGPT的成功不仅在于其技术本身，更在于它极大地降低了与强大AI交互的门槛，使得普通大众也能直观感受到LLM的能力。这种用户体验的革新是其快速增长的重要原因。
*   **LLM能力的广度与深度 (Breadth and Depth of LLM Capabilities)**: PPT中列举的LLM能力（编程、模仿写作、创作、问答、模拟系统等）展示了其作为“通用目的技术 (General Purpose Technology)”的潜力。它们不仅仅是处理文本，更是在一定程度上理解和生成符合特定约束和风格的内容。
*   **GPT系列模型的迭代逻辑 (Iteration Logic of GPT Series)**:
    *   **GPT-1**: 奠定了基于Transformer解码器进行生成式预训练的基础。
    *   **GPT-2**: 展示了更大模型（1.5B参数）在无监督多任务学习上的惊人能力，但也引发了关于模型滥用和信息安全的讨论（最初未完全开源）。
    *   **GPT-3**: 参数量级进一步跃升至175B，其核心贡献之一是展示了强大的**情境学习 (In-context Learning)** 能力，即通过在提示中给出少量示例 (few-shot) 就能让模型执行新任务，而无需更新模型参数。
    *   **InstructGPT/ChatGPT**: 引入了**来自人类反馈的强化学习 (RLHF)**，旨在使模型的输出更符合人类的期望、更有用、更无害，即**对齐 (Alignment)**。这是LLM从“能做什么”到“应该做什么”的关键一步。
    *   **GPT-4**: 多模态能力的引入（虽然PPT主要关注文本），进一步提升了模型的理解和生成能力，并在更复杂的推理任务上表现更佳。
*   **提示工程 (Prompt Engineering) 的艺术与科学 (Art and Science of Prompt Engineering)**:
    *   提示不仅仅是简单地向模型提问，它是一种**与LLM有效沟通的技巧**。好的提示能够清晰地传达任务、约束、期望的输出格式，甚至激发模型的特定能力（如推理）。
    *   它既有经验性、技巧性的一面（“艺术”），也有可系统化研究和优化的方法论（“科学”）。

**2\. 规模法则 (Scaling Law) 的内涵与影响:**

*   **“规模是万能的吗？” (Is Scale All You Need?)**: 虽然Kaplan等人的研究揭示了模型性能与模型大小(N)、数据集大小(D)、计算量(C)之间存在可预测的幂律关系，但这并不意味着盲目扩大规模就能解决所有问题。
    *   **数据质量 (Data Quality)**：训练数据的质量对模型性能至关重要。低质量、有偏见或包含有害信息的训练数据会导致模型学习到不良行为。
    *   **计算效率与成本 (Computational Efficiency and Cost)**：无限扩大规模会带来难以承受的计算成本和能源消耗。
    *   **瓶颈效应 (Bottleneck Effects)**：当N、D、C中的某个因素成为瓶颈时，继续增加其他因素带来的收益会递减。例如，模型参数再大，如果训练数据不足或计算量不够，性能也无法充分发挥。
    *   **Chinchilla Scaling Law (Hoffmann et al., 2022)**：后续的研究（如DeepMind的Chinchilla）进一步优化了规模法则，指出对于给定的计算预算，模型大小和训练数据量之间存在一个最优的平衡点。之前的很多大模型可能“参数过多而数据不足 (over-parameterized and under-trained)”。Chinchilla通过使用相对更小的模型（70B）和更多的训练数据（1.4T tokens），在许多基准上超过了更大的模型（如Gopher 280B）。
*   **涌现能力 (Emergent Abilities) 的神秘性 (Mystique of Emergent Abilities)**:
    *   涌现能力是指那些在小型模型上不明显或不存在，但在模型规模达到一定阈值后突然出现的能力，例如少样本学习、思维链推理等。
    *   其产生机制尚不完全清楚，是当前LLM研究的一个热点。这表明LLM不仅仅是简单地“记住”训练数据，而是在大规模训练后发展出了某种形式的抽象和泛化能力。
    *   **理解涌现能力**对于预测未来LLM的发展方向和发掘其潜力至关重要。

**3\. 对齐 (Alignment) 的多维度考量:**

*   **对齐的“目标”是什么？ (What are the "Goals" of Alignment?)**:
    *   **有用性 (Helpfulness)**：模型应提供准确、相关、有用的信息。
    *   **诚实性 (Honesty/Truthfulness)**：模型应避免生成虚假或误导性信息（减少幻觉）。
    *   **无害性 (Harmlessness)**：模型应避免生成有偏见的、歧视性的、仇恨的或鼓励非法/不道德行为的内容。
*   **指令微调 (Instruction Finetuning) 的作用**:
    *   通过在大量的（指令，期望输出）对上进行微调，LLM学会了理解和遵循各种自然语言指令，这是使其变得更有用的关键一步。它将通用的语言模型“特化”为可以执行特定指令的助手。
*   **RLHF的复杂性与挑战 (Complexity and Challenges of RLHF)**:
    *   **人类反馈的质量和一致性 (Quality and Consistency of Human Feedback)**：奖励模型的好坏高度依赖于人类标注员提供的比较数据。标注员的偏好可能不一致，也可能存在偏见。
    *   **奖励模型的泛化能力 (Generalization of Reward Model)**：奖励模型学到的是对已有反馈的拟合，它能否泛化到新的、未见过的模型输出上，并给出合理的奖励，是一个挑战。
    *   **“对齐税” (Alignment Tax)**：有时，为了使模型更“安全”或更符合特定指令，可能会牺牲其在某些任务上的原始性能或创造力。如何在对齐和能力之间取得平衡是一个重要问题。
    *   **价值的定义与多元化 (Defining and Diversifying Values)**：对齐到谁的价值观？不同文化、不同群体对“有用”、“无害”的定义可能不同。如何处理这种价值多元性是一个开放性问题。

* * *

### L8: LLM Prompting and Alignment (LLM提示与对齐 - 承接L7)

这部分更深入地探讨了提示工程的细节和对齐的重要性。

**1\. 提示学习 (Prompt Learning) 的深化:**

*   **提示的“力量”来源 (Source of Prompt's "Power")**:
    *   **上下文学习 (In-context Learning)**：LLM在预训练阶段接触了海量的文本数据，学习到了丰富的模式和知识。提示通过提供与目标任务相关的上下文（指令、示例），能够激活LLM中已经存在的、与该任务相关的知识和能力。
    *   **任务形式的统一 (Unifying Task Format)**：将不同的下游任务都转化为语言模型擅长的（掩码）填充或文本续写形式，简化了任务适配的过程。
*   **手动提示工程的技巧 (Techniques in Manual Prompt Engineering)**:
    *   **明确性与简洁性 (Clarity and Conciseness)**：提示应该清晰地表达任务要求，避免歧义，同时尽量简洁。
    *   **提供示例 (Providing Examples / Few-shot Prompting)**：对于复杂任务，提供1到几个高质量的输入输出示例，可以极大地提升LLM的性能。示例的质量和多样性很重要。
    *   **角色扮演 (Role Playing)**：例如，“假设你是一位经验丰富的旅行规划师，请帮我规划一个为期一周的东京之旅。”
    *   **逐步引导 (Step-by-step Guidance / Chain-of-Thought)**：对于需要推理的任务，引导模型逐步思考。
    *   **输出格式约束 (Output Format Constraints)**：明确要求模型输出的格式，如JSON、列表、特定结构的文本等。
*   **自动化提示搜索的必要性 (Necessity of Automated Prompt Searching)**:
    *   手动设计最优提示费时费力且依赖经验。自动化提示搜索（无论是离散的还是连续的）旨在通过算法自动找到能最大化模型在特定任务上性能的提示。
    *   **离散提示搜索 (Discrete Prompt Search)**：在人类可理解的自然语言词汇空间中搜索。
        *   挑战：搜索空间巨大，难以高效找到全局最优。
    *   **连续提示搜索 (Continuous Prompt Search / Prompt Tuning / Prefix Tuning)**：在模型的嵌入空间中直接优化提示向量。
        *   优点：可以找到离散空间中不存在的、更优的“软提示”。参数量远小于全模型微调，非常高效。
        *   挑战：学习到的软提示通常不可解释。
*   **P-Tuning v2 的意义**: P-Tuning v2 表明，对于足够大的模型，经过优化的Prompt Tuning可以在各种任务和模型规模上达到与全模型微调相当的性能，同时只调整极少量的参数（通常不到模型总参数的0.1%）。这使得为每个任务定制高效的“软提示”成为可能，而无需存储和部署多个大型微调模型。

**2\. 思维链提示 (Chain of Thought Prompting) 的进一步探讨:**

*   **CoT为何有效？(Why does CoT work?)**:
    *   **分解复杂问题 (Decomposing Complex Problems)**：CoT鼓励模型将复杂问题分解为一系列更小、更易于管理的可解步骤。
    *   **模拟人类思考过程 (Simulating Human Thought Process)**：人类在解决复杂问题时通常也会进行逐步推理。CoT使得LLM的“思考”过程更接近人类。
    *   **提供中间检查点 (Providing Intermediate Checkpoints)**：生成的中间步骤可以作为模型自我纠错或人类审查的依据。
*   **Self-Consistency CoT 的鲁棒性 (Robustness of Self-Consistency CoT)**:
    *   单个CoT路径仍然可能出错。通过采样多个不同的推理路径并对最终答案进行投票，可以显著提高结果的鲁棒性和准确性。这类似于人类通过从不同角度思考或听取不同意见来做出更可靠的决策。
*   **Zero-Shot CoT ("Let's think step by step") 的简洁与强大**:
    *   这一发现非常重要，因为它表明大型语言模型自身已经内隐地具备了逐步推理的能力，只需要一个简单的触发短语就能被激活。这极大地简化了CoT的应用。
*   **Auto-CoT 的自动化探索 (Automated Exploration in Auto-CoT)**:
    *   解决了手动设计CoT示例的痛点。通过聚类相似问题并自动为代表性问题生成CoT演示，使得CoT能够更方便地应用于新的问题集。
*   **Least-to-Most Prompting 的结构化推理 (Structured Reasoning in Least-to-Most Prompting)**:
    *   将问题分解和子问题顺序解决这两个阶段明确分开，使得模型可以更系统地处理依赖前序步骤结果的复杂问题。这是一种更结构化的CoT方法。
*   **Tree of Thoughts (ToT) / Graph of Thoughts (GoT) 的前瞻性**:
    *   这些方法将CoT的线性推理扩展到更复杂的树状或图状探索，允许模型在推理过程中进行更广泛的搜索、回溯和评估，有望解决更具挑战性的规划和推理问题。这代表了LLM推理能力发展的一个重要方向。

**3\. 对齐 (Alignment) 在提示工程中的体现:**

*   即使有了强大的提示技术，如果LLM的底层行为与用户期望不符（例如，不愿意遵循指令、生成有害内容），提示的效果也会大打折扣。
*   RLHF等对齐技术是确保LLM能够“良好地”响应提示的基础。一个经过良好对齐的LLM，即使是简单的提示也能产生有用的输出；而一个未对齐的LLM，可能需要非常复杂和巧妙的提示才能勉强得到期望的结果。
*   因此，提示工程和模型对齐是相辅相成的，共同决定了LLM的最终用户体验和实用性。

* * *

### L9: LLM Agents (LLM智能体)

这部分将LLM的能力从被动的文本处理扩展到主动的与环境交互和工具使用。

**1\. LLM作为智能体核心的意义 (Significance of LLM as Agent Core)**:

*   **赋予LLM“行动”能力 (Empowering LLM with "Action" Capability)**：传统的LLM主要处理文本输入和输出。将其作为智能体的“大脑”，可以使其具备感知环境、做出决策、执行动作（通常是通过生成代码或API调用）的能力，从而与真实或虚拟世界进行交互。
*   **克服LLM固有局限 (Overcoming LLM's Inherent Limitations)**：
    *   **幻觉/知识过时**: 通过工具（如搜索引擎）获取最新、准确的信息。
    *   **复杂计算/精确操作**: 通过工具（如计算器、代码解释器）执行精确计算或符号操作。
    *   **与外部世界交互**: 通过API调用控制其他软件、硬件或服务。
*   **智能体层级的演进 (Evolution of Agent Levels)**:
    *   从简单的文本交互（ELIZA）到LLM驱动的行动（SayCan）再到LLM驱动的推理与行动（ReAct, AutoGPT），反映了智能体自主性和复杂性的不断提升。

**2\. 推理在LLM智能体中的核心作用 (Core Role of Reasoning in LLM Agents)**:

*   **任务分解 (Task Decomposition)**：LLM需要将高级的用户指令分解为一系列可执行的子任务或动作。CoT等推理技术在这里至关重要。
*   **规划 (Planning)**：LLM需要制定一个行动计划，决定动作的顺序和条件。
*   **状态跟踪 (State Tracking)**：在多轮交互或执行复杂任务时，LLM需要跟踪当前状态和已完成的步骤。
*   **工具选择与使用 (Tool Selection and Usage)**：LLM需要判断何时需要使用工具、选择哪个合适的工具、以及如何向工具提供正确的输入并理解其输出。
*   **错误处理与反思 (Error Handling and Reflection)**：当动作执行失败或遇到意外情况时，LLM需要能够分析原因并调整计划。

**3\. 工具学习 (Tool Learning) 的多样性与挑战:**

*   **工具的种类 (Variety of Tools)**：
    *   **信息获取类**: 搜索引擎API、数据库查询、知识图谱API。
    *   **计算与代码执行类**: 计算器、Python解释器、代码生成与执行环境。
    *   **特定领域API**: 如天气查询、航班预订、图像生成模型API等。
    *   **物理世界交互**: 机器人控制API。
*   **学习使用工具的方法 (Methods for Learning to Use Tools)**:
    *   **基于提示 (Prompt-based)**：通过在提示中提供工具的描述和使用示例，让LLM学会调用工具（如BINDER, PAL, ReAct中的部分实现）。这种方法灵活，但对LLM的理解和推理能力要求高。
    *   **微调/预训练 (Fine-tuning/Pre-training)**：专门训练模型生成API调用或工具使用指令（如TALM, Toolformer）。
        *   **TALM的自我博弈 (Self-play in TALM)**：通过迭代地让模型生成工具使用尝试，并根据工具执行结果的好坏来更新模型，实现从少量演示中引导性能提升。
        *   **Toolformer的自监督学习 (Self-supervised learning in Toolformer)**：通过让LLM预测哪些API调用能有效帮助其完成语言建模任务（例如，填补信息空缺、进行计算），从而自监督地学会使用工具。这种方法不需要大量人工标注的工具使用数据。
*   **LLM作为工具制造者 (LLM as a Tool Maker) 的创新性**:
    *   这是一种元学习 (meta-learning) 的思路。与其让LLM直接解决所有重复性的问题，不如让它针对一类问题生成一个可重用的“工具”（如一个Python函数），然后让更轻量级的模型或流程来使用这个工具。这在成本和效率上可能有优势。
*   **挑战 (Challenges in Tool Learning)**:
    *   **工具的发现与选择 (Tool Discovery and Selection)**：当有大量可用工具时，如何让LLM高效地找到并选择最合适的工具？
    *   **API的理解与泛化 (API Understanding and Generalization)**：如何让LLM理解新的、未见过的API文档并学会使用它们？
    *   **组合多个工具 (Composing Multiple Tools)**：如何让LLM有效地组合使用多个工具来完成复杂任务？
    *   **错误处理与鲁棒性 (Error Handling and Robustness)**：当工具调用失败或返回非预期结果时，LLM如何处理？
    *   **工具使用的成本与延迟 (Cost and Latency of Tool Use)**：频繁调用外部工具可能会带来额外的成本和时间延迟。

**4\. 知识整合-RAG在LLM智能体中的应用 (Application of RAG in LLM Agents)**:

*   RAG不仅仅是用于问答系统，它可以被视为一种通用的**为LLM智能体提供外部知识的机制**。
*   当智能体需要执行依赖特定、最新或私有知识的任务时，可以通过RAG检索相关信息，并将其作为上下文提供给LLM进行决策和行动生成。
*   例如，一个需要根据公司内部文档回答问题的客服智能体，或者一个需要参考最新产品信息进行推荐的购物助手智能体，都可以利用RAG。
*   **RAG与工具使用的协同 (Synergy between RAG and Tool Use)**：RAG本身可以被视为一种“知识检索工具”。LLM智能体可以决定何时调用RAG来获取信息，然后基于检索到的信息再决定下一步行动或调用其他工具。

* * *

### L10: Efficient Training of LLMs (LLM的高效训练)

这部分关注训练超大规模语言模型时面临的内存和计算挑战，以及相应的并行和优化技术。

**1\. 内存消耗的细致分析 (Detailed Analysis of Memory Consumption)**:

*   **模型参数 (Model Parameters)**：这是最直观的内存占用。参数量越大，这部分占用越多。fp16比fp32能节省一半。
*   **梯度 (Gradients)**：在反向传播过程中，需要为每个可训练参数计算并存储梯度，其大小与参数本身相当。
*   **优化器状态 (Optimizer States)**：像Adam这样的优化器通常需要存储参数的一阶矩（动量）和二阶矩（方差）的估计，而且通常是以fp32存储以保证更新的精度。这部分内存占用可能是模型参数本身（fp16）的数倍（例如，对于Adam，是 4Ψ+4Ψ\=8Ψ，加上fp32的参数副本是 12Ψ）。
*   **激活值 (Activations)**：在前向传播过程中，为了后续的反向传播计算梯度，需要存储每一层的输出（或某些中间计算结果）。对于深层网络和长序列，激活值的内存占用非常巨大，甚至可能超过参数、梯度和优化器状态的总和。
*   **临时缓冲区 (Temporary Buffers)**：各种计算操作（如梯度聚合、数据通信）可能需要临时的内存空间。
*   **显存碎片 (Memory Fragmentation)**：频繁的内存分配和释放可能导致显存碎片，使得即使总的可用显存足够，也无法分配一块连续的大内存。

**2\. 模型并行 (Model Parallelism) 的不同策略与权衡:**

*   **朴素模型并行 (层间并行 / Pipeline Parallelism - Naïve version)**:
    *   **优点**: 概念简单，易于理解。
    *   **缺点**: “气泡” (bubbles) 导致的GPU利用率低下。第一个GPU完成第一部分的计算后，后续GPU才能开始，造成了大量的等待时间。反向传播时同样存在类似问题。通信开销也可能成为瓶颈。
*   **GPipe / PipeDream (流水线并行优化)**:
    *   **微批次 (Micro-batching)**：将一个大批次数据切分成多个小微批次，并将这些微批次流水线式地送入并行的GPU阶段。
    *   **流水线调度 (Pipeline Scheduling)**：通过精心设计的调度，使得不同GPU上的不同微批次的前向和反向计算能够部分重叠，从而减少气泡，提高GPU利用率。
    *   **挑战**: 需要仔细调整微批次数量和调度策略以达到最优效率。梯度累积 (gradient accumulation) 的方式可能与朴素数据并行略有不同，可能影响收敛动态。
*   **张量并行 (Tensor Parallelism - Megatron-LM)**:
    *   **核心**: 将单个层内的矩阵运算（如全连接层、注意力头）分割到多个GPU上并行执行。
    *   **列并行和行并行**:
        *   对于 Y\=XA，可以将A按列切分到不同GPU，每个GPU计算 XAi​，然后将结果拼接或直接用于后续的行并行操作。
        *   对于 Z\=YB，如果Y是按列分布的（例如，前一步是列并行），那么B可以按行切分，每个GPU计算 Yi​Bi​，然后对结果进行聚合（all-reduce）。
    *   **通信模式**: 张量并行通常需要在层内进行多次通信（如all-reduce, all-gather, scatter）。
    *   **适用场景**: 非常适合在单个节点内（GPU之间有高速NVLink等连接）进行，因为对通信带宽要求很高。跨节点进行张量并行的效率会显著下降。
    *   **与流水线并行的结合**: 通常，对于非常大的模型，会结合使用流水线并行（跨节点或跨多个GPU组）和张量并行（在每个流水线阶段内部的GPU之间）。

**3\. 数据并行 (Data Parallelism) 的演进:**

*   **朴素数据并行 (PyTorch `nn.DataParallel`)**:
    *   **单进程多线程**: 在主GPU上进行数据分发、梯度聚合和参数更新，导致主GPU负载过重，其他GPU利用率不足。
    *   **全局解释器锁 (GIL)**: Python的GIL可能限制多线程的并行效率。
*   **分布式数据并行 (PyTorch `nn.parallel.DistributedDataParallel`, DDP)**:
    *   **多进程**: 每个GPU运行一个独立的进程，拥有自己完整的模型副本和优化器。
    *   **去中心化梯度同步**: 在反向传播过程中，梯度计算完成后，通过高效的通信原语（如ring all-reduce）在所有进程间进行同步和平均。没有中心节点瓶颈。
    *   **更高的效率和可扩展性**: 通常比`DataParallel`更快，并且能更好地扩展到多机多卡环境。
*   **DeepSpeed ZeRO (Zero Redundancy Optimizer)**:
    *   **解决了数据并行的内存冗余问题**: 传统DP中，每个GPU都保存完整的模型参数、梯度和优化器状态的副本，导致内存效率低下。ZeRO通过将这些状态在数据并行的GPU之间进行**分区 (partitioning)**，从而显著减少每个GPU的内存占用。
    *   **ZeRO的三个阶段**:
        *   **Stage 1 (Pos​)**: 只分区优化器状态。参数和梯度仍然是复制的。
        *   **Stage 2 (Pos+g​)**: 分区优化器状态和梯度。参数仍然是复制的。
        *   **Stage 3 (Pos+g+p​)**: 分区优化器状态、梯度和模型参数。这是最节省内存的阶段，每个GPU只负责模型状态的一部分。在前向和反向传播过程中，当需要某个参数时，通过高效的集体通信操作（如all-gather）从持有该参数分片的GPU获取。
    *   **通信开销**: ZeRO-3会引入额外的通信开销（用于收集参数），但通过优化通信调度和利用现代GPU间的高速互联，这种开销通常是可接受的，并且带来的内存节省效益巨大，使得能够训练远超单个GPU内存容量的模型。
    *   **ZeRO-Offload**: 进一步将模型状态（参数、梯度、优化器状态）的一部分卸载 (offload) 到CPU内存或NVMe SSD，以支持更大规模的模型训练，尽管这会牺牲一些训练速度。

**4\. 参数高效微调 (Parameter-Efficient Fine-Tuning, PEFT) 的动机与方法:**

*   **动机 (Motivation)**: 全量微调 (full fine-tuning) 大型预训练模型需要更新所有参数，这在计算资源和存储上成本很高，尤其是在需要为多个下游任务分别微调模型时。PEFT旨在通过只微调模型参数的一小部分（或添加少量额外参数）来达到与全量微调相当的性能。
*   **Adapter Tuning**:
    *   **结构**: 在Transformer的每个（或部分）层的多头注意力和前馈网络之后插入小的、瓶颈状的Adapter模块。这些Adapter模块包含少量可训练参数。
    *   **优点**: 显著减少可训练参数数量，降低存储成本，允许在同一预训练模型基础上快速适配多个任务（每个任务只需存储其对应的Adapter权重）。
    *   **缺点**: 引入了额外的计算层，增加了推理延迟。Adapter的顺序执行特性可能与模型的并行计算优化冲突。
*   **LoRA (Low-Rank Adaptation)**:
    *   **核心思想**: 预训练权重 W0​ 保持冻结。权重的更新量 ΔW 被近似为一个低秩矩阵分解 ΔW\=BA，其中A和B是两个小的可训练矩阵。
    *   **与Adapter的区别**: LoRA的更新是**并行地**添加到原始权重路径上的（h\=W0​x+αBAx，α是缩放因子），而不是顺序插入模块。
    *   **推理效率**: 训练完成后，可以将学习到的 BA 合并到 W0​ 中（W′\=W0​+αBA），从而在推理时**不引入任何额外的计算层或延迟**。这是LoRA相比Adapter的一个显著优势。
    *   **效果**: LoRA在许多任务上用极少的参数就能达到接近甚至超过全量微调的性能。
*   **其他PEFT方法**:
    *   **Prefix Tuning / P-Tuning**: 在输入序列前添加可训练的连续向量（前缀或提示嵌入）。
    *   **BitFit**: 只微调模型的偏置 (bias) 参数。
    *   **(IA)^3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)**: 通过学习三个向量来重新缩放和偏移激活值，参数量极小。

PEFT技术使得在资源相对有限的情况下也能有效地利用和适配超大型预训练模型，极大地推动了LLM在各种下游任务中的应用。

* * *

### L11: Retrieval Augmented Generation (RAG) (检索增强生成)

这部分专注于RAG技术，作为一种有效缓解LLM幻觉、引入外部知识并提高回答准确性的方法。

**1\. RAG的核心价值与动机深化:**

*   **知识的动态性与LLM的静态性矛盾 (Conflict between Dynamic Knowledge and Static LLM)**：LLM的知识存储在其参数中，这些知识在预训练完成后基本是固定的。然而，现实世界的知识是不断更新的。RAG提供了一种在**推理时 (at inference time)** 动态地将最新外部知识注入LLM的方法。
*   **超越参数化知识 (Beyond Parametric Knowledge)**：LLM无法“记住”所有信息，特别是长尾知识或非常具体的细节。RAG允许LLM访问并利用其参数之外的庞大知识源。
*   **可追溯性与可信度 (Traceability and Trustworthiness)**：RAG生成的答案可以追溯到其检索到的源文档，这提高了答案的可信度，并允许用户验证信息的来源。这对于需要高可靠性的应用（如医疗、法律）至关重要。
*   **减少幻觉的机制 (Mechanism for Reducing Hallucination)**：通过将LLM的生成过程“锚定”在检索到的真实上下文上，RAG限制了LLM“自由发挥”的空间，从而减少了凭空捏造信息的可能性。

**2\. RAG架构中各组件的细节与挑战:**

*   **检索器 (Retriever)**:
    *   **稀疏检索 (Sparse Retrieval)**：如BM25, TF-IDF。基于关键词匹配，计算效率高，对于关键词明确的查询效果好。
    *   **密集检索 (Dense Retrieval)**：如DPR (Dense Passage Retriever)。使用双编码器 (bi-encoder) 结构将查询和文档分别编码为密集向量，然后在向量空间中计算相似度（如余弦相似度、点积）。能够更好地捕捉语义相似性，但计算成本较高。
    *   **混合检索 (Hybrid Retrieval)**：结合稀疏和密集检索的优点。
    *   **检索器的训练与微调**: 检索器本身的质量对RAG系统至关重要。可以针对特定领域或任务对检索器进行微调。
*   **数据源 (Data Source) 与索引 (Indexing)**:
    *   **数据预处理与切分 (Data Preprocessing and Chunking)**：如何将原始文档（PDF, HTML, TXT等）有效地切分成大小合适、语义完整的文本块 (chunks) 是一个关键步骤。
        *   **块大小 (Chunk Size)**：太小可能丢失上下文，太大可能引入过多噪声或超出LLM的上下文窗口限制。
        *   **块重叠 (Chunk Overlap)**：在块之间设置一定的重叠，可以避免信息在边界处被切断。
    *   **元数据的重要性 (Importance of Metadata)**：为文本块附加元数据（如来源、日期、章节标题等）可以用于过滤检索结果或为LLM提供更丰富的上下文。
    *   **嵌入模型的选择 (Choice of Embedding Model)**：嵌入模型的质量直接影响检索的准确性。需要选择与查询和文档领域相关的、性能良好的嵌入模型。
    *   **向量数据库的优化 (Optimization of Vector Databases)**：如HNSW, IVFADC等索引结构可以加速大规模向量的相似性搜索。
*   **生成器 (Generator - LLM)**:
    *   **如何有效地将检索到的上下文融入提示 (Effectively Incorporating Retrieved Context into Prompt)**：简单的拼接可能不是最优的。需要设计好的提示模板，引导LLM关注并利用检索到的信息。
    *   **处理多个检索文档 (Handling Multiple Retrieved Documents)**：当检索到多个相关文档时，如何进行排序、选择、融合这些信息是一个挑战。
    *   **处理不相关或噪声文档 (Handling Irrelevant or Noisy Documents)**：如果检索器返回了不相关或包含错误信息的文档，LLM需要有能力识别并忽略它们，或者至少不被误导。
    *   **忠实性与抽象性的权衡 (Faithfulness vs. Abstractiveness)**：RAG的目标是生成忠实于检索内容的答案，但也希望LLM能进行一定程度的综合、推理和抽象，而不是简单地复制粘贴。

**3\. “检索什么/何时检索/如何使用”的深入思考:**

*   **检索什么 (What to retrieve?) - 粒度的权衡**:
    *   **细粒度（如词元、短语）**: 更精确，但可能缺乏上下文。适合需要特定事实片段的任务。
    *   **粗粒度（如段落、文档）**: 上下文更丰富，但可能引入更多噪声。适合需要更全面理解的任务。
    *   **结构化数据（如知识图谱节点/边）**: 可以提供明确的实体和关系信息，但构建和维护成本高。
*   **何时检索 (When to retrieve?) - 静态 vs. 动态/迭代**:
    *   **一次性检索 (Single-shot retrieval)**：在生成开始前检索一次。简单高效，但可能无法应对生成过程中出现的新信息需求。
    *   **迭代/自适应检索 (Iterative/Adaptive retrieval)**：在生成过程中根据需要多次检索。
        *   **FLARE**: 当LLM对当前生成的词元不确定时触发检索。这是一种“按需检索”。
        *   **Self-RAG**: LLM自身学习判断何时需要检索以及检索到的内容是否有用。这赋予了LLM更大的自主性。
        *   **Toolformer**: 将检索视为一种可学习的“工具调用”。
    *   **挑战**: 如何有效地决定何时进行下一次检索？如何避免过多的检索带来的延迟？
*   **如何使用检索到的信息 (How to use retrieved information?) - 融合的艺术**:
    *   **输入层融合 (Input-layer fusion)**：最常见的方式，将检索内容作为LLM输入提示的一部分。
    *   **中间层融合 (Intermediate-layer fusion)**：如RETRO模型，在Transformer的中间层引入检索到的邻近样本信息来影响注意力计算。这种方式理论上可以让检索信息更深层地影响模型表示。
    *   **输出层融合 (Output-layer fusion)**：如kNN-LM，在最终预测词元时，结合检索到的相似上下文的真实后续词元。
    *   **挑战**: 如何设计有效的融合机制，使得检索到的信息能够被LLM充分理解和利用，同时又不过度干扰其自身的生成能力？

**4\. RAG的评估挑战 (Evaluation Challenges for RAG)**:

*   **端到端评估的复杂性**: RAG系统的最终性能取决于检索器和生成器的协同工作。评估时需要区分是检索出了问题还是生成出了问题。
*   **忠实度 (Faithfulness / Groundedness)**：评估生成的答案是否真实地基于检索到的上下文，没有捏造或歪曲。
*   **相关性 (Relevance)**：评估检索到的上下文以及最终生成的答案是否与用户查询相关。
*   **简洁性与全面性的平衡 (Balance between Conciseness and Comprehensiveness)**：答案既要包含足够的信息，又要避免冗余。
*   **无法回答的情况 (Handling Unanswerable Questions)**：当知识库中没有相关信息时，RAG系统应该能够优雅地承认“不知道”，而不是强行生成一个错误的答案。
*   **自动化评估指标的局限性**: 传统NLG评估指标（如BLEU, ROUGE）可能不完全适用于评估RAG的忠实度和知识准确性。需要新的、更针对性的评估方法（如基于LLM的评估、人工评估）。

**5\. RAG的未来趋势与开放问题:**

*   **更智能的检索器 (Smarter Retrievers)**：能够更好地理解查询意图，进行多跳检索，处理复杂查询。
*   **检索与生成的更深度融合 (Deeper Fusion of Retrieval and Generation)**：不仅仅是在输入端拼接，而是在模型的各个层面进行更紧密的交互。
*   **端到端可训练的RAG (End-to-end Trainable RAG)**：同时优化检索器和生成器，使其更好地协同工作。
*   **处理多种模态的RAG (Multimodal RAG)**：检索和生成不仅仅局限于文本，还包括图像、音频、视频等。
*   **个性化RAG (Personalized RAG)**：根据用户的历史、偏好和上下文进行个性化的检索和生成。
*   **RAG与LLM Agent的结合**: LLM Agent将RAG作为其获取外部知识、与环境交互的关键工具之一。

* * *

通过以上对L7到L11的细致补充，我们可以看到大型语言模型从其核心能力（基于规模的涌现、提示学习），到如何使其行为更符合人类期望（对齐技术），再到如何通过与外部工具和知识库交互来扩展其能力边界（LLM Agent, RAG），以及如何高效地训练这些庞然大物，构成了一个逻辑连贯且不断发展的技术图景。这些技术共同推动着自然语言处理乃至整个人工智能领域的前沿。

希望这些补充说明能够帮助您更深入地理解这些重要的概念和技术！

信息来源和相关内容

---

