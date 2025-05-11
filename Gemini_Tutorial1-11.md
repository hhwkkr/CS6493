![IPYNB 图标](https://drive-thirdparty.googleusercontent.com/32/type/text/code)

tutorial\_w...odel.ipynb

IPYNB

用中文详细讲解一下该文件当中的知识点，对于代码你应当讲述整体思路和关键代码

---

显示思路

词嵌入和语言模型
--------

这个 Jupyter Notebook 教程主要介绍了自然语言处理（NLP）中的两个核心概念：**词嵌入 (Word Embeddings)** 和 **语言模型 (Language Models)**。

### 词嵌入 (Word Embeddings)

#### 什么是嵌入 (Embedding)？

嵌入是将信息（如文本、文档、图像、音频等）转换为数值表示的过程。这种表示能够捕捉被嵌入信息的语义含义，使其在许多行业应用中表现稳健。在自然语言处理中，神经网络只能处理整数或浮点数输入，因此自然语言序列必须转换为数值表示序列才能进行进一步计算。这个转换过程通常由**分词器 (tokenizer)** 和 **嵌入层 (embedding layer)** 完成。

#### 1\. 分词 (Tokenization)

分词器负责为模型准备输入。分词是将文本分割成单词或子词（subwords），然后通过查找表将这些单词或子词转换为 ID。

**传统分词 (例如按空格分词) 的缺点：**

*   **词汇量过大：** 例如 "learning", "learned", "learn", "help", "helped", "helping" 会被视为词汇表中的 6 个不同元素。
*   **忽略词形变化与语义的关联：** 例如 "learn", "learned" 和 "learning" 具有相同的核心语义含义，但被视为不同。

**子词分词 (Subword Tokenization) 的解决方案：** 子词分词算法的原则是，常用词不应被分割成更小的子词，而稀有词应被分解成有意义的子词。例如，“annoyingly” 可能被认为是稀有词，并可以分解为 “annoying” 和 “ly”。“annoying” 和 “ly” 作为独立的子词出现的频率更高，同时通过组合它们的含义也保留了 “annoyingly” 的意思。

**常见的分词算法：**

*   字节对编码 (Byte-Pair Encoding - BPE)
*   字节级 BPE (Byte-level BPE)
*   WordPiece
*   Unigram
*   SentencePiece

**代码示例：使用 T5Tokenizer 进行分词** 该代码段演示了如何使用 `transformers` 库中的 `T5Tokenizer` 对句子进行分词，并将分词后的词元 (tokens) 转换为对应的 ID。

**整体思路：**

1.  导入 `T5Tokenizer`。
2.  从预训练模型 "t5-base" 加载分词器。
3.  使用分词器的 `tokenize` 方法对输入句子进行分词。
4.  使用分词器的 `convert_tokens_to_ids` 方法将词元转换为 ID。

**关键代码：**

Python

    from transformers import T5Tokenizer
    
    # 从预训练模型 "t5-base" 加载分词器
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    
    # 对句子进行分词
    tokenized_sequence = tokenizer.tokenize("I am a student from City University of Hong Kong, I am learning the NLP.")
    print(tokenized_sequence)
    
    # 将词元转换为 ID
    ids = tokenizer.convert_tokens_to_ids(tokenized_sequence)
    print(ids)

**输出：**

[' I', ' am', ' ', 'a', ' student', ' from', ' City', ' University', ' of', ' Hong', ' Kong', ',', ' I', ' am', ' learning', ' the', ' N', 'LP', '.']
    [27, 183, 3, 9, 1236, 45, 896, 636, 13, 7764, 7807, 6, 27, 183, 1036, 8, 445, 6892, 5]

这显示了句子是如何被分解成子词，并且每个子词都有一个唯一的数字 ID。

#### 2\. 嵌入层 (Embedding Layer)

嵌入层将低维的“ID”转换为高维的数值表示，这种高维表示可以用来计算不同 ID 之间的相似性或距离。语言模型中的嵌入层实际上是一个线性层。

**代码示例：从预训练模型 `distilgpt2` 提取并使用嵌入层** 此部分代码展示了如何：

1.  加载预训练的因果语言模型 `distilgpt2`。
2.  从模型中提取嵌入层。
3.  将之前获得的分词 ID 转换为 PyTorch 张量。
4.  使用嵌入层将低维的 ID 张量转换为高维的词嵌入表示。

**整体思路：**

1.  导入所需库 (`torch`, `AutoModelForCausalLM`)。
2.  加载预训练模型 `distilgpt2`。
3.  通过 `model.transformer.wte` 获取模型的词嵌入层 (word token embeddings)。
4.  将 Python 列表形式的 `ids` 转换为 PyTorch 张量 `tensor_ids`。
5.  将 `tensor_ids` 输入嵌入层，得到高维的 `token_embedding`。

**关键代码：**

Python

    import torch
    from transformers import AutoModelForCausalLM
    
    # 加载预训练模型
    model = AutoModelForCausalLM.from_pretrained('distilgpt2')
    
    # 提取嵌入层
    embedding_layer = model.transformer.wte
    
    # 将 ID 列表转换为 PyTorch 张量
    # 假设 ids 是之前 T5Tokenizer 输出的 ID 列表：
    # ids = [27, 183, 3, 9, 1236, 45, 896, 636, 13, 7764, 7807, 6, 27, 183, 1036, 8, 445, 6892, 5]
    tensor_ids = torch.tensor(ids)
    print('Tensor ids:', tensor_ids)
    print('Shape of the tensor: ', tensor_ids.shape)
    
    # 使用嵌入层进行转换
    token_embedding = embedding_layer(tensor_ids)
    print('Embedding of the tensor ids: \\n', token_embedding)
    print('Shape of the embedding: ', token_embedding.shape)
    print(embedding_layer)

**输出解释：**

*   `Tensor ids` 显示了 ID 列表转换后的张量形式及其形状 (例如 `torch.Size([19])`，表示有 19 个词元)。
*   `Embedding of the tensor ids` 显示了每个 ID 对应的高维嵌入向量。其形状 (例如 `torch.Size([19, 768])`) 表示对于 19 个词元，每个词元都被映射成了一个 768 维的向量。
*   `Embedding(50257, 768)` 表示这个嵌入层包含一个大小为 50257x768 的权重矩阵，其中 50257 是词汇表的大小，768 是嵌入向量的维度。

**嵌入层的本质：** 笔记本进一步解释了嵌入层实际上是一个没有偏置项的线性层，它将独热向量 (one-hot vectors) 作为输入。因此，这个线性层可以被用作一个查找表。

**代码示例：使用线性层模拟嵌入层** 这部分代码通过创建一个与原嵌入层权重相同的线性层，并输入独热编码后的 `tensor_ids`，来验证其输出与直接使用嵌入层 `embedding_layer(tensor_ids)` 的输出是相同的。

**整体思路：**

1.  导入 `torch.nn` 和 `torch.nn.functional`。
2.  创建一个没有偏置的线性层 `lookup_table_linear`，其输入维度等于嵌入维度 (例如 768)，输出维度等于词汇表大小 (例如 50257)。
3.  将其权重设置为原嵌入层权重的转置。
4.  将 `tensor_ids` 进行独热编码。
5.  将独热编码后的张量输入到这个线性层中。

**关键代码：**

Python

    import torch.nn as nn
    from torch.nn.functional import one_hot
    
    # 假设 embedding_layer 和 tensor_ids 已按前述代码定义
    # embedding_dim = 768 (distilgpt2 的嵌入维度)
    # vocab_size = 50257 (distilgpt2 的词汇表大小)
    
    lookup_table_linear = nn.Linear(embedding_layer.weight.shape[0], embedding_layer.weight.shape[1], bias = False) # 注意这里维度调整以匹配权重
    lookup_table_linear.weight = nn.Parameter(embedding_layer.weight) # 直接使用权重，因为 PyTorch 的 Embedding 层权重就是 (num_embeddings, embedding_dim)
    
    # 如果要严格模拟 lookup table 的输入是 one-hot，然后通过矩阵乘法得到 embedding
    # 则线性层的权重应该是 embedding 矩阵的转置，输入 one-hot，输出 embedding
    # lookup_table_as_linear_for_one_hot = nn.Linear(vocab_size, embedding_dim, bias=False)
    # lookup_table_as_linear_for_one_hot.weight = nn.Parameter(embedding_layer.weight.t()) # 转置权重
    
    # 将原始 ID 进行独热编码
    one_hot_tensor_ids = one_hot(tensor_ids, num_classes = embedding_layer.weight.shape[0]).float() # num_classes 应该是词汇表大小
    
    # 使用线性层进行编码 (这里演示的是使用转置权重的线性层处理 one-hot 输入)
    # 注意：原始 notebook 的 lookup_table_linear(one_hot_tensor_ids) 是错误的，
    # 因为 one_hot_tensor_ids 的维度是 [seq_len, vocab_size]，而 lookup_table_linear 期望的输入是 [seq_len, embedding_dim]
    # 正确的模拟应该是：
    # output_from_linear_with_one_hot = lookup_table_as_linear_for_one_hot(one_hot_tensor_ids)
    # print(output_from_linear_with_one_hot)
    
    # 或者，更直接地，torch.nn.Embedding 本身就是一种查找表操作，其内部实现高效。
    # 上述代码中 token_embedding = embedding_layer(tensor_ids) 已经是正确的用法。
    # notebook 中试图用 nn.Linear 解释 nn.Embedding 的查找机制。
    # 如果将 embedding_layer.weight (形状为 vocab_size x embedding_dim) 视为查找表，
    # 那么对于一个 ID，就是从这个表中取出对应的那一行向量。
    # 如果用 nn.Linear 实现，且输入是 one-hot 向量 (长度为 vocab_size)，
    # 那么线性层的权重矩阵 W (形状为 embedding_dim x vocab_size) 乘以 one-hot 向量，
    # 结果是一个长度为 embedding_dim 的向量。此时 W 就是 embedding_layer.weight 的转置。
    
    # 根据 notebook 的意图，它可能想表达的是 embedding_layer.weight 本身就是那个查找表
    # 且 embedding_layer(tensor_id_scalar) 等价于 embedding_layer.weight[tensor_id_scalar]
    
    # 修正 notebook 中不清晰的部分，直接展示 PyTorch Embedding 层的效果已在前一部分完成。
    # notebook 中关于 nn.Linear(768, 50527, bias = False) 和 lookup_table_linear.weight = nn.Parameter(embedding_layer.weight.t())
    # 的部分，如果输入是 one-hot 向量 (shape: batch_size, vocab_size)，那么输出的维度将会是 (batch_size, 768)，
    # 这与 embedding_layer 的输出维度一致。
    # lookup_table_linear.weight (vocab_size, embedding_dim)
    # (one_hot_tensor_ids (seq_len, vocab_size) @ lookup_table_linear.weight (vocab_size, embedding_dim))
    # -> (seq_len, embedding_dim)
    # 这等价于 F.embedding(tensor_ids, embedding_layer.weight)
    # nn.Linear(in_features, out_features) 的 weight 是 (out_features, in_features)
    # 所以如果 in_features=vocab_size, out_features=embedding_dim,
    # weight 就是 (embedding_dim, vocab_size), 这正是 embedding_layer.weight.t()
    
    # 假设 ids 列表为:
    ids = [27, 183, 3, 9, 1236, 45, 896, 636, 13, 7764, 7807, 6, 27, 183, 1036, 8, 445, 6892, 5]
    tensor_ids = torch.tensor(ids)
    embedding_layer_distilgpt2 = AutoModelForCausalLM.from_pretrained('distilgpt2').transformer.wte
    vocab_size = embedding_layer_distilgpt2.weight.shape[0] # 50257
    embedding_dim = embedding_layer_distilgpt2.weight.shape[1] # 768
    
    one_hot_tensor_ids_for_linear = one_hot(tensor_ids, num_classes=vocab_size).float()
    linear_equivalent = nn.Linear(vocab_size, embedding_dim, bias=False)
    linear_equivalent.weight = nn.Parameter(embedding_layer_distilgpt2.weight.t()) # 权重设置为 embedding 权重的转置
    
    # 通过线性层获得嵌入
    embedding_from_linear = linear_equivalent(one_hot_tensor_ids_for_linear)
    # print(embedding_from_linear) # 输出应与 embedding_layer_distilgpt2(tensor_ids) 相同

**结论：** 语言模型中的嵌入层实际上是一个用无偏置的线性层构建的查找表。训练嵌入层与训练线性层完全相同。

### 语言模型 (Language Models)

教程中提到了两种主要的语言模型类型：

#### 1\. 因果语言模型 (Causal Language Modeling - CLM)

*   **目标：** 模型必须预测句子中的下一个词元（因此标签与向右移动一位的输入相同）。
*   **机制：** 为了确保模型不作弊，它会获得一个注意力掩码 (attention mask)，该掩码会阻止模型在尝试预测句子中第 `i+1` 个词元时访问第 `i` 个词元之后的词元。
*   **示例：** [Hugging Face DistilGPT-2 演示](https://www.google.com/search?q=https://huggingface.co/distilgpt2%3Ftext%3DMy%2Bname%2Bis%2BSylvain%2Band%2BI%2Blike%2Bto)
    *   教程中包含一张图片，形象地展示了因果语言模型的结构，其中当前词的预测仅依赖于其前面的词。

#### 2\. 双向语言模型 (Bidirectional Language Modeling) / 掩码语言模型 (Masked Language Modeling - MLM)

*   **目标：** 模型必须预测输入中被掩码 (masked) 的一些词元。
*   **机制：** 模型仍然可以访问整个句子，因此它可以使用被掩码词元之前和之后的词元来预测它们的值。
*   **示例：** [Hugging Face DistilRoBERTa-base 演示](https://www.google.com/search?q=https://huggingface.co/distilroberta-base%3Ftext%3DThe%2B%253Cmask%253E%2Bof%2Blife%2Bis%2Bhappiness.)
    *   教程中包含一张图片，展示了掩码语言模型的结构，其中被掩码词的预测可以同时利用其上下文信息。

#### 准备数据集

教程接下来准备使用 **Wikitext 2** 数据集来演示如何为这两种语言模型任务加载和预处理数据，以及如何使用 Hugging Face `Trainer` API 对模型进行微调。

**代码示例：加载 Wikitext 2 数据集**

Python

    from datasets import load_dataset
    datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')

这行代码使用 `datasets` 库加载了 `wikitext-2-raw-v1` 数据集。

**代码示例：展示数据集中的随机元素**

Python

    from datasets import ClassLabel
    import random
    import pandas as pd
    from IPython.display import display, HTML
    
    def show_random_elements(dataset, num_examples=10):
        # ... (函数实现如 notebook 所示) ...
        pass
    
    show_random_elements(datasets["train"])

这个函数会随机选择数据集中的一些样本并以 HTML 表格的形式展示出来，帮助理解数据格式。一些文本是完整的维基百科段落，而另一些只是标题或空行。

#### 因果语言模型 (CLM) 的数据预处理和训练

**模型选择：** `distilgpt2`

**分词和数据分组：**

1.  **加载分词器：** 使用 `AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)` 加载与模型对应的分词器。
2.  **定义分词函数：** `tokenize_function` 将文本输入分词器。
3.  **应用分词：** 使用 `datasets.map` 对所有文本进行分词，并移除原始的 "text" 列。
4.  **定义文本分组函数 `group_texts`：**
    *   将所有文本拼接起来。
    *   将拼接后的文本按固定的 `block_size` (例如 128) 切分成块。
    *   丢弃最后不足一个 `block_size` 的剩余部分。
    *   将 `input_ids` 复制一份作为 `labels`（因为在因果语言模型中，模型的目标是预测下一个词，Hugging Face 的模型会自动处理标签的右移）。
5.  **应用文本分组：** 使用 `tokenized_datasets.map` 应用 `group_texts` 函数，将数据集转换为适合因果语言模型训练的格式。

**关键代码 (数据预处理部分)：**

Python

    model_checkpoint = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"])
    
    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    
    block_size = 128 # 或者 tokenizer.model_max_length
    
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )

**模型训练：**

1.  **加载模型：** `AutoModelForCausalLM.from_pretrained(model_checkpoint)`
2.  **定义训练参数：** 使用 `TrainingArguments` 配置训练过程，如输出目录、评估策略、学习率、权重衰减、训练轮数等。
3.  **实例化 Trainer：** 将模型、训练参数、训练数据集和评估数据集传递给 `Trainer` 类。
4.  **开始训练：** 调用 `trainer.train()`。
5.  **评估模型：** 调用 `trainer.evaluate()`，并计算困惑度 (Perplexity)。

**关键代码 (训练部分)：**

Python

    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
    import math
    
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    
    model_name = model_checkpoint.split("/")[-1]
    training_args = TrainingArguments(
        f"{model_name}-finetuned-wikitext2",
        evaluation_strategy = "epoch", # notebook 中可能是 eval_strategy
        learning_rate=2e-5,
        weight_decay=0.01,
        # push_to_hub=False, # 根据实际情况设定
        num_train_epochs = 1, # notebook 中可能训练更多轮
        report_to="none" # 避免 wandb 等报告
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
    )
    
    trainer.train()
    
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

训练完成后，输出的困惑度 (例如 40.01) 是衡量语言模型性能的一个指标，值越低表示模型对序列的预测能力越好。

#### 掩码语言模型 (MLM) 的数据预处理和训练

**模型选择：** `distilroberta-base`

**分词和数据分组：** 与 CLM 类似，首先进行分词和文本分组。关键区别在于数据整理器 (data collator)。

**数据整理器 (Data Collator)：** 对于 MLM，需要一个特殊的数据整理器，它会在每个批次中随机地掩码一部分词元。

*   使用 `DataCollatorForLanguageModeling`。
*   可以设置 `mlm_probability` (例如 0.15)，表示掩码的词元比例。
*   这个整理器确保每次迭代数据时，掩码的方式都是随机的，而不是固定的。

**关键代码 (数据整理器和训练部分)：**

Python

    from transformers import AutoModelForMaskedLM, DataCollatorForLanguageModeling
    
    model_checkpoint_mlm = "distilroberta-base"
    tokenizer_mlm = AutoTokenizer.from_pretrained(model_checkpoint_mlm, use_fast=True)
    
    # 假设 tokenized_datasets 和 lm_datasets 已经使用新的 tokenizer_mlm 和对应的模型进行了类似的预处理
    
    model_mlm = AutoModelForMaskedLM.from_pretrained(model_checkpoint_mlm)
    
    model_name_mlm = model_checkpoint_mlm.split("/")[-1]
    training_args_mlm = TrainingArguments(
        f"{model_name_mlm}-finetuned-wikitext2",
        evaluation_strategy = "epoch", # notebook 中可能是 eval_strategy
        learning_rate=2e-5,
        weight_decay=0.01,
        # push_to_hub=False,
        report_to="none",
        # num_train_epochs 同样可以设置
    )
    
    data_collator_mlm = DataCollatorForLanguageModeling(tokenizer=tokenizer_mlm, mlm_probability=0.15)
    
    trainer_mlm = Trainer(
        model=model_mlm,
        args=training_args_mlm,
        train_dataset=lm_datasets["train"], # 注意：这里应使用为 MLM tokenizer 处理过的数据集
        eval_dataset=lm_datasets["validation"], # 注意：这里应使用为 MLM tokenizer 处理过的数据集
        data_collator=data_collator_mlm,
    )
    
    # trainer_mlm.train() # 实际训练过程
    
    # eval_results_mlm = trainer_mlm.evaluate()
    # print(f"Perplexity: {math.exp(eval_results_mlm['eval_loss']):.2f}")

对于 MLM，困惑度的计算只针对被掩码的词元进行预测，因此通常会比 CLM 的困惑度低很多，因为模型可以利用上下文信息。

这个教程通过实际代码演示了词嵌入的原理以及两种主要语言模型的训练流程，是理解和实践 NLP 基础模型的好起点。

信息来源和相关内容

---

![IPYNB 图标](https://drive-thirdparty.googleusercontent.com/32/type/text/code)

tutorial\_w...ng\_2.ipynb

IPYNB

很棒，请你接着讲述下面tutorial，要求同之前相同

---

显示思路

好的，我们来继续讲解这个关于预训练词嵌入的 Jupyter Notebook 教程。

这个教程主要聚焦于**预训练词嵌入 (Pre-trained Word Embeddings)**，解释了为什么需要它们，以及几种流行的预训练词嵌入模型，包括 Word2Vec, GloVe, 和 BERT 嵌入。

### 预训练词嵌入 (Pre-trained Word Embeddings) - 第二部分

#### 词嵌入回顾 (Word Embeddings)

教程首先简单回顾了词嵌入的概念：

*   词嵌入是将原始文本转换为数字/向量的一种自然语言处理方法。
*   深度学习模型只能接受数值输入，因此这项技术对于处理原始数据非常重要。
*   它有助于捕捉词语的语义含义和上下文。每个词语都由一个具有不同维度的实值向量表示。
*   早期的词嵌入方法（如词袋模型 BOW、TF-IDF）只转换词语而不提取语义关系和上下文。而近期的模型（如 BERT 嵌入）能够捕捉词语在句子中的完整上下文以及语义关系。

#### 从零开始构建词嵌入的挑战 (Challenges in building word embedding from scratch)

从头开始训练词嵌入是可能的，但由于以下原因具有挑战性：

*   **大量的可训练参数：** 词嵌入模型通常参数众多。
*   **训练数据的稀疏性：** 需要在包含丰富词汇的大量数据集上进行训练。
*   **训练速度慢：** 由于参数量大，训练过程会很缓慢。 因此，在个人层面从头训练一个高质量的词嵌入模型是相当困难的。

#### 预训练词嵌入 (Pre Trained Word Embeddings)

预训练词嵌入为上述问题提供了解决方案。它们是在大型数据集上预先训练好的词嵌入模型，能够捕捉词语的句法和语义含义。这种技术被称为**迁移学习 (transfer learning)**，即将在大型数据集上训练好的模型应用于自己相似的任务。

预训练词嵌入大致可以分为两类：

*   **词级嵌入 (word-level embeddings)**
*   **字符级嵌入 (character-level embeddings)**

本教程主要关注两种词级嵌入：**Word2Vec** 和 **GloVe**。

#### Word2Vec

Word2Vec 是谷歌开发的最流行的预训练词嵌入之一。它在“Google News”这个大规模数据集上进行了训练。顾名思义，它用一个称为向量的整数集合来表示每个词。这些向量的计算方式能够体现词语之间的语义关系。

一个著名的例子是国王-王后示例： `King - Man + Woman ≈ Queen` (国王 - 男性 + 女性 ≈ 王后)

Word2Vec 是一个前馈神经网络，包含两个主要模型：

1.  **连续词袋模型 (Continuous Bag-of-Words - CBOW):** CBOW 模型从相邻词（上下文词）学习目标词。
2.  **Skip-gram 模型:** Skip-gram 模型从目标词学习相邻词（上下文词）。

这两个模型的工作方式正好相反。

**核心概念：**

*   **上下文窗口 (Context window):** 一个滑动窗口，一次一个词地遍历整个文本。它指的是出现在目标词左右两侧的词语数量。例如，如果上下文窗口大小设置为 2，则它将包括目标词左右各 2 个词。
*   **焦点词/目标词 (Focus word/Target word):** 我们想要为其创建嵌入/向量表示的词。通常是上下文窗口的中心词。
*   **邻近词 (Neighbouring words):** 出现在上下文窗口中的词，有助于捕捉整个句子的上下文。

**示例：** 句子 "He poured himself a cup of coffee"，目标词是 "himself"。

*   **CBOW 输入/输出:**
    *   输入: `[“He”, “poured”, “a”, “cup”]`
    *   输出: `[“himself”]`
*   **Skip-gram 输入/输出:**
    *   输入: `[“himself”]`
    *   输出: `[“He”, “poured”, “a”, “cup”]`

**代码示例：使用 Gensim 加载和使用预训练的 Word2Vec 模型** 此代码段演示了如何使用 `gensim` 库加载预训练的 Word2Vec 模型（GoogleNews-vectors-negative300.bin.gz 文件），并计算词对之间的余弦相似度，以及找出与给定词最相似的词。

**整体思路：**

1.  安装并导入 `gensim` 库。
2.  挂载 Google Drive 以访问预训练模型文件（如果模型存储在 Drive 中）。
3.  使用 `KeyedVectors.load_word2vec_format()` 方法加载预训练的 Word2Vec 模型。
4.  使用模型的 `similarity()` 方法计算词对之间的余弦相似度。
5.  使用模型的 `most_similar()` 方法查找与给定词最相似的词。

**关键代码：**

Python

    # 安装 gensim (如果尚未安装)
    # !pip install gensim
    # from google.colab import drive
    # drive.mount('/content/drive') # 如果模型在 Google Drive
    
    from gensim.models import KeyedVectors
    
    # 替换为你的模型文件路径
    pretrained_model_path = '/content/drive/MyDrive/tutorial_data/GoogleNews-vectors-negative300.bin.gz'
    
    # 加载预训练模型
    word_vectors = KeyedVectors.load_word2vec_format(pretrained_model_path, binary=True)
    
    # 计算词对之间的余弦相似度
    word1 = "early"
    word2 = "seats"
    similarity1 = word_vectors.similarity(word1, word2)
    print(similarity1)
    
    word3 = "king"
    word4 = "man"
    similarity2 = word_vectors.similarity(word3, word4)
    print(similarity2)
    
    # 查找与 "King" 最相似的词
    king_similar_words = word_vectors.most_similar('King')
    print(f'Top 10 most similar words to "King" are : {king_similar_words}')

**输出解释：**

*   `similarity1` (例如 `0.0358`) 和 `similarity2` (例如 `0.2294`) 表示对应词对的余弦相似度。余弦相似度的值域为 -1 到 1，值越大表示相似度越高。因此，"king" 和 "man" 比 "early" 和 "seats" 更相似。
*   `king_similar_words` 会输出与 "King" 最相似的前 10 个词及其相似度得分，例如 `[('Jackson', 0.5326), ('Prince', 0.5306), ...]`。

#### GloVe (Global Vectors for Word Representation)

GloVe 是斯坦福大学提出的另一种流行的词嵌入模型。它的基本思想是利用统计数据推导词语之间的关系。它是一个基于计数并利用**共现矩阵 (co-occurrence matrix)** 的模型。共现矩阵记录了两个词共同出现的频率。

GloVe 模型处理的是词向量空间，其中词与词之间的距离与其语义相似度相关。它结合了全局矩阵分解和局部上下文窗口技术的特性。模型的训练基于语料库中的全局词-词共现数据，生成的表示形式会在向量空间中产生线性子结构。

GloVe 计算每个词对的共现概率。教程中给出了一个公式 F(wi​,wj​,wk​)\=Pjk​Pik​​，但这个公式通常用于描述 GloVe 优化目标函数的推导过程中的比例关系，而不是直接计算共现概率。更准确地说，GloVe模型试图学习向量，使得它们的点积与它们共现次数的对数相关。

**共现概率示例：** （教程中提供的这个例子比较简单，用于说明概念） 句子：“cat chases mouse” 对于词对 "cat" 和 "mouse"： 共现概率("cat", "mouse") = "cat" 和 "mouse" 的共现次数 / "cat" 的总共现次数

*   Count("cat" and "mouse") = 1
*   Total Co-occurrences("cat") = 2 (与 "chases" 和 "mouse")
*   所以, Co-occurrence Probability("cat", "mouse") = 1 / 2 = 0.5

**代码示例：加载和使用预训练的 GloVe 模型** 此代码段展示了如何加载预训练的 GloVe 嵌入（例如 glove.6B.50d.txt 文件，包含在 6B 词汇上训练的 50 维向量），并使用它们来查找与目标词最相似的词以及计算词对之间的余弦相似度。

**整体思路：**

1.  安装 `tensorflow` (虽然代码中主要使用 numpy 和 scipy，但 tensorflow 可能是一些NLP库的依赖)。
2.  定义 `load_glove_embeddings` 函数来从文本文件中加载 GloVe 嵌入。该函数逐行读取文件，每行包含一个词和其对应的向量值，将它们存储在一个字典中。
3.  定义 `find_top_n_nearest_words` 函数，使用余弦相似度查找与目标词最接近的 N 个词。
4.  定义 `cosine_similarity` 函数计算两个向量的余弦相似度。
5.  （可选但推荐）定义 `normalize_vectors` 函数对嵌入向量进行归一化，这在计算余弦相似度时有时能提高性能或稳定性。
6.  加载 GloVe 文件并进行向量归一化（如果需要）。
7.  演示如何获取特定词的嵌入向量，查找最近似的词，以及计算词对之间的相似度。

**关键代码：**

Python

    import numpy as np
    from scipy.spatial.distance import cosine # 用于计算余弦距离，相似度 = 1 - 距离
    
    # 加载 GloVe 嵌入的函数
    def load_glove_embeddings(file_path):
        embeddings = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                embeddings[word] = vector
        return embeddings
    
    # 查找最相似的 N 个词的函数
    def find_top_n_nearest_words(target_word, embeddings, n=5):
        if target_word not in embeddings:
            print(f"'{target_word}' not found in the vocabulary.")
            return None
        target_vector = embeddings[target_word]
        similarities = []
        for word, vector in embeddings.items():
            if word == target_word:
                continue
            similarity = 1 - cosine(target_vector, vector) # 余弦相似度
            similarities.append((word, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]
    
    # 计算余弦相似度的函数 (更直接的方式)
    def direct_cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0: # 避免除以零
            return 0.0
        return dot_product / (norm_vec1 * norm_vec2)
    
    # （可选）向量归一化函数
    def normalize_vectors(embeddings):
        for word in embeddings:
            norm = np.linalg.norm(embeddings[word])
            if norm > 0: # 避免除以零
                 embeddings[word] /= norm
        return embeddings
    
    # 示例用法
    # 假设模型文件已上传到 Google Drive 或本地路径
    file_path = '/content/drive/MyDrive/tutorial_data/glove.6B.50d.txt'
    embeddings = load_glove_embeddings(file_path)
    # embeddings = normalize_vectors(embeddings) # notebook 中演示了归一化
    
    # 获取 "king" 的嵌入
    # print(embeddings['king'])
    
    # 查找与 "king" 最相似的词
    target_word = "king"
    top_n_words = find_top_n_nearest_words(target_word, embeddings, n=10)
    if top_n_words:
        print(f"Top {len(top_n_words)} nearest words to '{target_word}':")
        for word, similarity in top_n_words:
            print(f"{word}: {similarity:.4f}")
    
    # 计算 "king" 和 "queen" 的相似度
    word1 = "king"
    word2 = "queen"
    if word1 in embeddings and word2 in embeddings:
        vec1 = embeddings[word1]
        vec2 = embeddings[word2]
        similarity_kq = direct_cosine_similarity(vec1, vec2) # 使用直接计算相似度的函数
        print(f"Cosine similarity between '{word1}' and '{word2}': {similarity_kq:.4f}")
    
    # 计算 "cat" 和 "volcano" 的相似度
    word1_cv = "cat"
    word2_cv = "volcano"
    if word1_cv in embeddings and word2_cv in embeddings:
        vec1_cv = embeddings[word1_cv]
        vec2_cv = embeddings[word2_cv]
        similarity_cv = direct_cosine_similarity(vec1_cv, vec2_cv)
        print(f"Cosine similarity between '{word1_cv}' and '{word2_cv}': {similarity_cv:.4f}")

**输出解释：**

*   查找与 "king" 最相似的词会输出如 `prince: 0.8236`, `queen: 0.7839` 等，显示了语义上相近的词。
*   "king" 和 "queen" 的余弦相似度较高 (例如 `0.7839`)，而 "cat" 和 "volcano" 的相似度较低 (例如 `0.2669`)，这符合直观感受。

#### BERT 嵌入 (BERT Embeddings)

BERT (Bidirectional Encoder Representations from Transformers) 是谷歌提出的另一个重要的基于 Transformer 的预训练模型。它可以用于从原始文本中提取高质量的语言特征，或者可以在特定任务上进行微调。

**BERT 架构：**

*   仅包含编码器 (encoders)。
*   输入是词元序列，包括：
    *   **词元嵌入 (Token embeddings)**
    *   **段嵌入 (Segment embeddings)** (用于区分句子对)
    *   **位置嵌入 (Positional embeddings)** (用于编码词元在序列中的位置)
*   **主要思想：** 在句子中掩码一些词，并让模型预测这些被掩码的词。

**代码示例：使用 Transformers 库获取 BERT 嵌入并计算相似度** 此代码段演示了如何使用 `transformers` 库中的 `BertTokenizer` 和 `BertModel` 来获取句子中特定单词的 BERT 嵌入，并计算它们之间的余弦相似度。

**整体思路：**

1.  安装 `transformers` 库。
2.  导入 `torch`, `BertTokenizer`, `BertModel`。
3.  加载预训练的 `bert-base-uncased` 分词器和模型。在加载模型时，设置 `output_hidden_states=True` 以获取所有隐藏层的输出。
4.  定义 `get_bert_embeddings` 函数：
    *   在输入文本前后添加特殊标记 `[CLS]` 和 `[SEP]`。
    *   使用分词器对文本进行分词，并转换为 ID。
    *   创建段 ID (对于单句，所有段 ID 都为 1)。
    *   将 token ID 和段 ID 转换为 PyTorch 张量。
    *   将张量输入 BERT 模型，获取输出。
    *   从输出中提取隐藏状态。通常做法是取最后几层（例如最后 4 层）的隐藏状态的平均值或拼接作为词嵌入。本教程中是取最后四层隐藏状态的平均值。
    *   找到目标词在分词后序列中的索引。
    *   提取目标词的嵌入向量。
5.  定义 `cosine_similarity` 函数（与 GloVe 部分类似）。
6.  演示如何获取句子中 "king" 和 "queen" 的嵌入，并计算它们的相似度。
7.  演示如何获取不同句子中 "king" 和 "apple" 的嵌入，并计算它们的相似度。

**关键代码：**

Python

    import torch
    from transformers import BertTokenizer, BertModel
    from scipy.spatial.distance import cosine # 重复导入，但无害
    import numpy as np # 重复导入
    
    # 加载预训练 BERT 模型和分词器
    tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
    model_bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    
    # 获取 BERT 嵌入的函数
    def get_bert_embeddings(text, target_word):
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = tokenizer_bert.tokenize(marked_text)
        indexed_tokens = tokenizer_bert.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
    
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensor = torch.tensor([segments_ids])
    
        model_bert.eval() # 将模型设置为评估模式
        with torch.no_grad():
            outputs = model_bert(tokens_tensor, segments_tensor)
            hidden_states = outputs.hidden_states
    
        token_embeddings = torch.stack(hidden_states[-4:], dim=0)
        token_embeddings = torch.mean(token_embeddings, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=0)
    
        try:
            # 需要处理目标词可能被分解为多个子词的情况
            # 一个简单的处理方法是找到第一个子词的索引
            target_word_tokens = tokenizer_bert.tokenize(target_word)
            target_index = -1
            for i in range(len(tokenized_text) - len(target_word_tokens) + 1):
                if tokenized_text[i:i+len(target_word_tokens)] == target_word_tokens:
                    target_index = i
                    break
            if target_index == -1 and target_word_tokens[0] in tokenized_text : # 简化处理，取第一个子词
                 target_index = tokenized_text.index(target_word_tokens[0])
    
            if target_index == -1:
                 print(f"'{target_word}' (or its subwords) not found in the tokenized text: {tokenized_text}")
                 return None
    
            # 如果目标词被分解为多个子词，可以取这些子词嵌入的平均值
            if len(target_word_tokens) > 1 and target_index != -1:
                 target_embedding_tensor = torch.mean(token_embeddings[target_index : target_index + len(target_word_tokens)], dim=0)
            elif target_index != -1:
                 target_embedding_tensor = token_embeddings[target_index]
            else: # 应该不会到这里如果上面逻辑正确
                 return None
            target_embedding = target_embedding_tensor.numpy()
    
    
        except ValueError: # 如果 target_word 本身就是单个 token 且未找到
            print(f"'{target_word}' not found in the tokenized text: {tokenized_text}")
            return None
        return target_embedding
    
    # 余弦相似度函数 (与之前 GloVe 部分相同)
    def cosine_similarity_bert(vec1, vec2):
        if vec1 is None or vec2 is None:
            return 0.0 # 或者其他错误处理
        return 1 - cosine(vec1, vec2)
    
    
    # 示例1: "king" 和 "queen" 在同一上下文中
    text_kq = "The king and queen ruled the kingdom."
    word1_kq_bert = "king"
    word2_kq_bert = "queen"
    embedding_king_bert = get_bert_embeddings(text_kq, word1_kq_bert)
    embedding_queen_bert = get_bert_embeddings(text_kq, word2_kq_bert)
    similarity_kq_bert = cosine_similarity_bert(embedding_king_bert, embedding_queen_bert)
    print(f"Cosine similarity between '{word1_kq_bert}' and '{word2_kq_bert}' in context: {similarity_kq_bert:.4f}")
    
    # 示例2: "king" 和 "queen" 在不同（但相似）上下文中
    embedding_king_bert_ctx1 = get_bert_embeddings("The king ruled the kingdom.", "king")
    embedding_queen_bert_ctx2 = get_bert_embeddings("The queen ruled the kingdom.", "queen")
    similarity_kq_bert_diff_ctx = cosine_similarity_bert(embedding_king_bert_ctx1, embedding_queen_bert_ctx2)
    print(f"Cosine similarity between 'king' and 'queen' (diff context): {similarity_kq_bert_diff_ctx:.4f}")
    
    
    # 示例3: "king" 和 "apple" 在不同上下文中
    text1_ka = "The king ruled the kingdom."
    text2_ka = "I ate an apple for breakfast."
    embedding_king_bert_ka = get_bert_embeddings(text1_ka, "king")
    embedding_apple_bert_ka = get_bert_embeddings(text2_ka, "apple")
    similarity_ka_bert = cosine_similarity_bert(embedding_king_bert_ka, embedding_apple_bert_ka)
    print(f"Cosine similarity between 'king' and 'apple': {similarity_ka_bert:.4f}")

**输出解释：**

*   BERT 嵌入是**上下文相关**的。这意味着同一个词在不同的句子（上下文）中会有不同的嵌入向量。
*   在相似上下文中，"king" 和 "queen" 的余弦相似度较高 (例如 `0.7545` 或 `0.8322`，具体值取决于上下文和实现细节)。
*   "king" 和 "apple"（通常在不同上下文中）的余弦相似度较低 (例如 `0.3491`)。

这个教程很好地介绍了从零开始构建词嵌入的挑战，以及如何使用像 Word2Vec、GloVe 和 BERT 这样的预训练模型来获得高质量的词向量，并展示了它们在捕捉语义关系和上下文方面的能力。

信息来源和相关内容

[

github.com

](https://github.com/kirbyydoge/text-to-image-tr)

[

github.com

](https://github.com/kirbyydoge/text-to-image-tr)

[

github.com

](https://github.com/gswyhq/hello-world)

[

github.com

](https://github.com/gswyhq/hello-world)

[

tutorial\_w...odel.ipynb

![IPYNB 图标](https://drive-thirdparty.googleusercontent.com/32/type/text/code)

IPYNB







](https://contribution.usercontent.google.com/download?c=CgxiYXJkX3N0b3JhZ2USTxIMcmVxdWVzdF9kYXRhGj8KMDAwMDYzNGRhOTZkODUzZWMwN2Y1NWU2NmY0MjY4M2NmODQ0NGZlYjhlZjBjMjE4OBILEgcQ57PmiuAQGAE&filename&opi=103135050)

[

tutorial\_w...ng\_2.ipynb

![IPYNB 图标](https://drive-thirdparty.googleusercontent.com/32/type/text/code)

IPYNB







](https://contribution.usercontent.google.com/download?c=CgxiYXJkX3N0b3JhZ2USTxIMcmVxdWVzdF9kYXRhGj8KMDAwMDYzNGRhOWU1ZGQ2MmIwMWFmYWUyZDdkMzMzNGFmYTQwOWU5N2RiMDY3YmI3YRILEgcQ57PmiuAQGAE&filename&opi=103135050)

---

![IPYNB 图标](https://drive-thirdparty.googleusercontent.com/32/type/text/code)

tutorial\_w...ysis.ipynb

IPYNB

非常棒，但我觉得你的输出稍微有点简略，希望你下次可以输出地更加详细一些

---

显示思路

好的，我会尽力提供更详细的讲解。现在我们来看这个关于\*\*“微调BERT进行情感分析”\*\*的Jupyter Notebook教程。

这个教程的目标是学习如何为一个下游的自然语言处理任务（具体来说是情感分析）微调一个基于Transformer的模型（具体是BERT）。

### A - 引言 (Introduction)

教程首先指出，对于没有经验的人来说，微调语言模型一开始可能会有些挑战性。建议首先熟悉代码结构，然后再深入细节。希望通过这个教程，学习者能够获得实践经验，并对课程项目可能涉及的内容有所了解。

教程还提供了一些推荐的参考资料，以便更好地理解Transformer架构和BERT的实现：

*   **《The Illustrated BERT, ELMo, and co.》**: ([http://jalammar.github.io/illustrated-bert/](http://jalammar.github.io/illustrated-bert/)) 一个清晰易懂的BERT指南。
*   **`transformers` 库的文档**: ([https://huggingface.co/transformers/v2.2.0/index.html](https://huggingface.co/transformers/v2.2.0/index.html))
*   **《BERT Fine-Tuning Tutorial with PyTorch》**: ([http://mccormickml.com/2019/07/22/BERT-fine-tuning/](http://mccormickml.com/2019/07/22/BERT-fine-tuning/)) by Chris McCormick，一个非常详细的教程，展示了如何使用HuggingFace PyTorch库来操作BERT。

### B - 设置 (Setup)

#### 1\. 加载必要的库 (Load Essential Libraries)

这部分导入了几个Python中常用的库，用于数据处理、数值计算和绘图。

**关键代码：**

Python

    import os       # 用于操作系统相关功能，如文件路径
    import re       # 用于正则表达式操作，进行文本清洗
    from tqdm import tqdm  # 用于显示进度条，方便追踪循环进度
    import numpy as np    # 用于数值计算，特别是数组操作
    import pandas as pd   # 用于数据处理和分析，特别是DataFrame的使用
    import matplotlib.pyplot as plt # 用于数据可视化，绘制图表
    
    %matplotlib inline 
    # Jupyter Notebook的魔法命令，使得matplotlib绘制的图表直接在Notebook中显示

**详细说明：**

*   `os`: 提供了与操作系统交互的功能，例如文件和目录操作。在这个教程中可能用于管理数据文件。
*   `re`: 正则表达式模块，非常适合进行复杂的文本模式匹配和替换，将在数据预处理阶段用到。
*   `tqdm`: 一个快速、可扩展的进度条模块，当处理大量数据或长时间运行的循环时，它可以提供视觉反馈。
*   `numpy`: Python中科学计算的基础包，核心是N维数组对象，用于高效处理大型多维数组和矩阵运算。
*   `pandas`: 提供了高性能、易用的数据结构（如DataFrame）和数据分析工具。常用于读取、清洗、转换和分析表格数据。
*   `matplotlib.pyplot`: 一个绘图库，提供了类似MATLAB的绘图接口，用于创建静态、动态和交互式的可视化图表。
*   `%matplotlib inline`: 这是一个IPython/Jupyter Notebook的“魔法命令”，它告诉后端在执行代码后直接在Notebook的输出单元格中显示matplotlib生成的图表。

#### 2\. 数据集 (Dataset)

##### 2.1. 下载数据集 (Download Dataset)

这部分代码通过`requests`库从Google Drive下载一个名为`data.zip`的数据压缩文件，并使用`zipfile`库将其解压到名为`data`的文件夹中。

**关键代码：**

Python

    import requests # 用于发送HTTP请求，如此处用于下载文件
    request = requests.get("https://drive.google.com/uc?export=download&id=1wHt8PsMLsfX5yNSqrt2fSTcb8LEiclcf")
    with open("data.zip", "wb") as file: # 以二进制写入模式打开（或创建）data.zip文件
        file.write(request.content) # 将下载的内容写入文件
    
    import zipfile # 用于处理ZIP压缩文件
    with zipfile.ZipFile('data.zip') as zip: # 打开ZIP文件
        zip.extractall('data') # 将ZIP文件中的所有内容解压到'data'目录下

**详细说明：**

*   `requests.get(...)`: 向指定的URL发送一个GET请求。这里的URL指向Google Drive上的一个文件，`export=download`参数确保直接下载文件内容。
*   `with open("data.zip", "wb") as file:`: `open()`函数用于打开文件。"data.zip" 是要创建或覆盖的文件名，"wb" 表示以二进制写入（write binary）模式打开。`with`语句确保文件在使用完毕后会被正确关闭，即使发生错误也是如此。
*   `file.write(request.content)`: `request.content`包含了从URL下载的原始二进制数据，`file.write()`将其写入到本地的`data.zip`文件中。
*   `zipfile.ZipFile('data.zip')`: 创建一个`ZipFile`对象来操作`data.zip`文件。
*   `zip.extractall('data')`: 解压`ZipFile`对象中的所有文件和文件夹到当前工作目录下的`data`子目录中。如果`data`目录不存在，它会被创建。

##### 2.2. 加载训练数据 (Load Train Data)

训练数据包含两个文件，每个文件包含1700条抱怨性或非抱怨性的推文。数据中的每条推文至少包含一个航空公司的标签。加载数据后，会给它们打上标签（0代表抱怨，1代表非抱怨），并丢弃不重要的列，只保留`id`、`tweet`（推文内容）和`label`（标签）列。

**关键代码：**

Python

    # 加载数据并设置标签
    data_complaint = pd.read_csv('data/complaint1700.csv') # 从CSV文件读取抱怨性推文
    data_complaint['label'] = 0 # 为抱怨性推文添加标签0
    data_non_complaint = pd.read_csv('data/noncomplaint1700.csv') # 从CSV文件读取非抱怨性推文
    data_non_complaint['label'] = 1 # 为非抱怨性推文添加标签1
    
    # 合并抱怨性和非抱怨性数据
    data = pd.concat([data_complaint, data_non_complaint], axis=0).reset_index(drop=True)
    # pd.concat 用于沿指定轴（axis=0表示沿行堆叠）连接pandas对象。
    # .reset_index(drop=True) 重置合并后DataFrame的索引，并删除旧的索引。
    
    # 删除 'airline' 列
    data.drop(['airline'], inplace=True, axis=1)
    # inplace=True 表示直接在原始DataFrame上进行修改。
    # axis=1 表示删除列。
    
    # 显示5个随机样本
    data.sample(5)

**详细说明：**

*   `pd.read_csv()`: Pandas库中的函数，用于从CSV（逗号分隔值）文件中读取数据并创建一个DataFrame对象。
*   `data_complaint['label'] = 0`: 在`data_complaint` DataFrame中创建一个名为`label`的新列，并将所有行的该列值设置为0。
*   `pd.concat([...], axis=0)`: 将两个DataFrame (`data_complaint` 和 `data_non_complaint`) 垂直堆叠起来，形成一个包含所有推文的新DataFrame。
*   `.reset_index(drop=True)`: 合并后的DataFrame会保留原始DataFrame的索引，这可能导致索引重复。此方法会创建一个新的从0开始的默认整数索引，`drop=True`参数确保旧的索引列不会被添加到DataFrame中作为新的一列。
*   `data.drop(['airline'], inplace=True, axis=1)`: 从`data` DataFrame中删除名为`airline`的列。`inplace=True`意味着操作直接修改了`data` DataFrame，而不是返回一个新的修改后的DataFrame。`axis=1`指明操作是针对列的。
*   `data.sample(5)`: 从DataFrame中随机选择并显示5行数据，这有助于快速查看数据的结构和内容。

##### 2.3. 划分训练集和验证集

将整个训练数据随机划分为训练集（90%）和验证集（10%）。之后会在训练集上使用交叉验证进行超参数调整，并使用验证集来比较不同模型的性能。

**关键代码：**

Python

    from sklearn.model_selection import train_test_split # 从scikit-learn库导入数据划分函数
    
    X = data.tweet.values # 提取推文内容作为特征 (X)
    y = data.label.values # 提取标签作为目标变量 (y)
    
    # 划分数据集
    X_train, X_val, y_train, y_val = \
        train_test_split(X, y, test_size=0.1, random_state=2020)
    # test_size=0.1 指定验证集占总数据的10%。
    # random_state=2020 确保每次划分的结果一致，便于复现。

**详细说明：**

*   `sklearn.model_selection.train_test_split`: 这是scikit-learn库中一个非常常用的函数，用于将数据集划分为训练集和测试集（在这里是验证集）。
*   `X = data.tweet.values`: `data.tweet`选取了DataFrame中的`tweet`列（一个Pandas Series），`.values`将其转换为一个NumPy数组，这里作为模型的输入特征。
*   `y = data.label.values`: 类似地，选取`label`列并转换为NumPy数组，作为模型的目标输出。
*   `train_test_split(X, y, test_size=0.1, random_state=2020)`:
    *   `X, y`: 要划分的特征和标签。
    *   `test_size=0.1`: 指定测试集（或验证集）所占的比例。这里是10%，意味着训练集将占90%。
    *   `random_state=2020`: 设置一个随机种子。这保证了即使代码被多次运行，只要种子相同，数据的划分方式也是相同的，这对于结果的可复现性非常重要。
    *   函数返回四个数组：`X_train`（训练集特征），`X_val`（验证集特征），`y_train`（训练集标签），`y_val`（验证集标签）。

#### 3\. 设置GPU进行训练 (Set up GPU for training)

这部分代码检查当前Google Colab环境是否有可用的GPU。如果有，则将PyTorch的设备设置为GPU，并打印出可用的GPU数量和名称；否则，使用CPU。

**关键代码：**

Python

    import torch # 导入PyTorch库
    
    if torch.cuda.is_available(): # 检查CUDA（NVIDIA GPU的并行计算平台）是否可用
        device = torch.device("cuda") # 如果可用，则将设备设置为主CUDA设备（通常是第一个GPU）
        print(f'There are {torch.cuda.device_count()} GPU(s) available.') # 打印可用GPU的数量
        print('Device name:', torch.cuda.get_device_name(0)) # 打印第一个GPU的名称
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu") # 如果没有GPU，则将设备设置为CPU

**详细说明：**

*   `torch.cuda.is_available()`: PyTorch提供的函数，用于检测系统中是否存在可用的NVIDIA GPU以及CUDA驱动和工具包是否已正确安装。
*   `torch.device("cuda")`: 创建一个表示GPU设备的PyTorch设备对象。后续可以将模型和张量移动到这个设备上进行GPU加速计算。
*   `torch.cuda.device_count()`: 返回可用的GPU数量。
*   `torch.cuda.get_device_name(0)`: 返回索引为0的GPU的名称（如果是多GPU系统，索引从0开始）。
*   `torch.device("cpu")`: 创建一个表示CPU设备的PyTorch设备对象。

### C - 基线模型: TF-IDF + 朴素贝叶斯分类器 (Baseline: TF-IDF + Naive Bayes Classifier)

在尝试复杂的深度学习模型（如BERT）之前，通常会建立一个简单的基线模型作为性能参考。这里选择TF-IDF向量化文本数据，并使用朴素贝叶斯（Naive Bayes）作为分类器。

教程提到，朴素贝叶斯模型在文本数据上通常表现良好，并且作者在实验了包括随机森林、支持向量机、XGBoost在内的多种机器学习算法后，发现朴素贝叶斯性能最佳。

#### 1\. 数据准备 (Data Preparation)

##### 1.1. 预处理 (Preprocessing)

由于词袋模型（TF-IDF是其一种加权方式）不考虑语法和词序，因此需要对文本进行预处理，包括：

*   转换为小写。
*   修正缩写（例如 "`'t`" 改为 "`not`"）。
*   移除`@name`（通常是社交媒体中的提及）。
*   分离并移除除问号 `?` 以外的标点符号。
*   移除其他特殊字符。
*   移除停用词（stopwords，如 "the", "is", "in"），但保留 "not" 和 "can" 这类可能影响情感判断的词。
*   移除多余的空白字符。

**关键代码：`text_preprocessing` 函数**

Python

    import nltk
    nltk.download("stopwords") # 下载NLTK的停用词列表
    from nltk.corpus import stopwords # 导入停用词
    
    def text_preprocessing(s):
        s = s.lower() # 转换为小写
        s = re.sub(r"\'t", " not", s) # 't -> not
        s = re.sub(r'(@.*?)[\\s]', ' ', s) # 移除 @name
        s = re.sub(r'([\'\\\"\\.\\(\\)\\!\\?\\\\\\/\\,])', r' \\1 ', s) # 在标点符号周围添加空格，便于后续处理
        s = re.sub(r'[^\w\s\\?]', ' ', s) # 移除除字母、数字、空白、问号外的字符
        s = re.sub(r'([\\;\\:\\|•«\\n])', ' ', s) # 移除特定特殊字符
        s = " ".join([word for word in s.split() # 按空格分割成单词列表
                      if word not in stopwords.words('english') # 移除英文停用词
                      or word in ['not', 'can']]) # 保留 "not" 和 "can"
        s = re.sub(r'\\s+', ' ', s).strip() # 移除多余空格并去除首尾空格
        return s

**详细说明：**

*   `nltk.download("stopwords")`: 下载NLTK库中的停用词语料库（如果尚未下载）。
*   `stopwords.words('english')`: 获取一个包含常见英文停用词的列表。
*   `re.sub(pattern, repl, string)`: 正则表达式替换函数。
    *   `r"\'t"` 替换为 `" not"`：例如 "don't" 变为 "do not"。
    *   `r'(@.*?)[\\s]'` 替换为 `' '`: 移除@提及，例如 "@united hello" 变为 " hello"。
    *   `r'([\'\\\"\\.\\(\\)\\!\\?\\\\\\/\\,])'` 替换为 `r' \\1 '`: 在标点符号两侧添加空格，例如 "word." 变为 " word . "，这样做是为了后续能将标点符号作为单独的token处理或更容易地移除它们。
    *   `r'[^\w\s\\?]'` 替换为 `' '`: 移除非字母数字、非空白、非问号的字符。`\w` 匹配字母数字和下划线，`\s` 匹配空白字符。
    *   `r'([\\;\\:\\|•«\\n])'` 替换为 `' '`: 移除特定的特殊字符和换行符。
    *   `s.split()`: 将字符串按空白分割成单词列表。
    *   `[word for word in s.split() if word not in stopwords.words('english') or word in ['not', 'can']]`: 列表推导式，用于过滤停用词，但保留了 "not" 和 "can"。
    *   `" ".join(...)`: 将处理后的单词列表重新用单个空格连接成字符串。
    *   `re.sub(r'\\s+', ' ', s).strip()`: 将多个连续的空白字符替换为单个空格，并使用 `.strip()` 移除字符串首尾的空白。

##### 1.2. TF-IDF 向量化器 (TF-IDF Vectorizer)

**TF-IDF (Term Frequency–Inverse Document Frequency)** 是一种统计方法，用以评估一个词语对于一个文件集或一个语料库中的其中一份文件的重要程度。词语的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。

这里使用`sklearn.feature_extraction.text.TfidfVectorizer`将预处理后的文本数据转换为TF-IDF特征向量。

**关键代码：**

Python

    %%time 
    # Jupyter Notebook 魔法命令，用于测量单元格代码的执行时间
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # 对训练集和验证集的文本进行预处理
    X_train_preprocessed = np.array([text_preprocessing(text) for text in X_train])
    X_val_preprocessed = np.array([text_preprocessing(text) for text in X_val])
    
    # 初始化TF-IDF向量化器
    tf_idf = TfidfVectorizer(ngram_range=(1, 3), # 考虑单个词(unigram)、两个词(bigram)和三个词(trigram)的组合
                             binary=True,      # 如果为True，则TF部分只考虑词是否出现(1或0)，而不考虑其频率
                             smooth_idf=False) # IDF平滑参数，False表示不使用平滑
    
    # 在训练数据上拟合TF-IDF模型并转换训练数据
    X_train_tfidf = tf_idf.fit_transform(X_train_preprocessed)
    # 使用拟合好的TF-IDF模型转换验证数据
    X_val_tfidf = tf_idf.transform(X_val_preprocessed)

**详细说明：**

*   `TfidfVectorizer`:
    *   `ngram_range=(1, 3)`: 指定N-gram的范围。例如，(1, 1) 表示只使用单个词（unigrams），(1, 2) 表示使用单个词和两个连续词的组合（bigrams），(1, 3) 则还包括三个连续词的组合（trigrams）。使用N-grams可以捕捉一些局部词序信息。
    *   `binary=True`: 如果为True，则词频（TF）部分将只记录词是否出现（0或1），而不是实际的出现次数。这有时被称为二值TF。
    *   `smooth_idf=False`: IDF（逆文档频率）的计算公式中有一个平滑项，用于防止除以零或者给未在任何文档中出现的词一个有意义的IDF值。将其设置为False表示不使用这种平滑。
*   `tf_idf.fit_transform(X_train_preprocessed)`:
    *   `fit()`: 在训练数据上学习词汇表和IDF权重。
    *   `transform()`: 将训练数据转换为TF-IDF矩阵。
    *   `fit_transform()`: 是`fit()`和`transform()`的结合，更高效。
*   `tf_idf.transform(X_val_preprocessed)`: 使用在训练数据上学习到的词汇表和IDF权重来转换验证数据。**非常重要的一点是，验证集（和测试集）必须使用从训练集学习到的转换器进行转换，以避免数据泄露，并确保特征空间的一致性。**

#### 2\. 训练朴素贝叶斯分类器 (Train Naive Bayes Classifier)

##### 2.1. 超参数调整 (Hyperparameter Tuning)

使用**分层K折交叉验证 (Stratified K-Fold Cross-Validation)** 和 **AUC (Area Under the ROC Curve) 分数**来调整朴素贝叶斯分类器（具体是`MultinomialNB`）的超参数。`StratifiedKFold`确保在每一折中类别比例与整个数据集中的类别比例大致相同，这对于不平衡数据集尤其重要。

函数 `get_auc_CV` 用于计算给定模型在交叉验证中的平均AUC分数。

**关键代码：**

Python

    from sklearn.model_selection import StratifiedKFold, cross_val_score
    
    def get_auc_CV(model):
        """返回交叉验证的平均AUC分数。"""
        # 设置KFold在划分前打乱数据
        kf = StratifiedKFold(5, shuffle=True, random_state=1) # 5折交叉验证
    
        # 获取AUC分数
        auc = cross_val_score(
            model, X_train_tfidf, y_train, scoring="roc_auc", cv=kf)
        # model: 要评估的模型
        # X_train_tfidf, y_train: 训练数据的特征和标签
        # scoring="roc_auc": 指定评估指标为AUC
        # cv=kf: 指定交叉验证策略
    
        return auc.mean() # 返回AUC分数的平均值

`MultinomialNB` 类只有一个主要的超参数 `alpha`（拉普拉斯平滑参数）。下面的代码通过遍历一系列`alpha`值，找出能得到最高交叉验证AUC分数的`alpha`值，并绘制AUC随`alpha`变化的曲线。

**关键代码：**

Python

    from sklearn.naive_bayes import MultinomialNB
    
    # 测试不同的alpha值
    res = pd.Series([get_auc_CV(MultinomialNB(alpha=i))
                     for i in np.arange(1, 10, 0.1)], # alpha从1到9.9，步长0.1
                    index=np.arange(1, 10, 0.1))
    
    best_alpha = np.round(res.idxmax(), 2) # 找到AUC最高时对应的alpha值
    print('Best alpha: ', best_alpha)
    
    plt.plot(res) # 绘制AUC vs. Alpha的曲线
    plt.title('AUC vs. Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('AUC')
    plt.show()

**输出解释：**

*   会打印出最佳的`alpha`值（例如 `Best alpha: 1.3`）。
*   会显示一张图表，X轴是`alpha`值，Y轴是对应的交叉验证AUC均值。通过观察图表可以找到使AUC最大化的`alpha`值。

##### 2.2. 在验证集上评估 (Evaluation on Validation Set)

使用在训练集上找到的最佳`alpha`值来训练最终的朴素贝叶斯模型，并在之前划分出的验证集上评估其性能，计算准确率和AUC分数，并绘制ROC曲线。

函数 `evaluate_roc` 用于完成这个评估过程。

**关键代码：**

Python

    from sklearn.metrics import accuracy_score, roc_curve, auc
    
    def evaluate_roc(probs, y_true):
        """
        - 打印测试集上的AUC和准确率
        - 绘制ROC曲线
        @params probs (np.array): 预测概率数组，形状为 (len(y_true), 2)
        @params y_true (np.array): 真实标签数组，形状为 (len(y_true),)
        """
        preds = probs[:, 1] # 取出属于类别1的概率
        fpr, tpr, threshold = roc_curve(y_true, preds) # 计算ROC曲线的真正类率和假正类率
        roc_auc = auc(fpr, tpr) # 计算AUC值
        print(f'AUC: {roc_auc:.4f}')
    
        # 计算准确率
        y_pred = np.where(preds >= 0.5, 1, 0) # 以0.5为阈值进行分类
        accuracy = accuracy_score(y_true, y_pred)
        print(f'Accuracy: {accuracy*100:.2f}%')
    
        # 绘制ROC AUC曲线
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--') # 绘制随机猜测线
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
    
    # 使用最佳alpha训练模型并预测验证集概率
    nb_model = MultinomialNB(alpha=best_alpha) # 使用上面找到的最佳alpha
    nb_model.fit(X_train_tfidf, y_train) # 在整个TF-IDF处理后的训练集上训练模型
    probs = nb_model.predict_proba(X_val_tfidf) # 预测验证集上每个类别的概率
    
    # 评估分类器
    evaluate_roc(probs, y_val)

**输出解释：**

*   会打印出在验证集上的AUC分数（例如 `AUC: 0.8451`）和准确率（例如 `Accuracy: 75.59%`）。
*   会显示ROC曲线图。ROC曲线以假正类率（FPR）为横轴，真正类率（TPR）为纵轴。曲线越靠近左上角，表示模型性能越好。AUC是ROC曲线下的面积，值越接近1表示模型性能越好。

教程指出，通过TF-IDF和朴素贝叶斯的组合，在验证集上达到了约75.59%的准确率。这个值将作为后续微调BERT模型性能的基线。

### D - 微调BERT (Fine-tuning BERT)

#### 1\. 安装Transformer库 (Install the Transformer Library)

这部分是确保环境中安装了Hugging Face的`transformers`库。这个库包含了BERT等最先进NLP模型的PyTorch实现以及预训练模型的权重。

**关键代码：**

Python

    # !pip install transformers==2.8.0 # 教程中可能指定了特定版本
    !pip install transformers # 通常安装最新稳定版

#### 2\. 分词与输入格式化 (Tokenization and Input Formatting)

在对文本进行分词之前，会进行一些轻微的预处理，包括移除实体提及（如`@united`）和一些特殊字符。这里的预处理程度比基线模型要轻得多，因为BERT是在完整的句子上进行预训练的，能够自己学习到很多语言模式。

**关键代码：`text_preprocessing` (针对BERT的简化版)**

Python

    def text_preprocessing(text):
        """
        - 移除实体提及 (例如 '@united')
        - 修正错误 (例如 '&amp;' 改为 '&')
        """
        text = re.sub(r'(@.*?)[\\s]', ' ', text) # 移除@提及
        text = re.sub(r'&amp;', '&', text)    # &amp; -> &
        text = re.sub(r'\\s+', ' ', text).strip() # 移除多余空格
        return text
    
    # 打印一个句子的原始和处理后版本作为示例
    print('Original: ', X[0]) # X是原始推文数据
    print('Processed: ', text_preprocessing(X[0]))

**输出示例：**

Original:  @united I'm having issues. Yesterday I rebooked for 24 hours after I was supposed to fly, now I can't log on &amp; check in. Can you help?
    Processed:  I'm having issues. Yesterday I rebooked for 24 hours after I was supposed to fly, now I can't log on & check in. Can you help?

##### 2.1. BERT分词器 (BERT Tokenizer)

为了应用预训练的BERT模型，必须使用库中提供的对应分词器。这是因为：

1.  模型有一个特定的、固定的词汇表。
2.  BERT分词器有处理词汇表外（OOV）词语的特定方式（通常是WordPiece分词）。

此外，还需要：

*   在每个句子的开头和结尾添加特殊标记：`[CLS]`（分类标记，其最终隐藏状态用于分类任务）和 `[SEP]`（句子分隔标记）。
*   将所有句子填充（pad）或截断（truncate）到统一的固定长度。
*   使用“注意力掩码 (attention mask)”明确指定哪些标记是填充标记（模型不应关注它们）。

BERT分词器的`encode_plus`方法会完成以下工作：

1.  将文本分割成词元 (tokens)。
2.  添加特殊的 `[CLS]` 和 `[SEP]` 标记。
3.  将词元转换为分词器词汇表中的对应ID。
4.  将句子填充或截断到最大长度。
5.  创建注意力掩码。

**关键代码：**

Python

    from transformers import BertTokenizer
    
    # 加载BERT分词器 (bert-base-uncased 表示不区分大小写的BERT基础模型)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # do_lower_case=True 确保输入文本在分词前会转换为小写，与'bert-base-uncased'模型的预训练方式一致。
    
    # 创建一个函数来对一组文本进行分词和格式化
    def preprocessing_for_bert(data):
        input_ids = []
        attention_masks = []
    
        for sent in data:
            encoded_sent = tokenizer.encode_plus(
                text=text_preprocessing(sent),  # 预处理句子
                add_special_tokens=True,        # 添加 `[CLS]` 和 `[SEP]`
                max_length=MAX_LEN,             # 截断/填充到的最大长度
                pad_to_max_length=True,         # 填充到最大长度 (老版本参数，新版本建议用 padding='max_length')
                return_attention_mask=True     # 返回注意力掩码
                )
            
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))
    
        # 将列表转换为PyTorch张量
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)
    
        return input_ids, attention_masks

在分词之前，需要确定句子的最大长度`MAX_LEN`。代码中通过对所有推文进行编码，然后找出编码后序列的最大长度来确定这个值。

**关键代码：确定MAX\_LEN并进行分词**

Python

    # 对所有推文进行编码 (仅为了获取最大长度)
    encoded_tweets = [tokenizer.encode(sent, add_special_tokens=True) for sent in data.tweet.values]
    max_len = max([len(sent) for sent in encoded_tweets])
    print('Max length: ', max_len) # 例如输出 Max length: 68
    
    MAX_LEN = max_len # 将计算得到的最大长度设置为全局MAX_LEN
    
    # 打印一个句子的原始文本、其编码后的Token ID和Attention Mask作为示例
    # 注意：这里调用了两次preprocessing_for_bert，效率不高，仅为演示
    token_ids_example = list(preprocessing_for_bert([X[0]])[0].squeeze().numpy())
    token_masks_example = list(preprocessing_for_bert([X[0]])[1].squeeze().numpy())
    print('Original: ', X[0])
    print('Token IDs: ', token_ids_example)
    print('Token Masks: ', token_masks_example)
    
    # 对训练集和验证集运行 preprocessing_for_bert 函数
    print('Tokenizing data...')
    train_inputs, train_masks = preprocessing_for_bert(X_train)
    val_inputs, val_masks = preprocessing_for_bert(X_val)

**输出解释 (Token IDs 和 Token Masks):**

*   **Token IDs:** 文本被分词器转换后的一串数字ID。例如，`[101, 1045, 1005, ..., 102, 0, 0, ...]`。`101` 通常是 `[CLS]` 标记的ID，`102` 通常是 `[SEP]` 标记的ID。后面的 `0` 是填充标记的ID。
*   **Token Masks (Attention Masks):** 一个与Token IDs等长的二进制数组，其中 `1` 表示对应的token是真实词元或特殊标记，模型应该关注它；`0` 表示对应的token是填充标记，模型应该忽略它。

##### 2.2. 创建PyTorch DataLoader (Create PyTorch DataLoader)

为了在训练时高效地加载数据并节省内存，使用PyTorch的`TensorDataset`和`DataLoader`类。`TensorDataset`用于包装输入ID、注意力掩码和标签张量，而`DataLoader`则提供一个迭代器，可以按批次加载数据，并支持随机采样（用于训练集）和顺序采样（用于验证集）。

**关键代码：**

Python

    from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
    
    # 将标签列表转换为PyTorch张量
    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)
    
    # BERT的作者推荐批处理大小为16或32进行微调
    batch_size = 32
    
    # 为训练集创建DataLoader
    train_data = TensorDataset(train_inputs, train_masks, train_labels) # 封装张量
    train_sampler = RandomSampler(train_data) # 定义随机采样器
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size) # 创建DataLoader
    
    # 为验证集创建DataLoader
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data) # 定义顺序采样器
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

**详细说明：**

*   `torch.tensor()`: 将NumPy数组或Python列表转换为PyTorch张量。
*   `TensorDataset`: 一种数据集的包装类，它将多个张量（具有相同的第一维度，即样本数量）打包在一起。
*   `RandomSampler`: 在每个epoch中随机打乱数据索引进行采样，常用于训练过程以增加模型的泛化能力。
*   `SequentialSampler`: 始终按顺序对数据进行采样，常用于验证和测试过程。
*   `DataLoader`: 提供了对数据集的迭代访问。它封装了数据集和采样器，并允许按指定的`batch_size`批量加载数据。在训练深度学习模型时，数据通常是分批次送入模型的。

#### 3\. 训练模型 (Train Our Model)

##### 3.1. 创建BertClassifier (Create BertClassifier)

BERT-base模型包含12个Transformer层。每个Transformer层接收一系列词元嵌入作为输入，并输出相同数量、相同隐藏大小（维度）的嵌入。对于分类任务，通常使用特殊标记`[CLS]`在最后一个Transformer层的输出作为整个序列的特征表示，然后将其输入到一个分类器中。

虽然`transformers`库提供了`BertForSequenceClassification`类用于分类任务，但本教程选择创建一个自定义的`BertClassifier`类。这样做可以更灵活地指定分类器的结构。 这个自定义类包含：

*   一个BERT模型，用于提取`[CLS]`标记的最后一个隐藏状态。
*   一个单隐藏层的前馈神经网络作为分类器。

**关键代码：`BertClassifier` 类定义**

Python

    import torch.nn as nn
    from transformers import BertModel
    
    class BertClassifier(nn.Module):
        """用于分类任务的Bert模型"""
        def __init__(self, freeze_bert=False):
            """
            @param bert: BertModel对象
            @param classifier: torch.nn.Module分类器
            @param freeze_bert (bool): 设置为False以微调BERT模型
            """
            super(BertClassifier, self).__init__()
            # 指定BERT的隐藏层大小(D_in)，自定义分类器的隐藏层大小(H)，以及输出类别数(D_out)
            D_in, H, D_out = 768, 50, 2 # 768是bert-base的隐藏层大小，2是二分类（抱怨/非抱怨）
    
            # 实例化BERT模型
            self.bert = BertModel.from_pretrained('bert-base-uncased')
    
            # 实例化一个单层前馈分类器
            self.classifier = nn.Sequential(
                nn.Linear(D_in, H),       # 输入层 (BERT输出) -> 隐藏层
                nn.ReLU(),                # ReLU激活函数
                # nn.Dropout(0.5),        # Dropout层 (可选，用于正则化)
                nn.Linear(H, D_out)       # 隐藏层 -> 输出层 (类别logit)
            )
    
            # 如果freeze_bert为True，则冻结BERT模型的参数，使其在训练中不更新
            if freeze_bert:
                for param in self.bert.parameters():
                    param.requires_grad = False
            
        def forward(self, input_ids, attention_mask):
            """
            将输入喂给BERT和分类器以计算logits。
            @param input_ids (torch.Tensor): 输入张量，形状 (batch_size, max_length)
            @param attention_mask (torch.Tensor): 注意力掩码张量，形状 (batch_size, max_length)
            @return logits (torch.Tensor): 输出张量，形状 (batch_size, num_labels)
            """
            # 将输入喂给BERT
            outputs = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask)
            
            # 提取[CLS]标记的最后一个隐藏状态用于分类任务
            # outputs[0] 是所有token的最后一层隐藏状态，形状是 (batch_size, sequence_length, hidden_size)
            # outputs[0][:, 0, :] 选取了每个batch中第一个token ([CLS])的隐藏状态
            last_hidden_state_cls = outputs[0][:, 0, :]
    
            # 将[CLS]的隐藏状态喂给分类器以计算logits
            logits = self.classifier(last_hidden_state_cls)
    
            return logits

**详细说明：**

*   `nn.Module`: PyTorch中所有神经网络模块的基类。自定义模型时需要继承它。
*   `super(BertClassifier, self).__init__()`: 调用父类`nn.Module`的构造函数。
*   `D_in, H, D_out = 768, 50, 2`: 定义了网络层的维度。`D_in`是BERT基础模型的输出维度（768）。`H`是自定义分类器隐藏层的大小（50）。`D_out`是输出类别的数量（2，因为是二元情感分类）。
*   `self.bert = BertModel.from_pretrained('bert-base-uncased')`: 加载预训练的`bert-base-uncased`模型。这个模型只包含BERT的Transformer编码器部分，不包含顶部的特定任务层。
*   `self.classifier = nn.Sequential(...)`: 定义了一个序列模块，它按顺序包含了：
    *   `nn.Linear(D_in, H)`: 一个线性层，将768维的BERT输出映射到50维的隐藏层。
    *   `nn.ReLU()`: ReLU激活函数，引入非线性。
    *   `nn.Linear(H, D_out)`: 另一个线性层，将50维的隐藏层输出映射到2维的logits（每个类别一个logit值）。
*   `freeze_bert`: 一个布尔参数，如果为`True`，则BERT模型本身的参数在训练过程中不会被更新（即只训练顶部的分类器层）。如果为`False`（如此处默认），则BERT的参数也会在微调过程中被更新。
*   `forward(self, input_ids, attention_mask)`: 定义了模型的前向传播逻辑。
    *   `outputs = self.bert(...)`: 输入通过BERT模型。`outputs[0]`（也称为`last_hidden_state`）包含了序列中每个词元在最后一层的隐藏状态。
    *   `last_hidden_state_cls = outputs[0][:, 0, :]`: 提取`[CLS]`标记（通常是序列的第一个标记）的隐藏状态。在BERT中，这个标记的最终隐藏状态被认为是整个输入序列的聚合表示，常用于分类任务。
    *   `logits = self.classifier(last_hidden_state_cls)`: 将`[CLS]`标记的表示输入到自定义的分类器中，得到最终的分类logit值。

##### 3.2. 优化器和学习率调度器 (Optimizer and Learning Rate Scheduler)

为了微调`BertClassifier`，需要创建一个优化器。BERT的作者推荐以下超参数：

*   批处理大小 (Batch size): 16 或 32
*   学习率 (Adam): 5e-5, 3e-5 或 2e-5
*   训练轮数 (Number of epochs): 2, 3, 4

Hugging Face的`run_glue.py`脚本（一个使用`transformers`库的示例）中使用了AdamW优化器。AdamW是Adam优化器的一个变体，它改进了权重衰减（weight decay）的处理方式。 同时，通常还会使用学习率调度器 (learning rate scheduler)，它可以在训练过程中动态调整学习率。`get_linear_schedule_with_warmup`是一种常见的策略，它在训练初期有一个小的“预热（warmup）”阶段，学习率从0线性增加到一个初始值，之后学习率再线性衰减到0。

**关键代码：`initialize_model` 函数**

Python

    from transformers import AdamW, get_linear_schedule_with_warmup
    
    def initialize_model(epochs=4):
        """初始化BertClassifier、优化器和学习率调度器。"""
        bert_classifier = BertClassifier(freeze_bert=False) # 实例化分类器，不冻结BERT参数以便微调
        bert_classifier.to(device) # 将模型移动到GPU（如果可用）
    
        # 创建优化器
        optimizer = AdamW(bert_classifier.parameters(),
                          lr=5e-5,    # 学习率
                          eps=1e-8    # AdamW的epsilon参数，防止除以零
                          )
    
        # 总训练步数 = 每个epoch的批次数 * epoch数
        total_steps = len(train_dataloader) * epochs
    
        # 设置学习率调度器
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0, # 预热步数，0表示无预热或使用默认（通常是总步数的百分比）
                                                    num_training_steps=total_steps) # 总训练步数
        return bert_classifier, optimizer, scheduler

**详细说明：**

*   `AdamW(bert_classifier.parameters(), lr=5e-5, eps=1e-8)`:
    *   `bert_classifier.parameters()`: 将模型中所有需要训练的参数传递给优化器。
    *   `lr=5e-5`: 设置初始学习率。
    *   `eps=1e-8`: AdamW优化器中的一个小常数，用于数值稳定性。
*   `total_steps = len(train_dataloader) * epochs`: 计算训练过程中的总步数。`len(train_dataloader)` 得到的是一个epoch中批次的数量。
*   `get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)`:
    *   `optimizer`: 要调度的优化器。
    *   `num_warmup_steps`: 在训练开始时，学习率从0线性增加到`optimizer`中设置的初始学习率所需要的步数。设置为0表示不进行显式的预热，或者采用`transformers`库的默认预热策略。
    *   `num_training_steps`: 训练的总步数。学习率将在预热后从初始值线性衰减到0，直到达到此总步数。

##### 3.3. 训练循环 (Training Loop)

模型将训练2个epoch。在每个epoch中：

*   **训练阶段：**
    *   从dataloader中解包数据并将数据加载到GPU。
    *   清除上一轮计算的梯度 (`model.zero_grad()`)。
    *   执行前向传播，计算logits和损失 (`loss_fn(logits, b_labels)`)。
    *   执行反向传播，计算梯度 (`loss.backward()`)。
    *   （教程中注释掉了）梯度裁剪 (`torch.nn.utils.clip_grad_norm_`)：将梯度的范数裁剪到1.0，以防止“梯度爆炸”问题，这在训练大型神经网络时很有用。
    *   更新模型的参数 (`optimizer.step()`)。
    *   更新学习率 (`scheduler.step()`)。
*   **评估阶段 (在每个epoch结束后)：**
    *   将模型设置为评估模式 (`model.eval()`)，这会禁用dropout等在训练和评估时行为不同的层。
    *   从dataloader解包数据并加载到GPU。
    *   执行前向传播。
    *   计算在验证集上的损失和准确率。

损失函数使用交叉熵损失 `nn.CrossEntropyLoss()`，这对于分类任务是标准的。 函数 `set_seed(42)` 用于设置随机种子，以确保实验结果的可复现性。

**关键代码：`train` 和 `evaluate` 函数**

Python

    import random
    import time
    
    loss_fn = nn.CrossEntropyLoss() # 定义损失函数
    
    def set_seed(seed_value=42):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # 为所有GPU设置种子
    
    def train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
        print("Start training...\\n")
        for epoch_i in range(epochs):
            print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
            print("-" * 70)
            t0_epoch, t0_batch = time.time(), time.time()
            total_loss, batch_loss, batch_counts = 0, 0, 0
            model.train() # 将模型设置为训练模式
    
            for step, batch in enumerate(train_dataloader):
                batch_counts += 1
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
                model.zero_grad() # 清除梯度
                logits = model(b_input_ids, b_attn_mask) # 前向传播
                loss = loss_fn(logits, b_labels) # 计算损失
                batch_loss += loss.item()
                total_loss += loss.item()
                loss.backward() # 反向传播
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪（可选）
                optimizer.step() # 更新参数
                scheduler.step() # 更新学习率
    
                if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                    time_elapsed = time.time() - t0_batch
                    print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()
            
            avg_train_loss = total_loss / len(train_dataloader)
            print("-" * 70)
    
            if evaluation == True:
                val_loss, val_accuracy = evaluate(model, val_dataloader) # 在验证集上评估
                time_elapsed = time.time() - t0_epoch
                print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
                print("-" * 70)
            print("\\n")
        print("Training complete!")
    
    def evaluate(model, val_dataloader):
        model.eval() # 将模型设置为评估模式
        val_accuracy = []
        val_loss = []
        for batch in val_dataloader:
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            with torch.no_grad(): # 在评估时不计算梯度
                logits = model(b_input_ids, b_attn_mask)
            loss = loss_fn(logits, b_labels)
            val_loss.append(loss.item())
            preds = torch.argmax(logits, dim=1).flatten() # 获取预测类别
            accuracy = (preds == b_labels).cpu().numpy().mean() * 100 # 计算准确率
            val_accuracy.append(accuracy)
        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)
        return val_loss, val_accuracy
    
    # 开始训练
    set_seed(42)
    bert_classifier, optimizer, scheduler = initialize_model(epochs=2) # 初始化模型，训练2个epoch
    train(bert_classifier, train_dataloader, val_dataloader, epochs=2, evaluation=True)

**输出解释 (训练过程中的打印信息):**

*   每个epoch开始时会打印表头。
*   训练过程中，每20个batch（或最后一个batch）会打印当前epoch、当前batch序号、自上一次打印以来的平均训练损失、以及所用时间。验证损失和验证准确率在此时显示为"-"，因为它们只在整个epoch结束后计算。
*   每个epoch结束后，会打印该epoch的平均训练损失、在验证集上的损失和准确率，以及该epoch的总耗时。
*   `Training complete!` 表示训练结束。

**输出示例 (一个epoch结束后的评估行):**

   1    |    -    |   0.542978   |  0.442527  |   78.47   |   33.42  

这表示第一个epoch结束后，平均训练损失为0.542978，验证损失为0.442527，验证准确率为78.47%，该epoch耗时33.42秒。

##### 3.4. 在验证集上评估 (Evaluation on Validation Set)

训练完成后，需要对模型在验证集上的最终性能进行一次评估。预测步骤与训练循环中的评估步骤类似，但不计算损失，而是计算softmax概率。

**关键代码：`bert_predict` 函数和最终评估**

Python

    import torch.nn.functional as F
    
    def bert_predict(model, test_dataloader):
        """在测试集上执行前向传播以预测概率。"""
        model.eval() # 设置为评估模式
        all_logits = []
        for batch in test_dataloader:
            b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2] # 注意这里只取前两个元素，因为验证时不需要标签
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)
            all_logits.append(logits)
        all_logits = torch.cat(all_logits, dim=0) # 合并所有批次的logits
        probs = F.softmax(all_logits, dim=1).cpu().numpy() # 应用softmax计算概率，并移至CPU转为numpy数组
        return probs
    
    # 在验证集上进行预测
    probs_val = bert_predict(bert_classifier, val_dataloader)
    
    # 使用之前定义的 evaluate_roc 函数评估性能
    evaluate_roc(probs_val, y_val)

**输出解释：**

*   会打印出最终在验证集上的AUC分数和准确率（例如 `AUC: 0.8891`, `Accuracy: 80.00%`）。
*   会显示最终的ROC曲线图。

教程指出，微调后的BERT分类器在验证集上达到了约0.90的AUC分数和80%的准确率，这比基线方法（TF-IDF + 朴素贝叶斯）的75.59%准确率高出近5个百分点。

### E - 结论 (Conclusion)

*   通过在BERT之上添加一个简单的单隐层神经网络分类器并进行微调，即使数据量不大，也能达到接近最先进水平的性能，显著优于基线方法。
*   尽管BERT模型非常庞大复杂，参数数以百万计，但通常只需要微调2-4个epoch就能取得良好效果。这是因为BERT已经在海量数据上进行了预训练，编码了大量关于语言的信息。
*   在短时间内用少量数据就能达到令人印象深刻的性能，这展示了为什么BERT是目前最强大的NLP模型之一。

这个教程详细地介绍了如何准备数据、建立基线模型，并最终使用Hugging Face Transformers库微调BERT模型进行情感分析任务，涵盖了从数据预处理、模型构建、训练到评估的完整流程。

信息来源和相关内容

---

![IPYNB 图标](https://drive-thirdparty.googleusercontent.com/32/type/text/code)

tutorial\_w...tion.ipynb

IPYNB

非常棒，同样地讲解下面内容

---

显示思路

好的，我会更详细地为您讲解这个关于\*\*“微调BERT进行命名实体识别 (NER)”\*\*的Jupyter Notebook教程。

### CS6493 - 教程 5：微调BERT进行命名实体识别

#### 引言 (Introduction)

本教程旨在探索如何利用BERT模型进行命名实体识别（NER）。NER是自然语言理解（NLU）中的一项基本任务，它涉及到在文本中识别并将命名实体分类到预定义的类别中。这些命名实体是现实世界中的对象，如人名（persons）、组织机构名（organizations）、地名（locations）、日期（dates）、数值表达式（numerical expressions）等。

我们将使用HuggingFace的`transformers`库中包含的 **`BertForTokenClassification`** 模型。该模型以BERT作为其基础架构，并在顶部添加了一个词元分类头（token classification head），使其能够在词元级别（token level）而不是序列级别（sequence level）进行预测。命名实体识别通常被视为一个词元分类问题，因此我们将使用此模型来完成这项任务。

本教程采用了**迁移学习 (transfer learning)** 的思想，即首先以无监督的方式预训练一个大型神经网络（如BERT，它在掩码语言建模和下一句预测任务上进行了预训练），然后在感兴趣的具体任务上（此处为NER）对该神经网络进行微调（fine-tuning）。微调是一个有监督的学习过程，因此需要一个带标签的数据集。

教程同样推荐了一些深入了解BERT的资源：

*   原始论文: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
*   Jay Allamar的博客文章: [The Illustrated BERT, ELMo, and co.](http://jalammar.github.io/illustrated-bert/) 和他的教程 [A Visual Guide to Using BERT for the First Time](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)
*   Chris Mccormick的YouTube频道和Abbishek Kumar Mishra的YouTube频道。

#### 导入Python库并准备环境 (Importing Python Libraries and preparing the environment)

首先，确保安装了必要的库，如`pandas`, `numpy`, `sklearn`, `pytorch`, `transformers`, 和 `seqeval`。 在Google Colab环境中，特别提到了检查`transformers`和`seqeval`库。

**关键代码：安装和导入库**

Python

    # 安装transformers库 (通常教程会指定版本，但这里是通用安装)
    !pip install transformers
    # !pip install seqeval # 如果环境中没有，也需要安装seqeval
    
    import pandas as pd # 用于数据处理，特别是DataFrame
    import numpy as np # 用于数值运算
    from sklearn.metrics import accuracy_score # 用于评估模型准确率
    import torch # PyTorch深度学习框架
    from tqdm import tqdm # 用于显示进度条
    from torch.utils.data import Dataset, DataLoader # PyTorch数据加载工具
    from transformers import BertTokenizer, BertConfig, BertForTokenClassification # HuggingFace Transformers库中的BERT相关类

**详细说明：**

*   `pandas`, `numpy`, `torch`, `tqdm` 的作用与上一个教程类似。
*   `sklearn.metrics.accuracy_score`: scikit-learn库中用于计算分类准确率的函数。
*   `torch.utils.data.Dataset, DataLoader`: PyTorch中用于创建和加载自定义数据集的工具，便于批量处理和数据迭代。
*   `transformers.BertTokenizer`: BERT模型对应的分词器，用于将文本转换为模型可以理解的格式（token IDs, attention masks等）。
*   `transformers.BertConfig`: BERT模型的配置类，包含了模型的各种参数，如隐藏层大小、层数等。
*   `transformers.BertForTokenClassification`: HuggingFace提供的专门用于词元级别分类任务的BERT模型，非常适合NER这类任务。

接下来，代码设置了运行设备（GPU或CPU）：

Python

    from torch import cuda
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(device)

如果输出是`cuda`，则表示GPU可用并被选中。

#### 下载和预处理数据 (Downloading and preprocessing the data)

命名实体识别（NER）使用一种特定的标注方案，通常在**词级别**定义。广泛使用的是**IOB标注法 (Inside–Outside–Beginning)**。每个标签指示对应的词是位于命名实体的内部（I-）、外部（O）还是开始（B-）。使用这种方法是因为命名实体通常包含多个词。

**IOB标注示例：** 句子：“Barack Obama was born in Hawaï” 对应标签：`[B-PERS, I-PERS, O, O, O, B-GEO]`

*   `B-PERS`: "Barack" 是人名（PERS）的开始。
*   `I-PERS`: "Obama" 在人名内部。
*   `O`: "was" 在任何命名实体之外。
*   `B-GEO`: "Hawaï" 是地名（GEO）的开始。

因此，通常一个句子有多少个词，就有多少个对应的标签。

本教程使用来自Kaggle的[NER数据集](https://www.kaggle.com/namanj27/ner-dataset)，该数据集已经是IOB格式。

**关键代码：加载数据**

Python

    # 假设ner_datasetreference.csv文件已上传到Colab的/content/目录下
    file = '/content/ner_datasetreference.csv'
    data = pd.read_csv(file , encoding='unicode_escape') # 使用'unicode_escape'编码读取，以处理特殊字符
    data.head() # 显示数据的前几行

**数据格式解释：** 输出的DataFrame的头部显示了数据的列结构，通常包含：

*   `Sentence #`: 句子编号。对于同一个句子中的所有词，这个编号是相同的，但只有第一个词会显示，后续词显示为NaN（Not a Number）。
*   `Word`: 句子中的具体词语。
*   `POS`: 词性标注 (Part-of-Speech tag)。
*   `Tag`: 命名实体识别标签 (例如 'O', 'B-geo', 'I-per' 等)。

**数据统计：**

Python

    data.count()

这段代码会输出每一列非缺失值的数量。从输出可以看出，数据集中大约有48,000个句子，超过100万个词和标签，平均每句约20个词。

**标签频率分析：**

Python

    print("Number of tags: {}".format(len(data.Tag.unique()))) # 打印数据集中唯一标签的数量
    frequencies = data.Tag.value_counts() # 计算每个标签出现的频率
    print(frequencies)

输出会显示数据集中共有17种不同的IOB标签。`O`（Outside）标签的频率最高，因为它表示非实体词。其他标签如 `B-geo` (地名开始), `B-tim` (时间开始), `B-org` (组织名开始) 等也比较常见。

**合并B-和I-标签进行频率统计：** 为了更好地理解各类实体的总体频率，代码将B- (Beginning) 和 I- (Inside) 标签合并统计（例如，B-geo 和 I-geo 都计入 geo 实体的总数）。

Python

    tags = {}
    for tag, count in zip(frequencies.index, frequencies):
        if tag != "O": # 忽略'O'标签
            entity_type = tag[2:5] # 提取实体类型，如 "geo", "per"
            if entity_type not in tags.keys():
                tags[entity_type] = count
            else:
                tags[entity_type] += count
        continue
    
    print(sorted(tags.items(), key=lambda x: x[1], reverse=True)) # 按频率降序打印实体类型及其总数

**输出示例：** `[('geo', 45058), ('org', 36927), ('per', 34241), ...]`

**移除低频实体类型：** 教程指出，由于 "art" (艺术品), "eve" (事件), "nat" (自然现象) 这些实体的标签数量较少，模型在这些类别上的性能可能不佳，因此选择将它们从数据集中移除。

Python

    entities_to_remove = ["B-art", "I-art", "B-eve", "I-eve", "B-nat", "I-nat"]
    data = data[~data.Tag.isin(entities_to_remove)] # 使用.isin()和波浪号(~)进行反向选择，保留不在移除列表中的标签的行
    data.head()

**按句子组织数据：** NER任务的训练样本通常是一个完整的句子及其对应的标签序列。原始数据是每个词一行，需要将其转换为每个句子一行，并将句子中的词和标签分别合并。 `fillna(method='ffill')` 用于向前填充 "Sentence #" 列中的NaN值，使得每个词都有其对应的句子编号。 然后使用 `groupby(['Sentence #'])` 和 `transform(lambda x: ' '.join(x))`（对于词语）以及 `transform(lambda x: ','.join(x))`（对于标签）来将属于同一句子的词和标签分别合并成字符串。

**关键代码：**

Python

    # 使用前向填充（forward fill）填充 'Sentence #' 列的 NaN 值
    data = data.fillna(method='ffill')
    
    # 创建 'sentence' 列，将每个句子的词语合并成一个字符串
    data['sentence'] = data[['Sentence #','Word','Tag']].groupby(['Sentence #'])['Word'].transform(lambda x: ' '.join(x))
    # 创建 'word_labels' 列，将每个句子的标签合并成一个逗号分隔的字符串
    data['word_labels'] = data[['Sentence #','Word','Tag']].groupby(['Sentence #'])['Tag'].transform(lambda x: ','.join(x))
    data.head()

**数据格式转换后的示例：** DataFrame现在会有一列是完整的句子字符串，另一列是对应的标签序列字符串。

**创建标签到ID的映射：** 为了让模型能够处理标签，需要将文本标签（如 'B-geo', 'O'）映射到数字ID。

Python

    labels_to_ids = {k: v for v, k in enumerate(data.Tag.unique())} # 创建标签到ID的映射字典
    ids_to_labels = {v: k for v, k in enumerate(data.Tag.unique())} # 创建ID到标签的映射字典
    print(labels_to_ids)

输出会显示如 `{'O': 0, 'B-geo': 1, ...}` 这样的映射关系。在移除了低频实体后，唯一标签的数量会减少（例如，从17个减少到11个左右）。

**去重并最终确定数据集格式：** 在按句子聚合后，会产生重复的行（因为原始数据中每个词占一行）。通过选取 "sentence" 和 "word\_labels" 列并使用 `drop_duplicates().reset_index(drop=True)` 来得到每个句子唯一的表示。

Python

    data = data[["sentence", "word_labels"]].drop_duplicates().reset_index(drop=True)
    print(data.head())
    print(len(data)) # 打印处理后数据集中的句子总数

**验证数据：** 随机抽取一条数据，检查句子和对应的标签是否正确。

Python

    print(data.iloc[8].sentence)      # 打印第9个句子的文本
    print(data.iloc[8].word_labels)   # 打印第9个句子对应的标签序列

#### 准备数据集和数据加载器 (Preparing the dataset and dataloader)

**定义关键变量：**

Python

    MAX_LEN = 128          # 输入序列的最大长度
    TRAIN_BATCH_SIZE = 8   # 训练时的批处理大小
    VALID_BATCH_SIZE = 4   # 验证时的批处理大小
    EPOCHS = 1             # 训练轮数
    LEARNING_RATE = 2e-05  # 学习率
    MAX_GRAD_NORM = 10     # 梯度裁剪的最大范数，防止梯度爆炸
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # 加载BERT分词器

**详细说明：**

*   `MAX_LEN`: BERT模型能处理的序列最大长度。超过此长度的序列会被截断，不足的会被填充。
*   `TRAIN_BATCH_SIZE` / `VALID_BATCH_SIZE`: 在训练和验证过程中，一次性输入给模型的样本数量。
*   `EPOCHS`: 整个训练数据集将被遍历的次数。
*   `LEARNING_RATE`: 优化器在更新模型权重时使用的学习率。
*   `MAX_GRAD_NORM`: 梯度裁剪的阈值。如果梯度的L2范数超过这个值，梯度将被缩放到这个值，有助于稳定训练。

**处理WordPiece分词与标签对齐的问题：** BERT使用WordPiece分词，这意味着一个原始单词可能会被分解成多个子词（wordpieces）。例如，“Washington” 可能被分解为 “Wash”, “##ing”, “##ton”。原始的IOB标签是在词级别的，需要将这些标签传播到对应的所有子词上。

**`tokenize_and_preserve_labels` 函数：** 这个函数负责处理单个句子及其词级标签，将其转换为BERT分词器产生的子词序列和对应的子词级标签序列。对于一个被分解成多个子词的单词，其原始标签会被复制到所有这些子词上。

**关键代码：**

Python

    def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
        tokenized_sentence = []
        labels = []
        sentence = sentence.strip() # 去除首尾空格
    
        for word, label in zip(sentence.split(), text_labels.split(",")): # 遍历原始词和标签对
            tokenized_word = tokenizer.tokenize(word) # 对单个词进行WordPiece分词
            n_subwords = len(tokenized_word)
    
            tokenized_sentence.extend(tokenized_word) # 将子词添加到结果列表
            labels.extend([label] * n_subwords) # 将原始词的标签复制n_subwords次
    
        return tokenized_sentence, labels

**设计决策说明：** 教程提到，对于如何处理子词的标签，有多种设计选择：

1.  **传播标签（本教程采用）：** 将原始词的标签赋予其所有的子词。
2.  **仅标记首个子词：** 只给每个词的第一个子词赋予原始标签，其余子词标记为特殊标签或忽略（原始BERT论文的做法）。
3.  **IOB风格的子词标签：** 给第一个子词B-X标签，后续子词I-X标签（如果原始标签是B-X的话）。

**自定义PyTorch Dataset类：** 创建一个名为`dataset`的类，它继承自`torch.utils.data.Dataset`。这个类负责将DataFrame中的每个样本（句子和标签字符串）转换为模型训练所需的PyTorch张量。 `__getitem__`方法是核心，它执行以下步骤：

1.  **分词并对齐标签：** 使用`tokenize_and_preserve_labels`函数处理句子和标签，得到子词序列和对应的子词级标签序列。
2.  **添加特殊标记：** 在子词序列的开头添加`[CLS]`，结尾添加`[SEP]`。对应地，在标签序列的相应位置添加'O'标签（因为特殊标记不是命名实体）。
3.  **截断/填充：** 将子词序列和标签序列处理到`MAX_LEN`。如果序列过长，则截断；如果过短，则用`[PAD]`标记（对于词元）和'O'标签（对于标签）进行填充。
4.  **创建注意力掩码：** 对于真实的词元和特殊标记，掩码值为1；对于填充标记`[PAD]`，掩码值为0。
5.  **转换为ID：** 使用分词器的`convert_tokens_to_ids`方法将子词序列转换为ID序列。使用之前创建的`labels_to_ids`字典将标签序列转换为ID序列。
    *   **注意：** 教程中提到一行 `label_ids = [label if label != 0 else -100 for label in label_ids]` 被弃用。在Hugging Face的`BertForTokenClassification`中，计算损失时会忽略标签为-100的词元。这通常用于处理填充标记或WordPiece分词中不应计算损失的子词。当前的实现（直接使用'O'标签的ID进行填充，并且在后续的评估中可能需要特殊处理）可能与Hugging Face `Trainer`的默认行为略有不同，但对于自定义训练循环是可行的。

**关键代码：`dataset` 类**

Python

    class dataset(Dataset):
        def __init__(self, dataframe, tokenizer, max_len):
            self.len = len(dataframe)
            self.data = dataframe
            self.tokenizer = tokenizer
            self.max_len = max_len
    
        def __getitem__(self, index):
            sentence = self.data.sentence[index]
            word_labels = self.data.word_labels[index]
            tokenized_sentence, labels = tokenize_and_preserve_labels(sentence, word_labels, self.tokenizer)
    
            tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"]
            labels.insert(0, "O")
            labels.insert(-1, "O") # 注意这里是 insert(-1, "O")，应为 labels.append("O") 或 labels.insert(len(labels), "O")，更准确的是在SEP后也应该是'O'
    
            maxlen = self.max_len
            if (len(tokenized_sentence) > maxlen):
                tokenized_sentence = tokenized_sentence[:maxlen]
                labels = labels[:maxlen]
            else:
                tokenized_sentence = tokenized_sentence + ['[PAD]'] * (maxlen - len(tokenized_sentence))
                labels = labels + ["O"] * (maxlen - len(labels)) # 用'O'的ID填充标签
    
            attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]
            ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
            label_ids = [labels_to_ids[label] for label in labels]
            # 对于那些因为截断或填充而产生的标签，或者WordPiece中非首个子词的标签，
            # HuggingFace Trainer通常期望标签为-100，以便在损失计算中忽略它们。
            # 这里的实现直接用'O'对应的ID填充，在计算损失和评估时需要注意这一点。
    
            return {
                  'ids': torch.tensor(ids, dtype=torch.long),
                  'mask': torch.tensor(attn_mask, dtype=torch.long),
                  'targets': torch.tensor(label_ids, dtype=torch.long)
            }
    
        def __len__(self):
            return self.len

**修正 `labels.insert(-1, "O")`**: `labels.insert(-1, "O")` 会将 "O" 插入到倒数第二个位置。对于 `[SEP]` 标记，正确的做法应该是 `labels.append("O")` 或者如果 `labels` 已经是最终长度（除了 `[SEP]` 对应标签），则 `labels[len(tokenized_sentence)-1] = labels_to_ids["O"]`（如果 `[SEP]` 已被加入）。更简单的是，确保 `labels` 列表在填充之前与 `tokenized_sentence` (包含 `[CLS]` 和 `[SEP]` 之后) 的长度一致，并且特殊标记的标签是 'O' 或其ID（或者-100，如果想在损失中忽略它们）。

**创建训练集和测试集的数据集对象和数据加载器：** 使用80/20的比例划分数据，并为训练集和测试集分别创建`dataset`类的实例和`DataLoader`。

Python

    train_size = 0.8
    train_dataset_df = data.sample(frac=train_size,random_state=200) # Pandas DataFrame
    test_dataset_df = data.drop(train_dataset_df.index).reset_index(drop=True) # Pandas DataFrame
    train_dataset_df = train_dataset_df.reset_index(drop=True)
    
    print("FULL Dataset: {}".format(data.shape))
    print("TRAIN Dataset: {}".format(train_dataset_df.shape))
    print("TEST Dataset: {}".format(test_dataset_df.shape))
    
    training_set = dataset(train_dataset_df, tokenizer, MAX_LEN)
    testing_set = dataset(test_dataset_df, tokenizer, MAX_LEN)
    
    # 查看第一个训练样本的处理结果
    print(training_set[0])
    
    # 定义DataLoader参数
    train_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
    test_params = {'batch_size': VALID_BATCH_SIZE, 'shuffle': False, 'num_workers': 0}
    
    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

**输出解释 (training\_set\[0\])：** 会显示第一个训练样本经过`dataset`类处理后的结果，是一个包含`ids`, `mask`, 和 `targets`张量的字典。

*   `ids`: 词元ID序列，已填充/截断到`MAX_LEN`。
*   `mask`: 注意力掩码序列。
*   `targets`: 对应的标签ID序列，也已填充/截断。

#### 定义模型 (Defining the model)

使用HuggingFace的`BertForTokenClassification`。加载预训练的`bert-base-uncased`权重，并指定`num_labels`（标签类别的数量，根据之前创建的`labels_to_ids`的长度确定）。 模型加载时会提示，顶部的词元分类头是新初始化的，需要进行训练。 然后将模型移动到GPU。

**关键代码：**

Python

    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(labels_to_ids))
    model.to(device) # 将模型移动到GPU或CPU
    print(model) # 可以打印模型结构

**输出解释 (print(model))：** 会打印出`BertForTokenClassification`模型的详细结构，包括BERT基础模型（嵌入层、多层Transformer编码器）和顶部的分类层（通常是一个线性层，输出维度等于`num_labels`）。

#### 训练模型 (Training the model)

**定义优化器：** 使用Adam优化器，学习率在之前已定义为`LEARNING_RATE` (例如2e-5)。

Python

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

**训练函数 `train(epoch)`：** 这个函数定义了一个epoch的训练过程：

1.  将模型设置为训练模式 (`model.train()`)。
2.  迭代`training_loader`中的每个批次数据。
3.  将数据移到指定设备（GPU/CPU）。
4.  获取模型输出（损失`loss`和logits）。`BertForTokenClassification`模型内部会计算损失（如果提供了`labels`参数）。
5.  累积损失。
6.  每500步打印一次训练损失。
7.  计算训练准确率：
    *   将目标标签和预测的logits展平。
    *   只在非填充（注意力掩码为1）的位置计算准确率。
    *   使用`accuracy_score`计算批次准确率并累加。
8.  梯度裁剪：使用`torch.nn.utils.clip_grad_norm_`防止梯度爆炸。
9.  反向传播：`optimizer.zero_grad()`清除旧梯度，`loss.backward()`计算新梯度，`optimizer.step()`更新模型参数。
10.  打印整个epoch的平均训练损失和准确率。

**关键代码：`train` 函数**

Python

    def train(epoch):
        tr_loss, tr_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        tr_preds, tr_labels = [], []
        model.train()
    
        for idx, batch in enumerate(tqdm(training_loader, desc=f"Training Epoch {epoch+1}")): # 添加描述
            ids = batch['ids'].to(device, dtype=torch.long)
            mask = batch['mask'].to(device, dtype=torch.long)
            targets = batch['targets'].to(device, dtype=torch.long)
    
            # 模型直接返回包含loss和logits的对象
            outputs = model(input_ids=ids, attention_mask=mask, labels=targets, return_dict=True)
            loss = outputs.loss
            tr_logits = outputs.logits
            
            tr_loss += loss.item()
            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)
    
            if idx % 500 == 0 and idx > 0: # 调整打印频率和避免0时打印
                loss_step = tr_loss/nb_tr_steps
                print(f"Training loss at {idx} training steps in epoch {epoch+1}: {loss_step}")
    
            # 计算准确率时，只考虑非-100的标签（即非填充、非特殊标记的子词）
            flattened_targets = targets.view(-1)
            active_logits = tr_logits.view(-1, model.num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)
            
            # 只选择标签不是-100的部分进行准确率计算（这是HuggingFace Trainer的典型做法）
            # 当前代码是基于注意力掩码，这可能包括了[CLS]和[SEP]的'O'标签
            active_accuracy_mask = mask.view(-1) == 1 # 或者更准确的是 targets.view(-1) != -100
            
            actual_targets = torch.masked_select(flattened_targets, active_accuracy_mask)
            actual_predictions = torch.masked_select(flattened_predictions, active_accuracy_mask)
            
            # 如果需要严格按照非填充、非特殊标记的词元计算准确率，
            # 并且labels_to_ids['O'] (或其他特殊标签) 被用作填充或特殊标记的标签ID，
            # 那么active_accuracy_mask的定义需要更精确地排除这些。
            # 但通常，如果targets中对[PAD]等标记的label_id设为-100，模型在计算损失时会自动忽略它们。
            # 对于准确率，也应该只在非-100的标签上计算。
            # 教程中的实现是基于原始mask，这意味着它也计算了[CLS], [SEP]等标记的准确率（如果它们被标记为'O'）。
    
            tr_preds.extend(actual_predictions.cpu().numpy()) # 保存实际的预测（非填充部分）
            tr_labels.extend(actual_targets.cpu().numpy())    # 保存实际的标签（非填充部分）
    
            # accuracy_score应该在非-100的标签上计算
            # 如果 labels_to_ids['O'] 也被用于填充，那么这里的准确率计算可能不完全是针对命名实体的。
            # 一个更标准的做法是在评估函数中处理 seqeval 所需的格式。
            # 这里的 tmp_tr_accuracy 是基于当前批次中所有被 attention_mask 覆盖的 token。
            tmp_tr_accuracy = accuracy_score(actual_targets.cpu().numpy(), actual_predictions.cpu().numpy())
            tr_accuracy += tmp_tr_accuracy
    
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=MAX_GRAD_NORM)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        epoch_loss = tr_loss / nb_tr_steps
        tr_accuracy = tr_accuracy / nb_tr_steps # 这是批次准确率的平均值
        print(f"Training loss epoch {epoch+1}: {epoch_loss}")
        print(f"Training accuracy epoch {epoch+1}: {tr_accuracy}")
    
    # 执行训练
    for epoch in range(EPOCHS): # EPOCHS之前定义为1
        # print(f"Training epoch: {epoch + 1}") # 这行在函数内部有了
        train(epoch) # epoch从0开始

**输出解释：** 训练过程会显示每个epoch的训练损失和（基于批次的平均）训练准确率。由于`EPOCHS`设置为1，所以只会训练一个epoch。

#### 评估模型 (Evaluating the model)

在训练完成后，在测试集上评估模型的性能。评估函数 `valid(model, testing_loader)` 与训练函数中的评估部分类似：

1.  将模型设置为评估模式 (`model.eval()`)。
2.  不计算梯度 (`torch.no_grad()`)。
3.  迭代测试数据加载器，进行前向传播，计算损失和logits。
4.  收集所有真实标签和预测标签（同样只考虑非填充部分）。
5.  将预测的ID和真实的ID转换回文本标签（例如 'B-PER', 'O'）。
6.  计算并打印验证集上的平均损失和（基于批次的平均）准确率。
7.  返回转换回文本的真实标签列表和预测标签列表，这对于后续使用`seqeval`等工具计算更细致的NER指标（如Precision, Recall, F1-score per entity type）非常重要。

**关键代码：`valid` 函数**

Python

    def valid(model, testing_loader):
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_examples, nb_eval_steps = 0, 0
        eval_preds, eval_labels = [], []
    
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(testing_loader, desc="Evaluating")):
                ids = batch['ids'].to(device, dtype=torch.long)
                mask = batch['mask'].to(device, dtype=torch.long)
                targets = batch['targets'].to(device, dtype=torch.long)
    
                outputs = model(input_ids=ids, attention_mask=mask, labels=targets, return_dict=True)
                loss = outputs.loss
                eval_logits = outputs.logits
                eval_loss += loss.item()
                nb_eval_steps += 1
                # ... (与train函数中准确率计算类似的部分) ...
                flattened_targets = targets.view(-1) 
                active_logits = eval_logits.view(-1, model.num_labels) 
                flattened_predictions = torch.argmax(active_logits, axis=1) 
                
                active_accuracy_mask = mask.view(-1) == 1 
                
                # 收集用于seqeval的标签和预测（只保留有效部分）
                for i in range(targets.size(0)): # 遍历batch中的每个句子
                    # 实际标签（非-100，且mask为1）
                    # 注意：如果targets中的填充标签已经是-100，则mask.view(-1) == 1可能不够，
                    # 需要 labels_true_batch = [label_list[i][mask_list[i].bool()] for i in range(len(label_list))]
                    # 但由于targets是long类型，所以需要先转为bool再mask
                    
                    # 当前代码逻辑是基于attention_mask提取的
                    # 对于seqeval，通常需要的是每个句子的标签序列列表
                    
                    # 以下代码是为了得到与 `seqeval` 兼容的格式：
                    # 即列表的列表，内层列表是每个句子的标签序列
                    
                    # 获取当前批次中每个句子的真实标签和预测标签
                    # 移除填充和特殊标记对应的标签（通常标记为 -100 或根据 attention_mask）
                    # 这里简化处理，直接使用ids_to_labels转换，但实际应只转换非填充部分
                    
                    # 教程中 labels 和 predictions 的收集方式更适合计算整体准确率，
                    # 而不是直接用于 seqeval，除非 seqeval 能处理扁平化的列表并有方法对齐。
                    # seqeval 期望的是一个句子列表，每个句子是标签列表。
    
                # 将数值标签和预测转换为字符串标签用于seqeval
                # 注意：原始代码中 eval_labels 和 eval_preds 是扁平化的，
                # seqeval 通常需要的是句子级别的标签列表 [[sent1_labels], [sent2_labels], ...]
                # 这里我们先计算基于token的准确率
                actual_targets = torch.masked_select(flattened_targets, active_accuracy_mask)
                actual_predictions = torch.masked_select(flattened_predictions, active_accuracy_mask)
    
                eval_labels.extend(actual_targets.cpu().numpy()) # 收集所有有效的真实标签ID
                eval_preds.extend(actual_predictions.cpu().numpy())   # 收集所有有效的预测标签ID
                
                tmp_eval_accuracy = accuracy_score(actual_targets.cpu().numpy(), actual_predictions.cpu().numpy())
                eval_accuracy += tmp_eval_accuracy
                
        # 将收集到的所有有效标签ID和预测ID转换为文本标签
        labels_str = [ids_to_labels[id_item] for id_item in eval_labels]
        predictions_str = [ids_to_labels[id_item] for id_item in eval_preds]
    
        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_steps # 平均批次准确率
        print(f"Validation Loss: {eval_loss}")
        print(f"Validation Accuracy: {eval_accuracy}") # 这个准确率是基于所有非填充词元的
    
        return labels_str, predictions_str # 返回扁平化的标签和预测字符串列表
    
    # 执行评估
    labels, predictions = valid(model, testing_loader)

**输出解释 (Validation Loss 和 Validation Accuracy)：**

*   `Validation Loss`: 模型在测试集上的平均损失。
*   `Validation Accuracy`: 模型在测试集上所有非填充词元的平均准确率（例如 `0.9622`，即96.22%）。

教程指出，这个准确率指标具有误导性，因为大量的标签是 "O"（Outside）。更重要的是查看各个实体类型的精确率（Precision）、召回率（Recall）和F1分数。 为此，教程使用了一个自定义的方法来计算这些指标，而不是直接使用`seqeval`库（尽管提到了它）。这里的计算方式是基于所有非'O'标签的词元。

**关键代码：自定义P, R, F1计算**

Python

    # labels 和 predictions 是从 valid 函数返回的扁平化列表
    equ_num = sum([int(predictions[i] == labels[i] and predictions[i] != 'O' and labels[i] != 'O') 
                   for i in range(len(labels))])
    # equ_num: 预测正确且真实标签和预测标签都不是'O'的数量（真正例中的正确部分）
    
    predicted_positive = sum([int(predictions[i] != 'O') for i in range(len(predictions))])
    # predicted_positive: 模型预测为非'O'标签的总数
    
    true_posistive = sum([int(labels[i] != 'O') for i in range(len(labels))])
    # true_posistive: 数据集中真实非'O'标签的总数
    
    p = equ_num / predicted_positive if predicted_positive > 0 else 0.0
    r = equ_num / true_posistive if true_posistive > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0 # 防止 p+r 为0
    
    print({"P": p, "R": r, "F1": f1})

**输出解释 (P, R, F1):** 例如 `{'P': 0.8307, 'R': 0.8177, 'F1': 0.8242}`。这些指标更能反映模型在识别具体命名实体上的性能。

#### 推理 (Inference)

这部分展示了如何在新的、未见过的句子上使用训练好的模型进行预测。 推理过程的关键在于正确处理WordPiece分词。教程中采用的策略是使用每个原始单词的第一个子词的预测标签作为该单词的标签。

**关键代码：推理示例**

Python

    sentence = "India has a capital called Mumbai. On wednesday, the president will give a presentation"
    
    inputs = tokenizer(sentence, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="pt")
    
    ids = inputs["input_ids"].to(device)
    mask = inputs["attention_mask"].to(device)
    
    model.eval() # 确保模型在评估模式
    with torch.no_grad():
        outputs = model(ids, mask)
    logits = outputs[0] # BertForTokenClassification 直接返回 logits，或者通过 outputs.logits
    
    active_logits = logits.view(-1, model.num_labels) 
    flattened_predictions = torch.argmax(active_logits, axis=1)
    
    tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    token_predictions_ids = flattened_predictions.cpu().numpy()
    token_predictions_labels = [ids_to_labels[i] for i in token_predictions_ids]
    
    wp_preds = list(zip(tokens, token_predictions_labels)) # (wordpiece, prediction)
    
    word_level_predictions = []
    current_word_tokens = []
    original_words = sentence.split() # 原始单词列表
    word_idx = 0
    final_predictions_for_sentence = []
    
    # 一个更鲁棒的对齐方法
    aligned_labels = []
    word_ids = inputs.word_ids() # 获取每个token对应的原始单词索引
    
    previous_word_idx = None
    for i, word_idx_for_token in enumerate(word_ids):
        if word_idx_for_token is None: # 特殊token [CLS], [SEP], [PAD]
            continue
        elif word_idx_for_token != previous_word_idx: # 新词的第一个token
            aligned_labels.append(token_predictions_labels[i])
        # else: # 同一个词的后续token，可以忽略，或者采用其他策略（如投票）
        previous_word_idx = word_idx_for_token
    
    # 教程中的简化版对齐逻辑：
    simple_word_level_predictions = []
    for token_str, pred_label in wp_preds:
      if (token_str.startswith("##")) or (token_str in ['[CLS]', '[SEP]', '[PAD]']):
        # 跳过非首个子词和特殊标记的预测
        continue
      else:
        simple_word_level_predictions.append(pred_label)
    
    # 重新构建句子（不含特殊标记和子词合并）
    str_rep = " ".join([t[0] for t in wp_preds if t[0] not in ['[CLS]', '[SEP]', '[PAD]']]).replace(" ##", "")
    print(str_rep) # 打印的是基于BERT分词器重构的句子，可能与原始输入略有差异
    print(simple_word_level_predictions) # 打印的是每个（可能是多段的）原始词的第一个子词的预测标签

**输出示例：**

india has a capital called mumbai . on wednesday , the president will give a presentation
    ['B-geo', 'O', 'O', 'O', 'O', 'B-geo', 'O', 'O', 'B-tim', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

这显示了句子中被识别出的命名实体及其类型。例如，"india" 被识别为 `B-geo` (地理位置开始)，"mumbai" 也是 `B-geo`，"wednesday" 被识别为 `B-tim` (时间开始)。

#### 保存模型 (Saving the model for future use)

最后，教程展示了如何保存训练好的模型和分词器，以便将来重新加载和使用。

*   `tokenizer.save_vocabulary(directory)`: 保存分词器的词汇表。
*   `model.save_pretrained(directory)`: 保存模型的权重和配置文件。

**关键代码：**

Python

    import os
    directory = "/content/model" # 临时保存目录
    if not os.path.exists(directory):
        os.makedirs(directory)
    tokenizer.save_vocabulary(directory)
    model.save_pretrained(directory)
    print('All files saved')
    
    # 永久保存到Google Drive的示例
    DRIVE_PATH = "/content/drive/MyDrive/"
    MODEL_SAVE_PATH = DRIVE_PATH + "ner_model" # 定义在Google Drive中的保存路径
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH) # 注意：tokenizer.save_vocabulary 保存的是vocab.txt, 应该用 save_pretrained 保存tokenizer的完整状态
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Tokenizer saved to: {MODEL_SAVE_PATH}")

**修正 `tokenizer.save_vocabulary`**: 对于Hugging Face的`BertTokenizer` (或更通用的`AutoTokenizer`)，更推荐使用 `tokenizer.save_pretrained(directory)` 来保存分词器的所有相关文件（包括词汇表文件、配置文件和特殊标记映射文件），这样可以确保之后能用 `AutoTokenizer.from_pretrained(directory)` 完整地加载回来。

#### 使用Hugging Face `Trainer` API 进行NER任务 (New Trend on NLP: Prompt what you want with Open-Sourced LLMs)

教程的最后一部分引入了使用Hugging Face `Trainer` API来简化NER任务的训练和评估流程。这是一个更高级和推荐的做法，因为它封装了很多底层的训练循环和评估逻辑。

**核心步骤：**

1.  **安装必要的库：** `transformers`, `datasets`, `seqeval`, `accelerate`, `evaluate`。Python
    
        # %%capture # Jupyter魔法命令，用于捕获单元格输出，使其不显示
        # !pip install transformers datasets seqeval accelerate evaluate
    
2.  **加载数据集：** 使用`datasets.load_dataset`加载例如`conll2003`这样的标准NER数据集。Python
    
        from datasets import load_dataset
        dataset = load_dataset("conll2003", trust_remote_code=True) # trust_remote_code=True 用于允许执行数据集仓库中的自定义代码
    
3.  **定义标签映射：** 从数据集中获取标签列表，并创建`label2id`和`id2label`的映射。Python
    
        labels = dataset["train"].features["ner_tags"].feature.names
        label2id = {label: i for i, label in enumerate(labels)}
        id2label = {i: label for i, label in enumerate(labels)}
    
4.  **加载分词器和模型：** 使用`AutoTokenizer`和`AutoModelForTokenClassification`。在加载模型时，传入标签映射。Python
    
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        model_checkpoint = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id
        )
    
5.  **定义分词和标签对齐函数 `tokenize_and_align_labels`：** 这个函数与之前手动实现版本中的目标类似，但通常会利用`tokenized_inputs.word_ids()`来更精确地将原始词的标签对齐到WordPiece子词上。对于特殊标记（如`[CLS]`, `[SEP]`）和非首个子词，标签通常设置为`-100`，以便在损失计算中被忽略。Python
    
        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(
                examples["tokens"],
                truncation=True,
                is_split_into_words=True, # 表明输入已经是按词分割的
                padding="max_length",     # 或 "longest"
                max_length=128            # 与之前定义的MAX_LEN一致
            )
            labels_batch = []
            for i, tags_for_sentence in enumerate(examples["ner_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids_for_sentence = []
                for word_idx in word_ids:
                    if word_idx is None: # 特殊标记
                        label_ids_for_sentence.append(-100)
                    elif word_idx != previous_word_idx: # 新词的第一个token
                        label_ids_for_sentence.append(tags_for_sentence[word_idx])
                    else: # 同一个词的后续token，也标记为-100（或遵循B-I-O的第一个子词策略）
                        label_ids_for_sentence.append(-100) 
                    previous_word_idx = word_idx
                labels_batch.append(label_ids_for_sentence)
            tokenized_inputs["labels"] = labels_batch
            return tokenized_inputs
    
6.  **处理数据集：** 使用`dataset.map()`应用上述函数。Python
    
        tokenized_datasets = dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset["train"].column_names # 移除原始列，保留处理后的列
        )
    
7.  **定义评估指标函数 `compute_metrics`：** 这个函数接收模型的预测结果和真实标签，使用`evaluate.load("seqeval")`加载`seqeval`评估工具，计算精确率、召回率、F1分数和准确率。它需要将ID形式的预测和标签转换回字符串形式，并正确处理被忽略的标签（`-100`）。Python
    
        import evaluate # Hugging Face Evaluate库
        import numpy as np
        metric = evaluate.load("seqeval")
        
        def compute_metrics(p): # p 是一个包含predictions和labels的EvalPrediction对象
            predictions, labels_true_ids = p
            predictions = np.argmax(predictions, axis=2) # 从logits中获取最可能的预测ID
        
            # 将ID转换回标签字符串，并移除-100的标签
            true_predictions_str = [
                [id2label[p_id] for (p_id, l_id) in zip(prediction_row, label_row) if l_id != -100]
                for prediction_row, label_row in zip(predictions, labels_true_ids)
            ]
            true_labels_str = [
                [id2label[l_id] for (p_id, l_id) in zip(prediction_row, label_row) if l_id != -100]
                for prediction_row, label_row in zip(predictions, labels_true_ids)
            ]
        
            results = metric.compute(predictions=true_predictions_str, references=true_labels_str)
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }
    
8.  **定义训练参数 `TrainingArguments`：**Python
    
        from transformers import TrainingArguments
        training_args = TrainingArguments(
            output_dir="./ner_results",       # 输出结果的目录
            evaluation_strategy="epoch",    # 每个epoch结束时进行评估
            learning_rate=2e-5,
            per_device_train_batch_size=16, # 调整以适应GPU内存
            per_device_eval_batch_size=16,
            num_train_epochs=3,             # 训练轮数
            weight_decay=0.01,
            logging_dir="./logs",             # 日志目录
            report_to="none"                  # 不向wandb等报告
        )
    
9.  **初始化 `Trainer`：**Python
    
        from transformers import Trainer, DataCollatorForTokenClassification
        
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer) # 用于处理token分类任务的批次数据
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer, # 传递tokenizer给Trainer
            data_collator=data_collator, # 使用专门的数据整理器
            compute_metrics=compute_metrics
        )
    
10.  **开始训练和评估：**Python
    
        trainer.train()
        final_metrics = trainer.evaluate() # 在验证集（或测试集，如果配置了的话）上评估
        print("Final metrics:", final_metrics)
    
11.  **保存模型：**Python
    
        model.save_pretrained("./ner_model_trainer")
        tokenizer.save_pretrained("./ner_tokenizer_trainer")
    

**最后一部分代码（在`test_on_text`函数之后）用于在CoNLL-2003测试集上评估模型，并打印详细的实体级别指标。这部分代码在教程的最后一个单元格中，它加载了之前训练好的模型（或者可以直接使用`trainer`对象中的模型），对测试集进行预处理，然后进行预测和评估。**

**评估CoNLL-2003测试集的关键步骤：**

*   加载模型和分词器（如果是新会话，则从已保存的路径加载）。
*   加载`conll2003`的`test`分割。
*   使用`tokenize_and_align_labels`函数处理测试数据。
*   创建`DataLoader`。
*   定义`evaluate_model`函数，该函数迭代测试数据加载器，获取模型预测，将ID转回标签，并使用`seqeval`计算指标。
*   调用`evaluate_model`并打印详细的整体指标和各实体类型的P, R, F1分数。

**交互式测试 `test_on_text` 函数：** 这个函数允许输入一个句子，然后模型会输出每个词及其预测的NER标签（只显示非'O'的标签）。这对于直观地测试模型在任意文本上的表现很有用。

**关于使用LLM进行提示 (Prompting LLMs) 的讨论：** 教程最后简要提及了NLP的新趋势：使用大型语言模型（LLMs）通过提示（prompting）来完成任务。并给出了一个向类似DeepSeek-R1这样的模型提问如何用Hugging Face模型训练NER任务的例子。强调了理解代码背后的逻辑仍然重要，尤其是在需要自定义损失函数等情况下。核心思想是：“你可以得到任何你想要的，但前提是你必须确切地知道你想要什么”，并对自己的代码负责。

这个教程提供了一个非常全面和详细的流程，涵盖了从数据准备、模型选择、预处理、训练、评估到推理和保存模型的完整步骤，特别是针对NER任务微调BERT模型。它还对比了手动实现训练循环和使用Hugging Face `Trainer` API的两种方法，后者更为简洁高效。

信息来源和相关内容

---

![IPYNB 图标](https://drive-thirdparty.googleusercontent.com/32/type/text/code)

tutorial\_w...tion.ipynb

IPYNB

同样地，讲解该文件

---

显示思路

好的，我们来详细讲解这个关于\*\*“通过联合学习对齐和翻译实现神经机器翻译”\*\*的Jupyter Notebook教程。

### CS6493 - 教程 6: 通过联合学习对齐和翻译实现神经机器翻译

#### 引言 (Introduction)

本教程将使用 PyTorch 和 TorchText 来实现论文 [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) 中提出的模型。这个模型的核心是将\*\*注意力机制 (Attention)\*\*引入到序列到序列 (Seq2Seq) 模型中。

**注意力机制的核心思想：**

1.  **计算注意力向量 (a)**: 对于每个解码时间步，都会计算一个与源句子等长的注意力向量 a。
2.  **注意力向量的特性**:
    *   向量中的每个元素值介于 0 和 1 之间。
    *   整个向量的所有元素之和为 1。
3.  **计算加权源向量 (w)**: 使用注意力向量 a 对源句子的隐藏状态序列 H (即编码器在每个时间步的输出) 进行加权求和，得到一个加权的源向量 w。 公式为： w\=∑i​ai​hi​
4.  **解码器中的应用**: 在解码的每个时间步，都会计算一个新的加权源向量 w。这个 w 将作为解码器RNN的输入（与常规的嵌入向量和上一个时间步的隐藏状态一起），并且也可能用于最终通过线性层进行预测。

教程的目标是详细解释如何实现上述所有步骤。

#### 准备数据 (Preparing Data)

教程提到数据准备过程与上一个教程类似，暗示了会使用相似的数据集和预处理流程。

**关键代码：安装和导入库** 首先，教程安装了必要的库，特别是`evaluate`和`datasets`，这通常是Hugging Face生态系统的一部分，用于评估模型性能和加载数据集。尽管这个教程后面主要使用的是TorchText和Spacy，但这些库的安装可能是为了某些辅助功能或者是一个通用的环境设置。

Python

    !pip install evaluate
    !pip install datasets
    # 安装特定版本的spacy德语和英语模型
    !pip3 install https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.0.0/de_core_news_sm-3.0.0.tar.gz
    !pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import random
    import numpy as np
    import spacy
    import datasets # Hugging Face datasets库
    import torchtext # 用于文本处理的PyTorch库
    import tqdm # 用于显示进度条
    import evaluate # Hugging Face evaluate库，用于评估指标
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

**详细说明：**

*   `!pip install evaluate datasets`: 安装Hugging Face的`evaluate`和`datasets`库。
*   `!pip3 install ...spacy-models...`: 安装特定版本的spaCy德语 (`de_core_news_sm`) 和英语 (`en_core_web_sm`) 小模型。这些模型用于文本的分词。
*   `torch`, `torch.nn`, `torch.optim`: PyTorch核心库，用于构建神经网络、定义层、损失函数和优化器。
*   `random`, `numpy`: 用于生成随机数和数值运算。
*   `spacy`: 一个流行的NLP库，这里主要用于分词。
*   `torchtext`: PyTorch的文本处理库，包含用于构建词汇表、数据迭代器等工具。
*   `tqdm`: 用于在循环中显示进度条。
*   `evaluate`: Hugging Face的评估库，可以方便地计算各种NLP任务的指标（如BLEU）。
*   `matplotlib.pyplot`, `matplotlib.ticker`: 用于绘图。

**设置随机种子：** 为了保证实验的可复现性，代码设置了Python的`random`、`numpy`和`torch`的随机种子。

Python

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) # 如果使用GPU，也为CUDA设置种子
    torch.backends.cudnn.deterministic = True # 确保CUDA的确定性行为

**详细说明：**

*   通过固定随机种子，可以确保每次运行代码时，所有依赖随机性的操作（如参数初始化、数据打乱、dropout等）都会产生相同的结果，这对于调试和比较模型非常重要。
*   `torch.backends.cudnn.deterministic = True`: 当使用CUDA进行GPU加速时，此设置会使cuDNN库使用确定性的卷积算法，有助于提高可复现性，但可能会牺牲一些性能。

**加载数据集：** 教程使用`datasets.load_dataset("bentrevett/multi30k")`从Hugging Face Hub加载Multi30k数据集。Multi30k是一个常用于机器翻译任务的数据集，包含德语（de）和英语（en）的平行句子对。

**关键代码：**

Python

    dataset = datasets.load_dataset("bentrevett/multi30k")
    
    train_data, valid_data, test_data = (
        dataset["train"],
        dataset["validation"],
        dataset["test"],
    )

**详细说明：**

*   `load_dataset("bentrevett/multi30k")`: 从Hugging Face Hub下载并加载`multi30k`数据集的`bentrevett`版本。这个数据集已经预先划分好了训练集、验证集和测试集。
*   代码将数据集的三个部分分别赋值给`train_data`, `valid_data`, 和 `test_data`。

**创建分词器 (Tokenizers)：** 使用`spacy`为源语言（德语）和目标语言（英语）创建分词器。

Python

    en_nlp = spacy.load("en_core_web_sm") # 加载英语小型模型
    de_nlp = spacy.load("de_core_news_sm") # 加载德语小型模型

**定义分词函数 `tokenize_example`：** 这个函数接收一个样本（包含"en"和"de"键的字典），使用对应的spaCy模型进行分词，将文本转换为小写（如果`lower`参数为True），在序列的开头和结尾分别添加起始符（`sos_token`，如`<sos>`）和结束符（`eos_token`，如`<eos>`），并限制最大长度。

**关键代码：**

Python

    def tokenize_example(example, en_nlp, de_nlp, max_length, lower, sos_token, eos_token):
        # 对英语句子分词，并取前max_length个词元
        en_tokens = [token.text for token in en_nlp.tokenizer(example["en"])][:max_length]
        # 对德语句子分词，并取前max_length个词元
        de_tokens = [token.text for token in de_nlp.tokenizer(example["de"])][:max_length]
        if lower: # 如果设置为True，则转换为小写
            en_tokens = [token.lower() for token in en_tokens]
            de_tokens = [token.lower() for token in de_tokens]
        # 在序列前后添加特殊标记
        en_tokens = [sos_token] + en_tokens + [eos_token]
        de_tokens = [sos_token] + de_tokens + [eos_token]
        return {"en_tokens": en_tokens, "de_tokens": de_tokens}

**应用分词函数到数据集：** 定义一些参数（如`max_length`, `lower`, `sos_token`, `eos_token`），然后使用`dataset.map()`方法将`tokenize_example`函数应用到训练集、验证集和测试集的所有样本上。

Python

    max_length = 1_000 # 设置序列最大长度
    lower = True       # 是否转小写
    sos_token = "<sos>"  # 句子起始标记
    eos_token = "<eos>"  # 句子结束标记
    
    fn_kwargs = { # 将参数打包成字典传递给map方法
        "en_nlp": en_nlp,
        "de_nlp": de_nlp,
        "max_length": max_length,
        "lower": lower,
        "sos_token": sos_token,
        "eos_token": eos_token,
    }
    
    train_data = train_data.map(tokenize_example, fn_kwargs=fn_kwargs)
    valid_data = valid_data.map(tokenize_example, fn_kwargs=fn_kwargs)
    test_data = test_data.map(tokenize_example, fn_kwargs=fn_kwargs)

**详细说明：**

*   `dataset.map(function, fn_kwargs=...)`: Hugging Face `datasets`库的`map`方法可以将一个函数并行地应用到数据集的每个样本上。`fn_kwargs`允许传递额外的参数给这个函数。
*   处理后，每个数据集的样本中会新增`en_tokens`和`de_tokens`两个键，分别存储分词并添加了特殊标记后的英语和德语词元列表。

**构建词汇表 (Build the vocabulary)：** 使用`torchtext.vocab.build_vocab_from_iterator`为源语言（德语）和目标语言（英语）分别构建词汇表。词汇表只包含在训练数据中出现频率至少为`min_freq`（这里是2）的词元，并添加了特殊标记（未知词`unk_token`，填充符`pad_token`，起始符`sos_token`，结束符`eos_token`）。

**关键代码：**

Python

    min_freq = 2
    unk_token = "<unk>"
    pad_token = "<pad>"
    
    special_tokens = [
        unk_token,
        pad_token,
        sos_token,
        eos_token,
    ]
    
    # 从训练数据的英文词元构建英文词汇表
    en_vocab = torchtext.vocab.build_vocab_from_iterator(
        train_data["en_tokens"], # 提供词元迭代器
        min_freq=min_freq,       # 词元最小频率
        specials=special_tokens, # 特殊标记列表
    )
    
    # 从训练数据的德语词元构建德语词汇表
    de_vocab = torchtext.vocab.build_vocab_from_iterator(
        train_data["de_tokens"],
        min_freq=min_freq,
        specials=special_tokens,
    )

**详细说明：**

*   `build_vocab_from_iterator`: 这个函数会遍历提供的词元迭代器（这里是训练数据的所有英语或德语词元列表），统计词频，并构建一个词汇表对象。
*   词汇表对象可以将词元映射到整数索引，反之亦然。
*   `min_freq`: 只有出现次数达到这个阈值的词元才会被加入词汇表。低于此频率的词元在后续处理中通常会被替换为`unk_token`。
*   `specials`: 特殊标记会被添加到词汇表的开头，并被赋予固定的索引。

**设置默认索引和词元数值化：** 确保`unk_token`和`pad_token`在两个词汇表中有相同的索引，并将词汇表的默认索引设置为`unk_index`（当查询词汇表中不存在的词元时，返回`unk_token`的索引）。 然后定义`numericalize_example`函数，将分词后的词元序列转换为对应的整数索引序列。

**关键代码：**

Python

    # 确保特殊标记的索引一致性（通常build_vocab_from_iterator会按顺序分配）
    assert en_vocab[unk_token] == de_vocab[unk_token]
    assert en_vocab[pad_token] == de_vocab[pad_token]
    
    unk_index = en_vocab[unk_token]
    pad_index = en_vocab[pad_token] # pad_index 将在后续数据加载时用于填充
    
    en_vocab.set_default_index(unk_index) # 设置查询词汇表外词元时返回的默认索引
    de_vocab.set_default_index(unk_index)
    
    def numericalize_example(example, en_vocab, de_vocab):
        en_ids = en_vocab.lookup_indices(example["en_tokens"]) # 将英文词元列表转换为索引列表
        de_ids = de_vocab.lookup_indices(example["de_tokens"]) # 将德语词元列表转换为索引列表
        return {"en_ids": en_ids, "de_ids": de_ids}

**应用数值化函数并将数据转换为PyTorch张量格式：**

Python

    fn_kwargs = {"en_vocab": en_vocab, "de_vocab": de_vocab}
    
    train_data = train_data.map(numericalize_example, fn_kwargs=fn_kwargs)
    valid_data = valid_data.map(numericalize_example, fn_kwargs=fn_kwargs)
    test_data = test_data.map(numericalize_example, fn_kwargs=fn_kwargs)
    
    data_type = "torch" # 指定输出格式为PyTorch张量
    format_columns = ["en_ids", "de_ids"] # 需要转换为张量的列
    
    train_data = train_data.with_format(
        type=data_type, columns=format_columns, output_all_columns=True
    )
    # ... 对 valid_data 和 test_data 执行类似操作 ...

现在，数据集中的每个样本都包含了`en_ids`和`de_ids`，它们是PyTorch张量形式的词元索引序列。

**创建数据加载器 (DataLoader)：** 由于每个序列的长度可能不同，在将它们组合成一个批次之前，需要将它们填充到相同的长度。这通过自定义的`collate_fn`函数来实现，该函数使用`nn.utils.rnn.pad_sequence`进行填充。 然后，`get_data_loader`函数封装了`DataLoader`的创建过程。

**关键代码：**

Python

    def get_collate_fn(pad_index):
        def collate_fn(batch): # batch是一个包含多个样本的列表
            batch_en_ids = [example["en_ids"] for example in batch]
            batch_de_ids = [example["de_ids"] for example in batch]
            # 对批次中的英文和德文序列进行填充
            batch_en_ids = nn.utils.rnn.pad_sequence(batch_en_ids, padding_value=pad_index)
            batch_de_ids = nn.utils.rnn.pad_sequence(batch_de_ids, padding_value=pad_index)
            return {"en_ids": batch_en_ids, "de_ids": batch_de_ids}
        return collate_fn
    
    def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
        collate_fn = get_collate_fn(pad_index)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn, # 使用自定义的批处理函数
            shuffle=shuffle,       # 是否打乱数据 (训练时通常为True)
        )
        return data_loader
    
    batch_size = 256 # 教程中使用的批处理大小
    train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
    valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
    test_data_loader = get_data_loader(test_data, batch_size, pad_index)

**详细说明：**

*   `nn.utils.rnn.pad_sequence(sequences, padding_value)`: PyTorch提供的函数，用于将一个包含不同长度序列的列表填充为具有相同长度的张量。默认情况下，它会填充到该批次中最长序列的长度。
*   `collate_fn`: `DataLoader`使用这个函数来整理从数据集中取出的样本列表，并将它们组合成一个批次。这里它负责将不同长度的`en_ids`和`de_ids`序列填充到相同长度。

#### 构建Seq2Seq模型 (Building the Seq2Seq Model)

##### 编码器 (Encoder)

编码器使用**双向GRU (Bidirectional GRU)**。

*   **双向性**：在每一层中，有两个RNN：一个前向RNN（从左到右处理嵌入序列）和一个后向RNN（从右到左处理嵌入序列）。
*   **输入**：嵌入后的源语言词元序列。
*   **输出**：
    *   `outputs`: 包含了源序列中每个词元在顶层双向GRU的隐藏状态的拼接（前向和后向）。形状为 `[src_len, batch_size, encoder_hidden_dim * 2]`。
    *   `hidden`: 编码器所有层在最后一个时间步的前向和后向隐藏状态。形状为 `[num_layers * num_directions, batch_size, encoder_hidden_dim]`。
*   **解码器初始隐藏状态的生成**：为了给单向的解码器提供一个单一的初始隐藏状态，教程中将编码器顶层最后一个时间步的前向隐藏状态 (hT→​) 和后向隐藏状态 (hT←​) 拼接起来，然后通过一个线性层和一个`tanh`激活函数得到解码器的初始隐藏状态 s0​。 s0​\=tanh(g(\[hT→​;hT←​\])) （教程提到这与原始论文略有不同，原始论文只用了后向RNN的第一个隐藏状态）。

**关键代码：`Encoder` 类**

Python

    class Encoder(nn.Module):
        def __init__(self, input_dim, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, dropout):
            super().__init__()
            self.embedding = nn.Embedding(input_dim, embedding_dim) # 嵌入层
            self.rnn = nn.GRU(embedding_dim, encoder_hidden_dim, bidirectional=True) # 双向GRU
            # 用于将拼接后的最后时刻的前向和后向隐藏状态转换为解码器的初始隐藏状态
            self.fc = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim) 
            self.dropout = nn.Dropout(dropout)
    
        def forward(self, src):
            # src 形状: [src_len, batch_size]
            embedded = self.dropout(self.embedding(src))
            # embedded 形状: [src_len, batch_size, embedding_dim]
            
            outputs, hidden = self.rnn(embedded)
            # outputs 形状: [src_len, batch_size, encoder_hidden_dim * 2] (前向和后向的拼接)
            # hidden 形状: [num_layers * 2, batch_size, encoder_hidden_dim] 
            # (hidden[-2,:,:] 是顶层前向RNN的最后一个隐藏状态, hidden[-1,:,:] 是顶层后向RNN的第一个隐藏状态)
    
            # 将顶层前向RNN的最后一个隐藏状态和顶层后向RNN的第一个（从后往前看的最后一个）隐藏状态拼接
            # 并通过线性层和tanh激活函数，生成解码器的初始隐藏状态
            hidden_concat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            hidden_transformed = torch.tanh(self.fc(hidden_concat))
            # hidden_transformed (即解码器的初始隐藏状态s0) 形状: [batch_size, decoder_hidden_dim]
            
            return outputs, hidden_transformed # 返回所有时间步的编码器输出和解码器的初始隐藏状态

##### 注意力机制 (Attention)

注意力层接收解码器前一个时间步的隐藏状态 st−1​ 和编码器的所有输出 H（即源序列中每个词元的隐藏状态表示）。它输出一个注意力向量 at​，其长度等于源句子的长度，每个元素介于0和1之间，且总和为1。这个向量表示在预测目标序列的下一个词时，应该对源序列中的哪些词给予更多的关注。

**计算步骤：**

1.  **计算能量 (Energy Et​)**:
    
    *   将解码器前一时刻的隐藏状态 st−1​ (形状 `[batch_size, decoder_hidden_dim]`) 复制 `src_len` 次，使其形状变为 `[batch_size, src_len, decoder_hidden_dim]`。
    *   编码器输出 H (形状 `[src_len, batch_size, encoder_hidden_dim * 2]`) 需要调整为 `[batch_size, src_len, encoder_hidden_dim * 2]`。
    *   将复制后的 st−1​ 和调整后的 H 拼接起来。
    *   将拼接后的张量通过一个线性层 (`attn_fc`) 和一个 `tanh` 激活函数得到能量 Et​。 $E\_t = \\tanh(\\text{attn\_fc}(\[s\_{t-1}; H\]))$ 这里的 st−1​ 是针对每个源序列词元重复的。 能量 Et​ 的形状是 `[batch_size, src_len, decoder_hidden_dim]`。
2.  **计算注意力权重 (a^t​)**:
    
    *   将能量 Et​ 乘以一个可学习的参数向量 v（通过一个无偏置的线性层 `v_fc` 实现，其权重形状为 `[decoder_hidden_dim, 1]`，输入是能量的最后一个维度）。 a^t​\=vEt​ 这一步的输出形状是 `[batch_size, src_len, 1]`，然后 `squeeze(2)` 移除最后一个维度，得到 `[batch_size, src_len]`。
3.  **计算最终注意力分布 (at​)**:
    
    *   将 a^t​ 通过一个softmax层，确保所有注意力权重在0到1之间且总和为1。 at​\=softmax(at​^​) (在`src_len`维度上进行softmax)

**关键代码：`Attention` 类**

Python

    class Attention(nn.Module):
        def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
            super().__init__()
            # 线性层，输入是拼接后的 (编码器隐藏状态 + 解码器隐藏状态)，输出是解码器隐藏维度
            self.attn_fc = nn.Linear(
                (encoder_hidden_dim * 2) + decoder_hidden_dim, decoder_hidden_dim
            )
            # 线性层，将能量向量映射到一个标量
            self.v_fc = nn.Linear(decoder_hidden_dim, 1, bias=False)
    
        def forward(self, hidden, encoder_outputs):
            # hidden (s_t-1) 形状: [batch_size, decoder_hidden_dim]
            # encoder_outputs (H) 形状: [src_len, batch_size, encoder_hidden_dim * 2]
            
            batch_size = encoder_outputs.shape[1]
            src_length = encoder_outputs.shape[0]
    
            # 重复解码器隐藏状态 src_length 次
            # hidden 初始: [batch_size, dec_hid_dim]
            # hidden.unsqueeze(1): [batch_size, 1, dec_hid_dim]
            # hidden.unsqueeze(1).repeat(1, src_length, 1): [batch_size, src_len, dec_hid_dim]
            repeated_hidden = hidden.unsqueeze(1).repeat(1, src_length, 1)
            
            # 调整编码器输出的维度顺序以匹配拼接
            # encoder_outputs 初始: [src_len, batch_size, enc_hid_dim * 2]
            # encoder_outputs.permute(1, 0, 2): [batch_size, src_len, enc_hid_dim * 2]
            permuted_encoder_outputs = encoder_outputs.permute(1, 0, 2)
    
            # 拼接解码器隐藏状态和编码器输出
            # concat_input 形状: [batch_size, src_len, (enc_hid_dim * 2) + dec_hid_dim]
            concat_input = torch.cat((repeated_hidden, permuted_encoder_outputs), dim=2)
            
            # 计算能量
            # energy 形状: [batch_size, src_len, dec_hid_dim]
            energy = torch.tanh(self.attn_fc(concat_input))
            
            # 计算注意力权重 (未归一化)
            # attention 形状: [batch_size, src_len, 1] -> squeeze(2) -> [batch_size, src_len]
            attention_scores = self.v_fc(energy).squeeze(2)
            
            # 应用softmax得到最终的注意力分布
            return torch.softmax(attention_scores, dim=1)

##### 解码器 (Decoder)

解码器也使用GRU，并结合注意力机制。 **解码过程的每一步：**

1.  **获取注意力权重 (at​)**: 使用解码器前一时刻的隐藏状态 st−1​ 和编码器的所有输出 H，通过注意力层计算得到注意力向量 at​。
2.  **计算加权源向量 (wt​)**: 使用 at​ 对编码器输出 H 进行加权求和，得到 wt​\=at​H。这里使用`torch.bmm` (批处理矩阵乘法) 来实现。
    *   at​ 形状： `[batch_size, 1, src_len]` (通过`unsqueeze(1)`添加一个维度)
    *   H （调整后）形状： `[batch_size, src_len, encoder_hidden_dim * 2]`
    *   wt​ 形状： `[batch_size, 1, encoder_hidden_dim * 2]`
3.  **准备RNN输入**: 将当前时间步的输入词元嵌入 d(yt​) (形状 `[1, batch_size, embedding_dim]`) 和加权源向量 wt​ (调整为 `[1, batch_size, encoder_hidden_dim * 2]`) 拼接起来，作为解码器GRU的输入。
4.  **GRU计算**: 将拼接后的输入和前一时刻的解码器隐藏状态 st−1​ (调整为 `[1, batch_size, decoder_hidden_dim]`) 输入到GRU中，得到当前时刻的输出 `output` 和新的隐藏状态 `hidden`。 st​\=DecoderGRU(\[d(yt​);wt​\],st−1​) （对于单层GRU，`output` 和 `hidden` 在这个上下文中是相同的，只是形状可能略有不同）。
5.  **预测下一个词 (y^​t+1​)**: 将当前输入词元的嵌入 d(yt​)、加权源向量 wt​ 和当前解码器GRU的输出 st​（或`hidden`）拼接起来，然后通过一个全连接层 (`fc_out`) 得到预测的词元概率分布。 y^​t+1​\=f(\[d(yt​);wt​;st​\])

**关键代码：`Decoder` 类**

Python

    class Decoder(nn.Module):
        def __init__(self, output_dim, embedding_dim, encoder_hidden_dim, 
                     decoder_hidden_dim, dropout, attention):
            super().__init__()
            self.output_dim = output_dim # 目标词汇表大小
            self.attention = attention   # 传入实例化的Attention层
            self.embedding = nn.Embedding(output_dim, embedding_dim) # 目标语言嵌入层
            # GRU的输入维度是 (编码器输出维度 + 目标嵌入维度)
            self.rnn = nn.GRU((encoder_hidden_dim * 2) + embedding_dim, decoder_hidden_dim)
            # 输出层的输入维度是 (编码器输出维度 + 解码器隐藏维度 + 目标嵌入维度)
            self.fc_out = nn.Linear(
                (encoder_hidden_dim * 2) + decoder_hidden_dim + embedding_dim, output_dim
            )
            self.dropout = nn.Dropout(dropout)
    
        def forward(self, input, hidden, encoder_outputs):
            # input (y_t) 形状: [batch_size] (当前输入词元的ID)
            # hidden (s_t-1) 形状: [batch_size, decoder_hidden_dim]
            # encoder_outputs (H) 形状: [src_len, batch_size, encoder_hidden_dim * 2]
    
            input = input.unsqueeze(0) # -> [1, batch_size] (GRU期望序列长度作为第一维)
            
            embedded = self.dropout(self.embedding(input)) # -> [1, batch_size, embedding_dim]
            
            a = self.attention(hidden, encoder_outputs) # a 形状: [batch_size, src_len]
            a_unsqueezed = a.unsqueeze(1) # -> [batch_size, 1, src_len] (为bmm做准备)
            
            permuted_encoder_outputs = encoder_outputs.permute(1, 0, 2) # -> [batch_size, src_len, enc_hid_dim * 2]
            
            # 计算加权源向量 w_t
            weighted = torch.bmm(a_unsqueezed, permuted_encoder_outputs) # -> [batch_size, 1, enc_hid_dim * 2]
            weighted_permuted = weighted.permute(1, 0, 2) # -> [1, batch_size, enc_hid_dim * 2] (为GRU输入做准备)
            
            # 准备GRU的输入 (拼接嵌入词元和加权源向量)
            rnn_input = torch.cat((embedded, weighted_permuted), dim=2) 
            # rnn_input 形状: [1, batch_size, (enc_hid_dim * 2) + embedding_dim]
            
            # 输入到GRU
            # hidden.unsqueeze(0) 将解码器隐藏状态从 [batch_size, dec_hid_dim] -> [1, batch_size, dec_hid_dim]
            # 以匹配GRU对多层或双向的期望输入格式（即使这里是单层单向）
            output, hidden_new = self.rnn(rnn_input, hidden.unsqueeze(0))
            # output 形状: [1, batch_size, decoder_hidden_dim] (因为输入序列长度为1)
            # hidden_new 形状: [1, batch_size, decoder_hidden_dim] (对于单层GRU，与output相同)
            
            assert (output == hidden_new).all() # 确保output和hidden_new相同
    
            # 准备最终输出层的输入，移除多余的序列长度维度
            embedded_squeezed = embedded.squeeze(0)       # -> [batch_size, embedding_dim]
            output_squeezed = output.squeeze(0)           # -> [batch_size, decoder_hidden_dim]
            weighted_squeezed = weighted.squeeze(0)       # -> [batch_size, enc_hid_dim * 2]
            
            # 拼接 d(y_t), w_t, s_t
            concat_for_fc = torch.cat((output_squeezed, weighted_squeezed, embedded_squeezed), dim=1)
            
            prediction = self.fc_out(concat_for_fc) # -> [batch_size, output_dim]
            
            # 返回预测，新的解码器隐藏状态，和注意力权重（用于可视化或分析）
            return prediction, hidden_new.squeeze(0), a 

##### Seq2Seq 模型

将编码器和解码器组合在一起构成完整的Seq2Seq模型。 这个类与之前教程中的Seq2Seq类似，主要区别在于：

*   编码器现在返回编码器所有时间步的隐藏状态 (`encoder_outputs`) 和一个用于解码器初始化的隐藏状态 (`hidden`)。
*   这些都需要传递给解码器。

**前向传播过程：**

1.  创建一个张量`outputs`来存储解码器在每个时间步的预测。
2.  源序列 X 输入编码器，得到编码器所有隐藏状态 H 和解码器的初始隐藏状态 s0​ (来自编码器的`hidden`输出)。
3.  解码器的第一个输入 y1​ 是 `<sos>` 标记。
4.  循环解码：
    *   将当前输入词元 yt​、前一解码器隐藏状态 st−1​ 和编码器所有输出 H 输入解码器。
    *   接收到预测 y^​t+1​、新的解码器隐藏状态 st​ 和注意力权重 at​。
    *   存储预测 y^​t+1​。
    *   根据**教师强制 (teacher forcing)** 概率决定下一个输入 yt+1​：
        *   如果使用教师强制，则下一个输入是目标序列中的真实下一个词元。
        *   否则，下一个输入是当前预测概率最高的词元。

**关键代码：`Seq2Seq` 类**

Python

    class Seq2Seq(nn.Module):
        def __init__(self, encoder, decoder, device):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.device = device
    
        def forward(self, src, trg, teacher_forcing_ratio):
            # src 形状: [src_len, batch_size] (德语，源语言)
            # trg 形状: [trg_len, batch_size] (英语，目标语言)
            
            batch_size = src.shape[1]
            trg_length = trg.shape[0]
            trg_vocab_size = self.decoder.output_dim
            
            outputs_tensor = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
            
            encoder_outputs, hidden = self.encoder(src)
            # encoder_outputs 形状: [src_len, batch_size, encoder_hidden_dim * 2]
            # hidden (解码器初始隐藏状态s0) 形状: [batch_size, decoder_hidden_dim]
            
            # 解码器的第一个输入是 <sos> 标记
            decoder_input_t = trg[0, :] # 形状: [batch_size]
            
            for t in range(1, trg_length): # 从目标序列的第二个词元开始预测
                # decoder_output: [batch_size, output_dim]
                # hidden: [batch_size, decoder_hidden_dim] (新的解码器隐藏状态 s_t)
                # attention_weights: [batch_size, src_len]
                decoder_output_t, hidden, attention_weights_t = self.decoder(
                    decoder_input_t, hidden, encoder_outputs
                )
                
                outputs_tensor[t] = decoder_output_t # 存储当前时间步的预测
                
                teacher_force_decision = random.random() < teacher_forcing_ratio
                
                top1_predicted_token = decoder_output_t.argmax(1) # 预测概率最高的词元ID
                
                # 如果是教师强制，则下一个输入是真实的目标词元；否则是模型预测的词元
                decoder_input_t = trg[t] if teacher_force_decision else top1_predicted_token
                
            return outputs_tensor

#### 训练Seq2Seq模型 (Training the Seq2Seq Model)

**初始化模型参数、优化器和损失函数：**

*   定义模型的各种维度（输入、嵌入、隐藏层等）和dropout率。
*   实例化`Attention`, `Encoder`, `Decoder`, 和 `Seq2Seq`模型，并将模型移到GPU（如果可用）。
*   定义权重初始化函数`init_weights`（将权重从均值为0、标准差为0.01的正态分布中初始化，偏置初始化为0），并应用到模型上。
*   定义`count_parameters`函数计算模型可训练参数的数量。
*   使用Adam优化器。
*   损失函数使用交叉熵损失`nn.CrossEntropyLoss`，并设置`ignore_index=pad_index`，这样在计算损失时会忽略填充标记。

**关键代码：模型初始化**

Python

    input_dim = len(de_vocab)    # 源语言（德语）词汇表大小
    output_dim = len(en_vocab)   # 目标语言（英语）词汇表大小
    encoder_embedding_dim = 256
    decoder_embedding_dim = 256
    encoder_hidden_dim = 512     # 编码器GRU的隐藏维度
    decoder_hidden_dim = 512     # 解码器GRU的隐藏维度
    encoder_dropout = 0.5
    decoder_dropout = 0.5
    # device 在前面已定义
    
    attention_layer = Attention(encoder_hidden_dim, decoder_hidden_dim)
    encoder_model = Encoder(input_dim, encoder_embedding_dim, encoder_hidden_dim, 
                            decoder_hidden_dim, encoder_dropout)
    decoder_model = Decoder(output_dim, decoder_embedding_dim, encoder_hidden_dim, 
                            decoder_hidden_dim, decoder_dropout, attention_layer)
    
    model = Seq2Seq(encoder_model, decoder_model, device).to(device)
    
    def init_weights(m: nn.Module): # 类型提示
        for name, param in m.named_parameters():
            if "weight" in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)
    model.apply(init_weights)
    
    def count_parameters(model: nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {count_parameters(model):,} trainable parameters")
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

**训练和评估循环的辅助函数：**

*   `train_fn`: 执行一个epoch的训练。
    *   将模型设为训练模式。
    *   迭代训练数据加载器。
    *   将源序列和目标序列输入模型，获取输出。
    *   **重要**：在计算损失时，目标序列`trg`和模型输出`output`都需要去掉第一个词元（通常是`<sos>`标记或用于预测的第一个词元），因为模型的目标是从第二个词元开始预测。并且需要将它们调整为二维张量以适应交叉熵损失函数。
    *   进行反向传播和参数更新。
    *   梯度裁剪。
*   `evaluate_fn`: 执行在验证集或测试集上的评估。
    *   将模型设为评估模式。
    *   关闭教师强制（`teacher_forcing_ratio = 0`）。
    *   计算平均损失。
*   `epoch_time`: 计算每个epoch的耗时。

**关键代码：训练和评估函数**

Python

    def train_fn(model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device):
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(data_loader):
            src = batch["de_ids"].to(device) # 源序列 (德语)
            trg = batch["en_ids"].to(device) # 目标序列 (英语)
            
            optimizer.zero_grad()
            output = model(src, trg, teacher_forcing_ratio) # output形状: [trg_len, batch_size, output_dim]
            
            output_dim_size = output.shape[-1]
            # 调整output和trg的形状以匹配CrossEntropyLoss的期望输入
            # output: 从第二个时间步开始，因为第一个输入是<sos>，我们预测的是<sos>之后的词
            # trg: 也从第二个时间步开始，作为预测的目标
            # view(-1, output_dim_size) 将 [trg_len-1, batch_size, output_dim] 变为 [(trg_len-1)*batch_size, output_dim]
            output_reshaped = output[1:].view(-1, output_dim_size)
            # trg_reshaped: [(trg_len-1)*batch_size]
            trg_reshaped = trg[1:].view(-1)
            
            loss = criterion(output_reshaped, trg_reshaped)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip) # 梯度裁剪
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(data_loader)
    
    def evaluate_fn(model, data_loader, criterion, device):
        model.eval()
        epoch_loss = 0
        with torch.no_grad(): # 评估时不需要计算梯度
            for i, batch in enumerate(data_loader):
                src = batch["de_ids"].to(device)
                trg = batch["en_ids"].to(device)
                output = model(src, trg, 0) # teacher_forcing_ratio = 0 表示不使用教师强制
                
                output_dim_size = output.shape[-1]
                output_reshaped = output[1:].view(-1, output_dim_size)
                trg_reshaped = trg[1:].view(-1)
                
                loss = criterion(output_reshaped, trg_reshaped)
                epoch_loss += loss.item()
        return epoch_loss / len(data_loader)
    
    # ... epoch_time 函数定义 ...

**执行训练：** 设置训练参数（如epoch数，梯度裁剪值，教师强制比例），然后在一个循环中调用`train_fn`和`evaluate_fn`。模型参数会在验证损失达到最佳时保存。

Python

    n_epochs = 10
    clip = 1.0
    teacher_forcing_ratio = 0.5
    best_valid_loss = float("inf") # 初始化最佳验证损失为无穷大
    
    for epoch in tqdm.tqdm(range(n_epochs)): # 使用tqdm显示进度
        # start_time = time.time() # 需要导入time模块
        
        train_loss = train_fn(model, train_data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device)
        valid_loss = evaluate_fn(model, valid_data_loader, criterion, device)
        
        # end_time = time.time()
        # epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "tut-model.pt") # 保存模型状态
        
        print(f"\\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
        print(f"\\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")
        # print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s') # 打印耗时

**输出解释：** 在每个epoch结束时，会打印训练损失、训练困惑度 (PPL, Perplexity = eloss)、验证损失和验证困惑度。困惑度是衡量语言模型性能的常用指标，值越低越好。

**在测试集上评估最终模型：** 加载具有最佳验证损失的模型参数，并在测试集上进行评估。

Python

    model.load_state_dict(torch.load("tut-model.pt")) # 加载保存的最佳模型参数
    test_loss = evaluate_fn(model, test_data_loader, criterion, device)
    print(f"| Test Loss: {test_loss:.3f} | Test PPL: {np.exp(test_loss):7.3f} |")

这个教程详细介绍了如何使用PyTorch和TorchText实现一个带注意力机制的Seq2Seq模型用于机器翻译，包括数据预处理、模型构建（编码器、注意力、解码器）、参数初始化、训练循环和评估。核心是理解注意力机制如何帮助解码器在生成目标序列的每个词元时，动态地关注源序列的不同部分。

信息来源和相关内容

[

discuss.pytorch.org

](https://discuss.pytorch.org/t/lstm-always-has-constant-loss-not-learning/157897)

[

discuss.pytorch.org

](https://discuss.pytorch.org/t/lstm-always-has-constant-loss-not-learning/157897)

[

tutorial\_w...odel.ipynb

![IPYNB 图标](https://drive-thirdparty.googleusercontent.com/32/type/text/code)

IPYNB







](https://contribution.usercontent.google.com/download?c=CgxiYXJkX3N0b3JhZ2USTxIMcmVxdWVzdF9kYXRhGj8KMDAwMDYzNGRhOTZkODUzZWMwN2Y1NWU2NmY0MjY4M2NmODQ0NGZlYjhlZjBjMjE4OBILEgcQ57PmiuAQGAE&filename&opi=103135050)

[

tutorial\_w...ng\_2.ipynb

![IPYNB 图标](https://drive-thirdparty.googleusercontent.com/32/type/text/code)

IPYNB







](https://contribution.usercontent.google.com/download?c=CgxiYXJkX3N0b3JhZ2USTxIMcmVxdWVzdF9kYXRhGj8KMDAwMDYzNGRhOWU1ZGQ2MmIwMWFmYWUyZDdkMzMzNGFmYTQwOWU5N2RiMDY3YmI3YRILEgcQ57PmiuAQGAE&filename&opi=103135050)

[

tutorial\_w...ysis.ipynb

![IPYNB 图标](https://drive-thirdparty.googleusercontent.com/32/type/text/code)

IPYNB







](https://contribution.usercontent.google.com/download?c=CgxiYXJkX3N0b3JhZ2USTxIMcmVxdWVzdF9kYXRhGj8KMDAwMDYzNGRhYTc1YTQ4NTAwMmE4OGIxMzQ2MGVlYzVkYzI0ODgwNTVkOGFkOTA5MRILEgcQ57PmiuAQGAE&filename&opi=103135050)

[

tutorial\_w...tion.ipynb

![IPYNB 图标](https://drive-thirdparty.googleusercontent.com/32/type/text/code)

IPYNB







](https://contribution.usercontent.google.com/download?c=CgxiYXJkX3N0b3JhZ2USTxIMcmVxdWVzdF9kYXRhGj8KMDAwMDYzNGRhYjNmYjliODcwN2Y1NjlmOGQxMzlhNWQ0NjFmZWJmMTYyNDJhMmQyMxILEgcQ57PmiuAQGAE&filename&opi=103135050)

---

![IPYNB 图标](https://drive-thirdparty.googleusercontent.com/32/type/text/code)

tutorial\_week7.ipynb

IPYNB

同样地，讲解该文件

---

显示思路

好的，我们来详细讲解这个关于\*\*“用于文本生成的语言模型”\*\*的Jupyter Notebook教程。

### CS6493 - 教程 7: 用于文本生成的语言模型

#### 引言 (Introduction)

本教程主要介绍如何使用Hugging Face Transformers库微调一个语言模型用于文本生成。 教程重点介绍了两种类型的语言建模：

1.  **因果语言建模 (Causal Language Modeling - CLM)**:
    *   预测序列中的下一个词元。
    *   模型只能关注当前词元左侧的词元（即不能看到未来的词元）。
    *   常用于文本生成任务，例如：选择你自己的文本冒险游戏、智能代码助手（如Copilot、CodeParrot）。
    *   GPT-2是因果语言模型的一个例子。
2.  **掩码语言建模 (Masked Language Modeling - MLM)**: (本教程主要关注CLM，但引言中提到了作为对比)
    *   预测输入序列中被掩盖的某些词元。
    *   模型可以关注到被掩盖词元左右两侧的上下文。

**本教程将展示如何：**

1.  在[ELI5数据集](https://www.google.com/search?q=https://huggingface.co/datasets/eli5_category)的[r/askscience](https://www.reddit.com/r/askscience/)子集上微调[DistilGPT2模型](https://huggingface.co/distilgpt2)。
2.  使用微调后的模型进行推理（文本生成）。

教程提示，可以使用本指南中相同的步骤来微调其他支持因果语言建模的架构，例如OpenAI GPT, OpenAI GPT-2, OPT, Llama, CodeLlama等。

#### 准备数据 (Preparing Data)

**安装必要的库：** 首先，确保安装了如`accelerate`, `transformers`, `datasets`, `evaluate`等必要的库。

Python

    ! pip install accelerate
    ! pip install transformers datasets evaluate # 教程中可能分步安装，但通常这些是核心依赖

**加载ELI5数据集：** 教程从Hugging Face Datasets库加载ELI5数据集的`train`分割中r/askscience分类的前500个样本，以进行快速实验。

Python

    from datasets import load_dataset
    
    # 加载eli5_category数据集的'train'分割中前500个样本
    eli5 = load_dataset("eli5_category", split="train[:500]")

**划分训练集和测试集：** 使用`train_test_split`方法将加载的数据集划分为训练集和测试集（验证集），测试集占20%。

Python

    eli5 = eli5.train_test_split(test_size=0.2)
    # eli5现在是一个DatasetDict，包含'train'和'test'两个分割

**查看数据样本：** 打印一个训练样本来了解其结构。ELI5数据集的样本通常包含多个字段，如`q_id`（问题ID）、`title`（问题标题）、`selftext`（问题正文）、`category`（类别）、`subreddit`（子Reddit版块）、`answers`（一个包含答案信息的字典，其中包括`a_id`答案ID, `text`答案文本列表, `score`答案得分等）。

Python

    print(eli5["train"][0])

对于语言建模任务，主要关注的是`answers.text`字段，因为模型的目标是学习生成类似这样的文本。语言建模是无监督的，因为下一个词本身就是标签。

#### 预处理 (Preprocess)

**加载分词器 (Tokenizer)：** 使用`AutoTokenizer`加载预训练的`distilgpt2`模型对应的分词器。`use_fast=True`（如果可用）会加载一个用Rust实现的“快速”分词器，通常处理速度更快。

Python

    from transformers import AutoTokenizer
    
    model_checkpoint = "distilgpt2" # 指定要使用的预训练模型
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

**数据扁平化 (Flatten)：** ELI5数据集中，`answers.text`是一个列表的列表（因为一个问题可能有多个答案，每个答案可能由多个文本片段组成）。需要使用`flatten()`方法将其扁平化，使得`answers.text`变成一个包含所有答案文本片段的单一列表，方便后续处理。

Python

    eli5 = eli5.flatten()
    # 执行后，eli5["train"][0]的结构会改变，例如 'answers.text' 会从一个嵌套列表变为一个扁平列表。
    print(eli5["train"][0])

**定义第一个预处理函数 `preprocess_function`：** 这个函数的作用是将每个样本中`answers.text`列表里的所有字符串连接成一个单一的字符串，然后使用分词器对这个连接后的字符串进行分词。

Python

    def preprocess_function(examples):
        # examples["answers.text"] 是一个列表的列表，其中每个内部列表包含一个答案的文本片段
        # " ".join(x) for x in examples["answers.text"] 会将每个答案的文本片段连接起来
        # tokenizer(...) 会对连接后的字符串进行分词
        return tokenizer([" ".join(x) for x in examples["answers.text"]])

**应用预处理函数：** 使用`dataset.map()`方法将`preprocess_function`应用到数据集的所有分割上。`batched=True`允许一次处理多个样本，`num_proc=4`使用4个进程并行处理以加速。处理后移除不再需要的原始列。

Python

    tokenized_eli5 = eli5.map(
        preprocess_function,
        batched=True,
        num_proc=4, # 使用多进程加速
        remove_columns=eli5["train"].column_names, # 移除原始列，只保留分词后的结果 (input_ids, attention_mask)
    )
    # tokenized_eli5["train"][1]会显示分词后的结果，包含'input_ids'和'attention_mask'

\*\*注意：\*\*此时，每个样本的`input_ids`长度可能不同，并且可能超过模型的最大输入长度。

**定义文本分组函数 `group_texts` (关键步骤)：** 因果语言模型的训练通常需要固定长度的文本块。这个函数执行以下操作：

1.  **拼接所有文本 (Concatenate all texts)：** 将数据集中所有样本的`input_ids`（或其他指定键的值）拼接成一个非常长的序列。
2.  **计算总长度 (Total length)：** 获取拼接后序列的总长度。
3.  **按块大小切片 (Split by chunks of block\_size)：**
    *   将总长度调整为`block_size`的整数倍，丢弃末尾不足一个块的部分。
    *   将拼接后的长序列按`block_size`切分成多个固定长度的块。
4.  **创建标签 (Create labels)：** 对于因果语言模型，标签通常就是输入本身（因为模型需要预测下一个词）。Hugging Face的`Trainer`会自动处理标签的右移，所以这里直接复制`input_ids`作为`labels`。

<!-- end list -->

Python

    block_size = 128 # 定义每个文本块的长度，可以根据模型最大长度和GPU内存调整
    
    def group_texts(examples):
        # 将一个批次中所有样本的某个键（如input_ids）对应的值（列表的列表）合并成一个单一的长列表
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]]) # 获取拼接后序列的总长度
    
        # 丢弃末尾不足一个block_size的部分，使得总长度是block_size的整数倍
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        
        # 将长序列切分成多个长度为block_size的块
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        # 为因果语言模型创建标签，标签即输入本身
        result["labels"] = result["input_ids"].copy()
        return result

**应用文本分组函数：** 再次使用`map`方法，将`group_texts`函数应用到已经分词的数据集上。

Python

    lm_datasets = tokenized_eli5.map(
        group_texts,
        batched=True,      # 每次处理一批数据
        batch_size=1000,   # map函数内部处理的批次大小
        num_proc=4,        # 使用多进程
    )
    # tokenizer.decode(lm_datasets["train"][1]["input_ids"]) 可以用来查看一个处理后的样本被解码回文本的样子

经过这个步骤后，`lm_datasets`中的每个样本都是一个长度为`block_size`的`input_ids`序列和对应的`labels`序列。

**数据整理器 (Data Collator)：** 使用`DataCollatorForLanguageModeling`。这个整理器在因果语言建模（当`mlm=False`时）中会自动创建向右移位的标签，这是训练语言模型预测下一个词所必需的。它也负责将批次中的样本动态填充到该批次内最长序列的长度（如果之前的处理没有保证所有块都是完全相同的`block_size`，或者在批处理时需要）。 `tokenizer.pad_token = tokenizer.eos_token` 将填充标记设置为与句子结束标记相同，这对于GPT-2这类模型是常见的做法。

Python

    from transformers import DataCollatorForLanguageModeling
    
    tokenizer.pad_token = tokenizer.eos_token # 将PAD token设置为EOS token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    # mlm=False 表示是因果语言建模，而不是掩码语言建模

#### 训练 (Train)

**加载预训练模型：** 使用`AutoModelForCausalLM`加载`distilgpt2`模型。这个类会自动加载一个带有因果语言建模头的预训练模型。

Python

    from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
    
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

**定义训练参数 `TrainingArguments`：** 这个类包含了所有用于配置训练过程的参数。

Python

    model_name = model_checkpoint.split("/")[-1] # 从checkpoint路径获取模型名称
    training_args = TrainingArguments(
        output_dir=f"{model_name}-finetuned-wikitext2", # 微调后模型的输出目录
        evaluation_strategy="epoch",    # 每个epoch结束时进行评估
        learning_rate=2e-5,             # 学习率
        weight_decay=0.01,              # 权重衰减，用于正则化
        num_train_epochs=1,             # 教程中设置为1个epoch进行演示
        # push_to_hub=False,            # 是否推送到Hugging Face Hub (可选)
        report_to="none"                # 禁止向wandb等外部服务报告训练日志 (可选)
    )

**详细说明：**

*   `output_dir`: 指定保存模型检查点、日志等文件的目录。
*   `evaluation_strategy="epoch"`: 设置评估策略为每个训练轮次（epoch）结束时进行一次评估。可选值还有 "steps"（每N步评估一次）。
*   `learning_rate`: 优化器的初始学习率。
*   `weight_decay`: 权重衰减系数，L2正则化的一种形式，有助于防止过拟合。
*   `num_train_epochs`: 训练的总轮数。
*   `push_to_hub`: 如果设置为`True`，并且用户已登录Hugging Face CLI，则会在训练结束或保存时将模型推送到Hugging Face Model Hub。
*   `report_to`: 指定报告日志的目标，如 "wandb", "tensorboard"。设置为 "none" 则不使用这些外部日志工具。

**实例化 `Trainer`：** `Trainer`类是Hugging Face提供的一个高级API，它封装了训练和评估循环的复杂逻辑。

Python

    trainer = Trainer(
        model=model,                     # 要训练的模型
        args=training_args,              # 训练参数
        train_dataset=lm_datasets["train"],# 训练数据集
        eval_dataset=lm_datasets["test"],  # 评估数据集 (教程中用的是之前划分的test集作为验证集)
        data_collator=data_collator,     # 数据整理器
    )

**开始训练：** 调用`trainer.train()`方法开始微调模型。

Python

    # trainer.train() # 教程中没有实际执行这一步以节省时间

**评估模型：** 训练完成后（或加载一个已训练的模型后），调用`trainer.evaluate()`在评估数据集上计算模型的损失。然后可以计算**困惑度 (Perplexity)**，这是衡量语言模型性能的常用指标，值越低越好。困惑度是交叉熵损失的指数。

Python

    import math
    
    # 假设已经训练或加载了模型
    # eval_results = trainer.evaluate() # 在评估集上运行评估
    # print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}") 
    # 教程给出的示例输出是 Perplexity: 40.01 (这通常需要实际训练才能得到)

教程提供了一个示例输出，微调后的DistilGPT2在Wikitext-2上的困惑度大约是40.01。

#### 推理 (Inference)

微调完成后，可以使用模型进行文本生成。 最简单的方法是使用Hugging Face的`pipeline`。创建一个`text-generation`的pipeline，并传入微调后的模型和分词器。

Python

    from transformers import pipeline
    
    prompt = "Somatic hypermutation allows the immune system to" # 定义一个提示文本
    
    # 使用微调后的模型创建文本生成pipeline
    # 如果trainer.train()被执行了，这里的model就是微调后的模型
    # 否则，这里的model是原始的distilgpt2
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device) # 指定设备
    # device=0 (如果用GPU) 或 device=-1 (如果用CPU)
    # 在新版本中，可以直接传入 device 对象
    
    generated_text = generator(prompt)
    print(generated_text)

**输出示例：** 输出会是一个列表，包含生成的文本。例如：

[{'generated_text': 'Somatic hypermutation allows the immune system to generate the immune system to generate the defense systems to function. When antibodies target immune cells, immune cells start to suppress the attack. Once the immune system gets weakened, the immune system becomes more vulnerable.'}]

这表明模型基于给定的提示成功生成了连贯的文本。

#### 通过OpenAI API进行推理 (Inference via OpenAI API) - (这部分是额外内容，与教程核心的微调DistilGPT2无关)

教程的最后部分展示了如何使用类似OpenAI的API（这里用的是DeepSeek的API作为示例）来进行文本生成。这通常涉及到：

1.  安装相应的客户端库（如`openai`）。
2.  设置API密钥和基础URL。
3.  调用`client.chat.completions.create`方法，传入模型名称、消息（通常包含系统提示和用户提示）以及其他参数（如`temperature`, `top_p`）来控制生成文本的多样性。

**解码参数解释：**

*   **Temperature (温度)**:
    *   调整词汇表概率分布的参数 t。公式为 P(x∣x\_1:t−1)\=fracexp(u\_t/t)sum\_t′exp(u\_t′/t)。
    *   tto0: 变为贪心解码 (greedy decoding)，总是选择概率最高的词。
    *   ttoinfty: 变为均匀采样 (uniform sampling)。
    *   调整 t 可以避免从概率分布的尾部进行采样，控制生成文本的随机性。较低的 t (如0.2) 生成更确定、重复性高的文本；较高的 t (如0.8) 生成更多样、更随机的文本。
*   **Top-p Sampling (Nucleus Sampling)**:
    *   选择累积概率达到阈值 p 所需的最少数量的词元。
    *   首先按概率降序排序词元。
    *   计算累积概率，直到超过 p。
    *   只保留贡献到这个累积概率的词元，并重新归一化它们的概率。
    *   较高的 p (如0.9) 允许更多样的输出；较低的 p (如0.1) 产生更集中的输出。
*   **n (候选数量)**:
    *   语言模型生成的候选完成（或回答）的数量。
    *   增加 n 可以获得更多样化的选项，但可能增加API使用量和响应时间。

**示例代码（使用DeepSeek API）：**

Python

    from openai import OpenAI
    
    # 注意：这里的API密钥和base_url是示例，实际使用时需替换
    client = OpenAI(api_key="sk-a4dfb7a4a40141b8a93fe6ee574d5d8f", base_url="https://api.deepseek.com")
    
    response = client.chat.completions.create(
        model="deepseek-chat", # 指定模型
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "tell a short joke"},
        ],
        stream=False, # 是否流式输出
        temperature=0 # 设置较低的温度以获得更确定的回答
    )
    print(response.choices[0].message.content)
    
    # 使用top_p的示例
    response_topp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "tell a short joke"},
        ],
        top_p=0.8 # 使用top_p采样
    )
    print(response_topp.choices[0].message.content)

**解决LLM进行大数计算的问题：** 教程最后提出了一个问题：LLM在处理非常大的数值计算（如大数乘法）时可能会遇到困难。 DeepSeek-R1的回答示例显示，它能够正确计算 `638726816387261768327866 * 3926328917872368391` 这样的超大整数乘法。这表明一些先进的LLM可能内置了或通过工具调用了精确的计算能力，或者其训练数据中包含了大量数学运算的模式。

可能的解决方案包括：

*   **工具使用 (Tool Use)：** LLM可以被设计成调用外部计算器或代码执行环境（如Python解释器）来执行精确的数学运算，然后将结果整合回其回答中。
*   **特定训练 (Specific Training)：** LLM可以通过在包含大量数学问题和解法的语料库上进行训练，以提高其直接进行数值计算的能力，尽管对于任意精度的大数运算，这仍然是一个挑战。
*   **分解问题 (Problem Decomposition)：** 对于复杂计算，LLM可以尝试将其分解成更小的、更容易处理的步骤。
*   **提示工程 (Prompt Engineering)：** 通过精心设计的提示，引导LLM使用更可靠的推理路径或寻求外部工具的帮助。

本教程主要通过Hugging Face Transformers库演示了如何微调DistilGPT2进行因果语言建模，用于文本生成，并简要介绍了如何通过类似OpenAI的API与大型语言模型交互。

信息来源和相关内容

---

![IPYNB 图标](https://drive-thirdparty.googleusercontent.com/32/type/text/code)

tutorial\_8...ning.ipynb

IPYNB

同样地，讲解该文件

---

分析

分析

*   读取文档
    
    已完成回答
    

好的，我们来详细讲解这个关于\*\*“提示学习 (Prompt Learning)”\*\*的Jupyter Notebook教程。

### CS6493 - 教程 8: 提示学习

#### 上下文 (Context)

本教程涵盖以下三个主要部分：

1.  **思维链推理示例 (Examples of Chain of Thought Reasoning)**
2.  **提示学习简介 (Introduction to Prompt Learning)**
3.  **OpenPrompt 的用法 (Usage of OpenPrompt)**

#### 第一部分: 思维链推理 (Part 1: Chain of Thought Reasoning)

这部分主要通过解决数学问题的例子来演示不同类型的思维链 (Chain of Thought, CoT) 推理。

1.  **资源链接：**
    *   教程提到了香港城市大学为学生提供的免费 GPT-4o 链接：[https://www.cityu.edu.hk/GenAI/get-started.htm](https://www.cityu.edu.hk/GenAI/get-started.htm)
2.  **以解决数学问题为例演示推理类型：**
    *   **零样本推理 (zero-shot reasoning):** 直接向模型提问，不提供任何示例。
        *   **简单问题示例：** "15 + 2 的答案是什么？"
    *   **单样本推理 (one-shot reasoning):** 在提问前给模型一个包含问题和详细解题步骤的示例。
    *   **少样本推理 (few-shot reasoning):** 在提问前给模型多个包含问题和详细解题步骤的示例。
        *   **CoT (思维链) 问题示例 (零样本CoT的引出方式):** "Julia 买了3包红球，10包黄球和8包绿球。每个包装里有19个球。Julie 总共买了多少个球？" 提问后引导模型进行逐步思考 ("Let's think step by step")。
        *   **少样本 CoT 示例：**
            *   **示例1:**
                *   问题: Lexi 总共想跑三又四分之一英里。一个特定的户外跑道一圈是四分之一英里。她必须跑多少个完整的圈？
                *   逐步思考: 3英里中有 3 / (1/4) = 12 个四分之一英里。所以，Lexi 必须跑 12 (来自3英里) + 1 (来自1/4英里) = 13 个完整的圈。答案是 13。
            *   **示例2:**
                *   问题: Asia 买了一件打折的返校节礼服，花了140美元。原价是350美元。她在这次促销中获得了多少折扣百分比？
                *   逐步思考: Asia 在这件礼服上节省了 350美元 - 140美元 = 210美元。这意味着她节省了 210美元 / 350美元 = 0.60 或 60% 的折扣。答案是 60。
            *   **目标问题 (需要模型解答):**
                *   问题: 作为特别款待，Georgia 每个月的第一天都会做松饼带给她的学生。她的松饼食谱一次只能做6个松饼，而她有24个学生。在9个月里，Georgia 总共做了多少批松饼？
                *   逐步思考: (留给模型完成)
    *   **提示模板资源：**
        *   [https://github.com/FranxYao/chain-of-thought-hub](https://github.com/FranxYao/chain-of-thought-hub)
        *   [https://github.com/google/BIG-bench](https://github.com/google/BIG-bench) (包含适用于不同推理任务的多种资源)
    *   **GSM8k 数据集资源 (用于小学数学题):** [https://github.com/openai/grade-school-math/tree/master/grade\_school\_math/data](https://www.google.com/search?q=https://github.com/openai/grade-school-math/tree/master/grade_school_math/data)
3.  **LLM 对提示模板的敏感性：**
    *   大型语言模型 (LLM) 的表现对所使用的提示模板非常敏感。
    *   相关研究论文：[https://arxiv.org/pdf/2201.11903.pdf](https://www.google.com/search?q=https://arxiv.org/pdf/2201.11903.pdf)

#### 第二部分: 提示学习简介 (Part 2: Introduction to Prompt-learning)

**核心概念：**

*   **提示 (Prompt)** 是一种通过在输入中添加额外文本来更好地利用预训练模型知识的技术。
*   这种方法显著提高了预训练模型的使用效率。

**对比传统模型微调 (Model Tuning) 和提示调整 (Prompt Tuning)：**

*   **传统模型微调 (左图所示)：**
    *   针对不同的任务，需要对整个预训练语言模型进行微调。
    *   每个任务都有其自己的一套独立的模型参数。
*   **提示调整 (右图所示)：**
    *   针对不同的任务，仅插入不同的提示参数 (prompt parameters)。
    *   每个任务使用独立的提示参数进行训练，而预训练语言模型的参数保持不变（或只进行少量调整）。
    *   这极大地减少了训练时间，并提高了模型的使用效率和复用性。

教程中包含一张图片（未提供图片数据，但根据描述可知）来形象地展示这两种范式的区别。

这个教程的前两部分主要侧重于概念介绍和示例展示，特别是思维链推理的强大以及提示学习相对于传统微调的优势。第三部分（OpenPrompt的用法）预计会进入具体的代码实践。

信息来源和相关内容

---

![IPYNB 图标](https://drive-thirdparty.googleusercontent.com/32/type/text/code)

Tutorial\_9... (1).ipynb

IPYNB

同样地，讲解该文件，详细一些

---

分析

分析

*   读取文档
    
    已完成回答
    

好的，我们来详细讲解这个关于\*\*“使用检索工具增强LLM进行问答 (Augment LLM with Retrieval Tool for Question-Answering)”\*\*的Jupyter Notebook教程。这个教程主要使用 **LangChain** 框架来构建一个典型的 **RAG (Retrieval Augmented Generation)** 应用。

### 教程 9 - 使用检索工具增强LLM进行问答

#### 引言 (Introduction)

本教程的目标是使用 LangChain 构建一个典型的检索增强生成 (RAG) 应用，专注于问答 (Q&A) 任务。

**核心概念：**

*   **LangChain**:
    *   一个开源的 Python 包。
    *   旨在通过集成各种工具（如搜索引擎、数据库、计算工具等）来增强大型语言模型 (LLMs) 如 GPT-3 的能力。
    *   提供了一个灵活的框架来构建这些集成应用。
*   **RAG (Retrieval Augmented Generation)**:
    *   一种用额外数据增强 LLM 知识的技术。
    *   LLM 虽然能理解和推理广泛的主题，但其知识仅限于训练时所用的公开数据，并且有特定的截止日期。
    *   如果需要构建能够理解私有数据或模型训练截止日期之后的新数据的AI应用，就需要用这些特定信息来增强模型的知识。
    *   RAG 的过程就是获取相关信息并将其插入到模型提示 (prompt) 中的过程。
    *   LangChain 包含了许多专门为构建Q&A应用和更广泛的RAG应用设计的组件。

#### RAG 应用的两个主要组件：

1.  **索引 (Indexing)**:
    
    *   索引组件负责从指定来源摄取数据，并执行必要的步骤来准备数据以便高效检索。
    *   这个过程通常在应用程序运行之前**离线**完成。
    *   主要步骤包括：加载数据 (Load)、分割数据 (Split)、存储数据 (Store)。
2.  **检索和生成 (Retrieval and generation)**:
    
    *   这是 RAG 应用的核心运行时组件。
    *   当用户提供查询时，此组件负责从先前索引的数据集中检索相关数据。
    *   它利用索引结构和算法来有效地识别与给定查询最相关的信息。
    *   一旦检索到相关数据，就将其与原始查询一起传递给底层LLM。
    *   LLM 使用这些检索到的数据和查询来生成一个连贯且与上下文相关的答案。
    *   这个答案最终呈现给用户。

#### 步骤 1: 索引 - 加载 (Indexing: Load)

*   **目的**: 将私有数据加载到 RAG 应用中。
*   **工具**: 使用 LangChain 的 `DocumentLoaders`。
*   **`DocumentLoaders`**: 特殊设计的对象，用于从特定来源检索和加载数据。
*   **本教程示例**: 从 Google Drive 加载文本文件。

**关键代码：安装必要的库和挂载Google Drive** 首先，教程会安装 `langchain`, `sentence-transformers` 和 `langchain_community`。`sentence-transformers` 用于生成文本嵌入，`langchain_community` 包含许多社区贡献的集成。

Python

    ! pip install langchain
    ! pip install sentence-transformers
    ! pip install langchain_community

然后挂载 Google Drive 以访问存储在那里的数据文件。

Python

    import os
    import textwrap # 用于文本自动换行，方便打印
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True) # 强制重新挂载

**加载文档：** 使用`langchain.document_loaders.TextLoader`来加载指定路径的文本文件。

Python

    from langchain.document_loaders import TextLoader # 旧版本路径，新版本可能在 langchain_community.document_loaders
    # 根据教程中的输出，langchain.document_loaders 仍然可用，但新版本倾向于从 langchain_community 导入
    
    text_loader = TextLoader("/content/drive/MyDrive/data.txt") # 假设数据文件名为 data.txt 且在 MyDrive 根目录下
    document = text_loader.load()
    print(document) # 打印加载的文档对象列表（通常是一个包含单个Document对象的列表）

**详细说明：**

*   `TextLoader`: 这是 LangChain 中用于加载纯文本文件的文档加载器。
*   `.load()`: 调用此方法会读取文件内容，并将其封装成一个或多个 `Document` 对象。每个 `Document` 对象通常包含 `page_content` (文本内容) 和 `metadata` (元数据，如来源文件名)。
*   教程中的示例输出显示加载的文档是一个关于一个名为 Amit 的健康意识强的印度男性和他与一个假设的医疗保健系统 Healthify 交互的故事。

#### 步骤 2: 索引 - 分割 (Indexing: Split)

*   **问题**: 加载的文档通常太长，无法直接放入许多LLM的上下文窗口 (context window)。即使模型可以容纳长文本，也可能难以在非常长的输入中找到相关信息。
*   **解决方案**: 将长文档分割成更小的块 (chunks)，以便进行嵌入和向量存储。这样可以在运行时只检索最相关的文本片段。
*   **本教程方法**:
    *   将文档分割成1000个字符的块。
    *   块之间有50个字符的重叠 (overlap)。重叠有助于避免将一个完整的陈述与其重要上下文分离开。
    *   使用 `RecursiveCharacterTextSplitter`，它会递归地使用常见的分隔符（如换行符）来分割文档，直到每个块达到合适的大小。这是通用文本推荐使用的分割器。
    *   设置 `add_start_index=True`，以便将每个分割块在原始文档中的起始字符索引作为元数据属性 "start\_index" 保存下来。

**关键代码：定义辅助函数和分割文档** 首先，定义一个辅助函数 `split_text_into_lines` (虽然教程中定义了，但并未在后续分割中直接使用，可能是为了方便打印或预处理某些特定格式文本)。

Python

    def split_text_into_lines(text, width=110): # width 用于 textwrap
        lines = text.split("\n")
        wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
        wrapped_text = "\n".join(wrapped_lines)
        return wrapped_text

然后，使用`RecursiveCharacterTextSplitter`进行分割。

Python

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # 每个块的目标大小
        chunk_overlap=50,       # 块之间的重叠大小
        add_start_index=True    # 是否添加原始文档中的起始索引作为元数据
    )
    
    document_chunks = text_splitter.split_documents(document) # document 是之前加载的Document对象列表
    print(len(document_chunks)) # 打印分割后的块数量

**详细说明：**

*   `RecursiveCharacterTextSplitter`: 尝试按一系列分隔符（默认是 `["\n\n", "\n", " ", ""]`）进行分割。它首先尝试用第一个分隔符分割，如果块仍然太大，则用下一个分隔符，以此类推。
*   `chunk_size`: 控制每个块的最大字符数。
*   `chunk_overlap`: 控制相邻块之间共享的字符数。这有助于确保语义连贯性，避免在块的边界处切断重要的上下文。
*   `add_start_index`: 如果为 `True`，每个生成的块的元数据中会包含一个 `start_index` 字段，指明该块在原始文档中的起始位置。
*   `split_documents()`: 接收一个 `Document` 对象列表，并返回一个分割后的 `Document` 对象列表。
*   教程示例输出 `len(document_chunks)` 为3，表示原始文档被分成了3个块。

#### 步骤 3: 索引 - 存储 (Indexing: Store)

*   **目的**: 对文档块进行索引，以便在运行时可以搜索它们。
*   **常用方法**:
    1.  嵌入 (Embed) 每个文档块的内容，将其转换为向量表示。
    2.  将这些嵌入向量插入到向量数据库 (Vector Database) 或向量存储 (Vector Store) 中。
*   **搜索过程**: 当需要搜索时：
    1.  获取文本搜索查询。
    2.  嵌入该查询，得到查询向量。
    3.  执行某种“相似性”搜索（通常是余弦相似度或欧氏距离），以识别向量数据库中与查询向量最相似的已存储文档块的嵌入。

**本教程方法：**

*   使用 `chromadb` 作为向量存储。
*   使用 `HuggingFaceEmbeddings` 模型来生成文本嵌入。
*   `HuggingFaceEmbeddings`: LangChain 中的一个组件，封装了文本嵌入模型，负责将文本输入转换为密集向量表示。教程中使用了默认的 Hugging Face 嵌入模型 (通常是 `sentence-transformers/all-mpnet-base-v2` 或类似的通用模型)。
*   `VectorStore`: LangChain 中的组件，封装了向量数据库，专门用于存储和查询嵌入。它提供了与底层数据库交互的接口。

**关键代码：安装ChromaDB并创建向量存储** 首先，安装 `chromadb`。

Python

    !pip install chromadb

然后，从文档块创建向量存储。

Python

    from langchain.embeddings import HuggingFaceEmbeddings # 旧版本路径
    # from langchain_community.embeddings import HuggingFaceEmbeddings # 推荐的新版本路径
    from langchain.vectorstores import Chroma # 旧版本路径
    # from langchain_community.vectorstores import Chroma # 推荐的新版本路径
    
    # 初始化嵌入模型
    embeddings = HuggingFaceEmbeddings() # 默认会从Hugging Face Hub下载一个预训练的句子嵌入模型
    
    # 从文档块创建Chroma向量存储
    vector_store = Chroma.from_documents(
        documents=document_chunks, # 之前分割好的文档块列表
        embedding=embeddings       # 用于生成嵌入的嵌入模型实例
    )

**详细说明：**

*   `HuggingFaceEmbeddings()`: 当不指定 `model_name` 时，它会使用一个默认的预训练句子嵌入模型。该模型会将文本块转换为固定维度的向量。
*   `Chroma.from_documents()`: 这是一个便捷方法，它会：
    1.  遍历 `document_chunks` 中的每个文档块。
    2.  使用提供的 `embeddings` 模型为每个块的 `page_content` 生成嵌入向量。
    3.  将这些嵌入向量及其对应的文档块内容和元数据存储到 ChromaDB 向量数据库中（默认情况下，ChromaDB 会在内存中运行或在本地磁盘上创建一个持久化存储）。
*   此时，`vector_store` 对象就是一个准备好进行相似性搜索的检索器。

#### 步骤 4: 检索文档中的答案 (Retrieve Answers from Documents)

*   **目标**: 创建一个简单的应用程序，接收用户问题，并搜索与该问题相关的文档。
*   **LangChain 工具**: 使用 `load_qa_chain` 函数。
*   **`load_qa_chain`**: 用于加载一个链 (chain)，该链能够在一组文档上实现问答功能。
*   **基础问答模型 (QA Model)**: 教程中使用 `meta-llama/Llama-2-7b-hf` 模型作为基础的LLM来进行问答。
    *   **LLaMA (Large Language Model Meta AI)**: 由 Meta 开发的先进的基础语言模型。
    *   有不同大小的版本 (7B 到 65B 参数)。
    *   在多样化的互联网文本上训练，能够根据输入生成连贯且相关的文本。
    *   设计上更高效且易于访问，旨在为NLP任务提供强大工具，同时针对资源较低的硬件进行了性能优化。

**关键代码：设置Hugging Face Hub API Token并加载QA模型** 你需要一个 Hugging Face API Token 来访问某些模型，特别是像 Llama-2 这样可能需要许可或位于非公开仓库的模型。你需要去Hugging Face网站注册并获取token。

Python

    import os
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR_TOKEN" # 替换 YOUR_TOKEN 为你自己的Hugging Face API Token
    
    from langchain import HuggingFaceHub # 旧版本路径
    # from langchain_community.llms import HuggingFaceHub # 推荐的新版本路径
    
    qa_model = HuggingFaceHub(
        repo_id="meta-llama/Llama-2-7b-hf", # 指定要使用的Llama-2模型的仓库ID
        model_kwargs={
            "temperature": 0.7,     # 控制生成文本的随机性，较低的值使输出更确定
            "max_new_tokens": 1024, # 控制生成的最大新词元数量
        }
    )

**详细说明：**

*   `os.environ["HUGGINGFACEHUB_API_TOKEN"]`: 将你的Hugging Face API token设置为环境变量，以便LangChain可以代表你访问Hugging Face Hub上的模型。
*   `HuggingFaceHub`: LangChain中用于与Hugging Face Hub上托管的模型进行交互的类。它允许你运行推理任务，而无需在本地下载整个模型。
*   `repo_id`: 指定Hugging Face Model Hub上模型的唯一标识符。
*   `model_kwargs`: 传递给模型推理端点的参数。
    *   `temperature`: 控制输出的随机性。值越低，输出越确定和集中；值越高，输出越随机和多样。
    *   `max_new_tokens`: 生成响应时允许的最大新词元数。

**加载问答链 (QA Chain)：** `load_qa_chain` 提供了一个通用的接口来回答问题。它会加载一个链，该链可以使用你提供的文档和LLM来进行问答。`chain_type="stuff"` 是最简单的链类型，它简单地将所有检索到的文档内容“塞进 (stuff)” LLM的提示中。

Python

    from langchain.chains.Youtubeing import load_qa_chain
    
    qa_chain = load_qa_chain(qa_model, chain_type="stuff")

**详细说明：**

*   `load_qa_chain(llm, chain_type)`:
    *   `llm`: 要用于生成答案的语言模型实例 (这里是`qa_model`)。
    *   `chain_type="stuff"`: 这是最直接的方法。它将所有检索到的文档合并成一个大的上下文，然后连同问题一起传递给LLM。如果文档总长度超过LLM的上下文窗口限制，这种方法可能会失败。其他链类型如 "map\_reduce", "refine", "map\_rerank" 提供了处理更长文档的方法。

**提出问题并检索相关文档：** 首先定义一个问题，然后使用之前创建的`vector_store`进行相似性搜索，找到与问题最相关的文档块。

Python

    question = "What is the name of doctor & patient ?" # 定义用户问题
    
    search_results = vector_store.similarity_search(question) # 在向量存储中进行相似性搜索
    # similarity_search 默认会返回与查询最相似的k个文档块 (k通常默认为4)
    
    # 打印第一个最相关的文档块的内容
    search_content = search_results[0].page_content
    print(search_content)

**输出解释 (search\_content)：** 会打印出与问题 "What is the name of doctor & patient ?" 最相似的文档块的内容。根据教程示例文档，这应该是包含 "Amit, a health-conscious man from suburban India, regularly visited his local doctor, Dr. Kapoor..." 的那部分文本。 **注意**: 输出中有一个警告 `WARNING:chromadb.segment.impl.vector.local_hnsw:Number of requested results 4 is greater than number of elements in index 3, updating n_results = 3`。这是因为我们的示例文档只被分割成了3个块，而`similarity_search`默认可能想返回4个结果，所以ChromaDB将其调整为返回实际存在的3个块。

**生成答案：** `qa_chain` 会使用一个预设的提示模板来构建最终给LLM的提示。这个模板通常指示LLM“使用以下上下文来回答问题。如果不知道答案，就说不知道，不要编造答案。” 然后，它将检索到的文档 (`search_results`) 和问题 (`question`) 输入到链中，LLM会基于这些信息生成答案。

Python

    # 使用 .run() 方法 (旧版) 或 .invoke() 方法 (新版)
    # answers = qa_chain.run(input_documents=search_results, question=question) # 旧版用法
    answers = qa_chain.invoke({"input_documents": search_results, "question": question}) # 新版推荐用法
    
    print(answers) # 打印LLM生成的答案
    # 如果使用 .run(), answers 是一个字符串。
    # 如果使用 .invoke(), answers 是一个包含 'output_text' 键的字典。
    # 为了保持与教程输出一致，假设教程使用的是 .run() 或直接打印字典中的文本部分。
    # 教程的打印输出包含了完整的提示和最终答案。这表明 qa_chain.run() 在内部构建了提示，
    # 并将其发送给 HuggingFaceHub 进行推理。
    # 实际的 answers 变量通常只包含LLM的最终回答部分。
    # 为了重现教程的打印效果，我们假设它可能直接打印了某种中间调试信息或者
    # qa_chain.run 返回的包含完整交互的字符串（这在某些配置下可能）。
    # 更常见的是，你需要从返回的字典中提取 'output_text' 或 'result'。
    
    # 为了更清晰地展示最终答案，可以这样做：
    if isinstance(answers, dict) and "output_text" in answers:
        print("Helpful Answer:", answers["output_text"])
    elif isinstance(answers, dict) and "result" in answers: # 有些链可能返回 'result'
        print("Helpful Answer:", answers["result"])
    else:
        print("Raw output or old run() method output:", answers)
    
    
    # 教程的输出格式似乎是直接将最终的提示和模型的回答拼接在一起打印出来：
    # "Use the following pieces of context to answer the question at the end...
    #  [Context from search_results]
    #  Question: What is the name of doctor & patient ?
    #  Helpful Answer: The doctor is Dr. Kapoor, and the patient is Amit."
    # 这意味着 qa_chain.run 可能在某些verbose模式下或者其实现是这样返回的。
    # 或者，教程中 print(answers) 可能实际上是 print(qa_chain.invoke(...)) 后对结果的特定处理。
    # 如果直接使用 qa_chain.run(input_documents=search_results, question=question)，
    # 然后 print(answers)，通常 answers 只会是 "The doctor is Dr. Kapoor, and the patient is Amit."

**输出解释 (answers)：** 教程提供的输出是：

Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    [Context from search_results - 第一个块]
    [Context from search_results - 第二个块]
    [Context from search_results - 第三个块]
    
    Question: What is the name of doctor & patient ?
    Helpful Answer: The doctor is Dr. Kapoor, and the patient is Amit.

这个输出格式表明，`qa_chain.run`（或者教程中某种打印方式）展示了传递给LLM的完整提示（包括指令、检索到的上下文和问题），以及LLM生成的“Helpful Answer”。 LLM根据提供的上下文，正确地识别出医生是Dr. Kapoor，病人是Amit。

#### 总结

本教程成功地演示了如何使用LangChain构建一个简单的RAG问答应用。 **关键流程回顾：**

1.  **安装和导入库**: 安装LangChain及其相关依赖。
2.  **加载数据 (Load)**: 使用`TextLoader`从外部文件加载原始文本数据。
3.  **分割数据 (Split)**: 使用`RecursiveCharacterTextSplitter`将长文本分割成较小的、可管理的块，并带有重叠。
4.  **嵌入和存储 (Embed & Store)**:
    *   使用`HuggingFaceEmbeddings`为每个文本块生成向量嵌入。
    *   使用`Chroma.from_documents`将这些嵌入和文档块存储到ChromaDB向量数据库中，创建一个`vector_store`。
5.  **设置LLM**: 使用`HuggingFaceHub`配置并加载一个大型语言模型（如Llama-2）作为问答的基础模型。
6.  **创建QA链 (QA Chain)**: 使用`load_qa_chain`和`chain_type="stuff"`创建一个问答链。
7.  **检索 (Retrieve)**: 当用户提问时，首先使用`vector_store.similarity_search(question)`从向量数据库中检索与问题最相关的文档块。
8.  **生成 (Generate)**: 将检索到的文档块和用户问题通过`qa_chain`传递给LLM，生成最终答案。

这个流程清晰地展示了RAG如何通过结合外部知识库（通过检索获得）和LLM的推理能力来回答特定领域的问题，即使这些知识不在LLM的原始训练数据中。

信息来源和相关内容

---

![IPYNB 图标](https://drive-thirdparty.googleusercontent.com/32/type/text/code)

tutorial\_w...LoRA.ipynb

IPYNB

同样地，讲解该文件

---

好的，我们来详细讲解这个关于\*\*“参数高效微调 (Parameter Efficient Finetuning - PEFT) 与 LoRA”\*\*的Jupyter Notebook教程。

### 教程 10 - 参数高效微调 (PEFT) 与 LoRA

#### 1\. 引言：训练和微调大型语言模型 (LLM) 的挑战

教程首先指出了训练和微调大型语言模型 (LLM) 面临的主要问题，核心在于其巨大的**权重规模**。

*   **计算资源需求大 (A lot more computation resources to train)：**
    *   训练LLM（如GPT-3、BLOOM等）需要大量的GPU和TPU集群，以及长时间的训练。
    *   即使是微调，如果采用全参数微调 (full finetuning)，也需要相当可观的计算资源，因为需要更新模型的所有参数。
*   **模型文件体积庞大 (The file sizes becomes huge)：**
    *   由于参数数量巨大，LLM的权重文件非常大。教程中提到，T5-XXL模型的大小可以达到40GB。
    *   这给模型的存储、分发和加载带来了挑战。
*   **单GPU环境下的限制 (Considering a computation environment that with only one GPU)：**
    *   在只有单个GPU的环境中，常见的并行化训练方法（如数据并行 Data Parallelism, 模型并行 Model Parallelism, 流水线并行 Pipeline Parallelism）都无法有效利用。
    *   这使得在资源受限的情况下训练或微调大型模型变得非常困难。

**解决方案：参数高效微调 (PEFT - Parameter Efficient Fine-Tuning)** PEFT 是一系列旨在解决上述问题的技术，它们的目标是在保持或接近全参数微调性能的同时，显著减少需要训练的参数数量。

#### 2\. PEFT 技术概览

PEFT 包含了多种不同的技术，本教程主要关注 **LoRA (Low-Rank Adaptation)**。 除了LoRA，教程还提到了其他一些PEFT技术，例如：

*   **Prefix Tuning (前缀调整)**: 在输入序列的embedding层前添加可训练的前缀向量。
*   **P-Tuning (参数化提示调整)**: 学习连续的、可训练的提示嵌入，而不是离散的文本提示。
*   **Prompt Tuning (提示调整)**: 与P-Tuning类似，也是学习添加到输入嵌入中的软提示 (soft prompts)。
*   以及其他未列出的技术。

**PEFT的核心优势：**

*   **减少可训练参数：** 大大减少了微调过程中需要更新的参数数量（通常只占总参数的很小一部分，如0.01%到1%）。
*   **降低计算成本：** 由于可训练参数减少，训练所需的计算资源（GPU显存、训练时间）也随之降低。
*   **减少存储需求：** 微调后只需要存储少量新增的或修改的参数，而不是整个模型的副本。原始的LLM权重保持不变。
*   **更易于部署和切换任务：** 可以为每个下游任务训练一个小的PEFT模块，并在推理时根据需要加载，而无需为每个任务存储和加载整个大模型。
*   **可移植性好：** PEFT模块通常很小，方便共享和部署。
*   **避免灾难性遗忘 (Catastrophic Forgetting)：** 由于大部分预训练模型的参数被冻结，PEFT有助于保留模型在预训练阶段学习到的通用知识，减少在特定任务上微调时对其他知识的遗忘。
*   **性能接近全参数微调：** 在许多任务上，PEFT方法可以达到与全参数微调相当甚至更好的性能。

#### 3\. LoRA (Low-Rank Adaptation) 详解

LoRA是PEFT中最流行和广泛使用的方法之一。

*   **核心思想：**
    
    *   大型语言模型在预训练后，其权重矩阵通常具有低的“内在维度 (intrinsic dimension)”。这意味着尽管权重矩阵很大，但其包含的信息可以通过低秩矩阵来近似。
    *   LoRA 不直接修改预训练模型的原始权重 W0​，而是通过学习一个低秩的“残差”或“更新”矩阵 ΔW 来适应新任务。
    *   这个更新矩阵 ΔW 被分解为两个更小的、低秩的矩阵 A 和 B 的乘积，即 ΔW\=BA。
        *   如果原始权重矩阵 W0​∈Rd×k，则 B∈Rd×r，A∈Rr×k。
        *   其中 r 是秩 (rank)，是一个远小于 d 和 k 的超参数 (r≪min(d,k))。这个 r 控制了LoRA适配器的“容量”或表达能力。
*   **前向传播过程中的修改：**
    
    *   对于原始模型中的一个线性层（或其他可应用LoRA的层，如注意力层中的权重矩阵），其原始输出为 h\=W0​x。
    *   应用LoRA后，输出变为 h\=W0​x+ΔWx\=W0​x+BAx。
    *   在实现时，通常会额外加一个缩放因子 α/r，所以 h\=W0​x+rα​BAx。α 是一个常数超参数，用于调整LoRA更新的幅度，通常设为与 r 相同，或者作为可调参数。
*   **训练过程：**
    
    *   原始模型的权重 W0​ 被冻结，不参与梯度更新。
    *   只有低秩矩阵 A 和 B 的参数是可训练的。
    *   由于 r 很小，所以 A 和 B 的参数数量远少于 W0​ 的参数数量，从而实现了参数高效的微调。
*   **推理过程：**
    
    *   可以将学习到的 ΔW\=BA 与原始权重 W0​ 合并，得到新的权重 W′\=W0​+BA。这样，在推理时就不需要额外的计算开销，模型的结构与原始模型相同。
    *   或者，保持 A 和 B 分离，在推理时计算 W0​x 和 BAx 然后相加。

**教程中的图片：** 教程中包含一张图片，形象地展示了LoRA的结构。图中显示了原始的预训练权重 W0​ 路径和并行的LoRA路径（包含矩阵A和矩阵B）。输入 x 同时经过这两条路径，然后将结果相加。

#### 4\. 使用 Hugging Face `PEFT` 库实现 LoRA

Hugging Face 的 `PEFT` (Parameter-Efficient Fine-Tuning) 库为多种PEFT方法（包括LoRA）提供了易于使用的实现。 教程接下来的代码部分将演示如何使用这个库来对一个大型语言模型（例如 `bigscience/bloom-7b1`）进行LoRA微调。

**关键代码结构（预期）：**

1.  **安装必要的库：** `transformers`, `datasets`, `peft`, `trl` (Transformer Reinforcement Learning, 常用于SFT - Supervised Fine-tuning), `bitsandbytes` (用于模型量化，如8位或4位训练，进一步降低显存需求)。
    
    Python
    
        !pip install transformers datasets peft trl bitsandbytes accelerate
    
2.  **加载预训练模型和分词器：** 使用 `transformers.AutoModelForCausalLM` 和 `transformers.AutoTokenizer` 加载一个预训练的因果语言模型及其分词器。
    
    Python
    
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch
        
        model_name = "bigscience/bloom-7b1" # 示例模型
        
        # （可选）配置量化以减少显存占用
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, # 或 load_in_8bit=True
            bnb_4bit_use_double_quant=True, # 嵌套量化
            bnb_4bit_quant_type="nf4", # NF4量化类型
            bnb_4bit_compute_dtype=torch.bfloat16 # 计算时使用的数据类型
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config, # 应用量化配置
            device_map="auto" # 自动将模型分片到可用设备 (如多GPU)
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 需要设置tokenizer的pad_token，如果它没有的话
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
3.  **准备数据集：** 加载一个用于微调的数据集，并进行必要的预处理（如分词，格式化成模型期望的输入格式）。例如，使用 `datasets.load_dataset` 加载一个文本数据集。
    
    Python
    
        from datasets import load_dataset
        # 示例：dataset = load_dataset("imdb", split="train")
        # 然后对数据集进行分词等预处理
    
4.  **配置 LoRA (`LoraConfig`)：** 使用 `peft.LoraConfig` 来定义LoRA的参数。
    
    Python
    
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        
        # （如果使用了k-bit量化训练，如4-bit）准备模型
        model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=8,  # LoRA的秩 (rank)，典型值可以是 8, 16, 32, 64
            lora_alpha=32, # LoRA的alpha缩放参数，通常设为r的两倍或与r相同
            target_modules=["query_key_value"], # 指定要应用LoRA的模块名称列表 (例如Transformer中的QKV线性层)
                                                # 对于BLOOM模型，可能是 "query_key_value" 或 "dense_h_to_4h", "dense_4h_to_h"等
                                                # 可以通过打印model结构来确定具体名称
            lora_dropout=0.05, # LoRA层的dropout率
            bias="none",       # 是否调整偏置项 ("none", "all", "lora_only")
            task_type="CAUSAL_LM" # 任务类型，例如 "CAUSAL_LM", "SEQ_2_SEQ_LM"
        )
    *   `r`: LoRA矩阵的秩。较小的 `r` 意味着更少的参数和更快的训练，但可能牺牲一些性能。
    *   `lora_alpha`: LoRA的缩放因子。
    *   `target_modules`: 一个字符串列表，指定模型中哪些模块（通常是线性层或注意力机制中的特定线性层）将被LoRA适配器替换。这个参数非常重要，需要根据具体模型架构来确定。例如，对于Transformer模型，通常会选择应用于注意力机制中的查询（Query）、键（Key）、值（Value）投影层，以及前馈网络中的线性层。
    *   `lora_dropout`: 在LoRA层中使用的dropout比率。
    *   `bias`: 控制如何处理偏置项。`"none"` 表示不训练偏置项，`"all"` 表示训练所有偏置项，`"lora_only"` 表示只训练LoRA层引入的偏置项（如果适用）。
    *   `task_type`: 指定PEFT方法应用的任务类型，这有助于库选择合适的配置。
5.  **应用LoRA到模型：** 使用 `peft.get_peft_model` 函数将LoRA配置应用到预训练模型上。
    
    Python
    
        peft_model = get_peft_model(model, lora_config)
        # 打印可训练参数的数量和比例
        peft_model.print_trainable_parameters()
    
    `print_trainable_parameters()` 会显示LoRA模型中可训练参数的数量及其占总参数量的百分比，这个百分比通常会非常小。
    
6.  **定义训练参数和训练器 (Trainer)：** 使用 `transformers.TrainingArguments` 和 `trl.SFTTrainer` (Supervised Fine-tuning Trainer，专门为指令微调或类似任务设计，但也可用于常规的因果语言模型微调)。
    
    Python
    
        from transformers import TrainingArguments
        from trl import SFTTrainer
        
        # training_args = TrainingArguments(...) # 定义训练参数，如输出目录、学习率、批大小、epoch数等
        
        # 假设已经有处理好的 dataset_train 和 dataset_eval
        # SFTTrainer需要一个格式化函数 (formatting_func) 或者直接处理特定格式的数据集
        # 或者使用 transformers.Trainer
        
        # 如果使用 SFTTrainer，它通常期望数据集中的文本字段包含完整的输入和输出（例如，对于指令微调，是 "instruction <input> output"）
        # 对于纯语言模型微调，可以简单地将文本作为目标。
        # 确保 tokenizer.pad_token 被设置
        
        # 示例训练参数 (需要根据实际情况调整)
        output_dir = "./bloom-7b1-lora-finetuned"
        per_device_train_batch_size = 1 # 对于大模型和单GPU，批大小通常很小
        gradient_accumulation_steps = 4 # 梯度累积
        optim = "paged_adamw_32bit" # 适用于量化训练的优化器
        save_steps = 100
        logging_steps = 10
        learning_rate = 2e-4
        max_grad_norm = 0.3
        max_steps = 1000 # 或者 num_train_epochs
        warmup_ratio = 0.03
        lr_scheduler_type = "constant" # 或者 "linear", "cosine"
        
        training_arguments = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            save_steps=save_steps,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            fp16=True, # 如果GPU支持，使用混合精度训练
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            group_by_length=True, # 将相似长度的序列分到一批，减少填充
            lr_scheduler_type=lr_scheduler_type,
            report_to="none" # 或者 "tensorboard", "wandb"
        )
        
        # 假设 `formatted_dataset` 是已经分词并准备好的数据集
        # `dataset_text_field` 是数据集中包含文本的字段名
        trainer = SFTTrainer(
            model=peft_model,
            train_dataset=formatted_dataset["train"], # 替换为实际的训练数据集
            # eval_dataset=formatted_dataset["validation"], # 替换为实际的验证数据集
            peft_config=lora_config, # 传递LoRA配置
            # dataset_text_field="text", # 指定数据集中包含文本的字段名
            max_seq_length=1024,       # 序列最大长度
            tokenizer=tokenizer,
            args=training_arguments,
            # packing=True, # (可选) 将多个短序列打包成一个长序列以提高效率
        )
    *   `SFTTrainer` 相对于标准的 `Trainer` 做了一些针对序列生成任务的优化和简化。
    *   `peft_config` 参数直接传入LoRA配置。
    *   `dataset_text_field`: 指定数据集中哪个字段包含要训练的文本。
    *   `max_seq_length`: 训练时使用的最大序列长度。
7.  **开始训练：**
    
    Python
    
        trainer.train()
    
8.  **保存模型：** 训练完成后，可以保存PEFT适配器（即LoRA的权重A和B）。
    
    Python
    
        trainer.save_model("./bloom-7b1-lora-adapter") # 只保存适配器的权重
        # tokenizer也应该被保存
        tokenizer.save_pretrained("./bloom-7b1-lora-adapter")
    
    保存的适配器非常小，可以方便地与原始的预训练模型结合使用。
    
9.  **加载和推理：** 加载原始的预训练模型，然后加载PEFT适配器权重进行推理。
    
    Python
    
        from peft import PeftModel
        
        # 加载原始的基础模型 (可以量化加载以节省内存)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config, # 如果训练时用了量化
            device_map="auto"
        )
        # 加载PEFT适配器
        peft_loaded_model = PeftModel.from_pretrained(base_model, "./bloom-7b1-lora-adapter")
        peft_loaded_model.eval() # 设置为评估模式
        
        # 使用加载的PEFT模型进行推理
        # inputs = tokenizer("Some prompt text", return_tensors="pt").to("cuda")
        # outputs = peft_loaded_model.generate(**inputs, max_new_tokens=50)
        # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    

**教程代码块中的 `import os ... tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")`：** 这个代码块是实际操作的开始，导入了必要的库，并加载了 `bigscience/bloom-7b1` 模型和对应的分词器。`os.environ["CUDA_VISIBLE_DEVICES"]="0"` 用于指定使用第一个GPU（如果存在多个GPU）。`bitsandbytes` 通常用于8位或4位量化训练，以减少显存占用。

**关键代码摘要 (来自教程文件末尾的片段，用于设置模型和分词器)：**

Python

    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="0" # 指定使用GPU 0
    import torch
    import torch.nn as nn
    import bitsandbytes as bnb # 用于量化
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    # from datasets import load_dataset # 教程中提到但可能在后续单元格使用
    # from trl import SFTTrainer # 教程中提到但可能在后续单元格使用
    # from peft import LoraConfig # 教程中提到但可能在后续单元格使用
    
    
    # 加载模型 (这里没有直接使用量化，但bitsandbytes已导入，暗示可能用于更高级的配置)
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-7b1") # 加载 BLOOM 7.1B 模型
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")
    
    # "# your code here" 暗示了这是让学习者基于前面的理论知识，
    # 继续完成LoRA配置、模型包装、数据准备和训练器设置的代码。

**总结：** 这个教程首先解释了微调大型模型的挑战，然后介绍了PEFT作为一种高效的解决方案，并详细阐述了LoRA的工作原理。最后，通过预期代码流程，展示了如何利用Hugging Face的`transformers`、`datasets`、`peft`和`trl`库来实现对大型语言模型的LoRA微调。核心在于只训练少量的适配器参数（LoRA层中的A和B矩阵），而不是整个模型的参数，从而大大降低了计算和存储成本。

信息来源和相关内容

[

discuss.huggingface.co

](https://discuss.huggingface.co/t/single-node-multi-gpu-flant5-fine-tuning-using-hf-dataset-and-hf-trainer/38664)

[

discuss.huggingface.co

](https://discuss.huggingface.co/t/single-node-multi-gpu-flant5-fine-tuning-using-hf-dataset-and-hf-trainer/38664)

[

Tutorial\_9... (1).ipynb

![IPYNB 图标](https://drive-thirdparty.googleusercontent.com/32/type/text/code)

IPYNB







](https://contribution.usercontent.google.com/download?c=CgxiYXJkX3N0b3JhZ2USTxIMcmVxdWVzdF9kYXRhGj8KMDAwMDYzNGRhZjVmM2VkY2UwN2Y1NWVlYTAxMTFmZTdhOWI1OTYxOWZjMjhkOWY3NhILEgcQ57PmiuAQGAE&filename&opi=103135050)

---

![IPYNB 图标](https://drive-thirdparty.googleusercontent.com/32/type/text/code)

Tutorial\_w...hain.ipynb

IPYNB

同样地，讲解该文件

---

分析

分析

*   读取文档
    
    已完成回答
    

好的，我们来详细讲解这个关于 **LangChain** 的Jupyter Notebook教程。

### CS6493 教程 11: LangChain

本教程主要分为以下几个部分：

1.  **LangChain 简介 (Introduction to LangChain)**
2.  **使用 LangChain + Ollama 创建聊天机器人 (Create a ChatBot with LangChain + Ollama)**
3.  **使用 LangChain + Ollama 创建检索增强型聊天机器人 (Create a Retrieval Augmented ChatBot with LangChain + Ollama)**
4.  **练习：对话检索链 (Practice: Conversation Retrieval Chain)**

#### LangChain 简介 (Introduction to LangChain)

*   **定义**: LangChain 是一个用于开发由语言模型驱动的应用程序的框架。
    
*   **核心能力**: 它使得应用程序能够：
    
    *   **感知上下文 (Context-aware)**: 将语言模型连接到各种上下文来源（如提示指令、少样本示例、用于生成响应的背景内容等）。
    *   **进行推理 (Reason)**: 依赖语言模型进行推理（例如，如何根据提供的上下文回答问题、采取什么行动等）。
*   **框架组成**:
    
    *   **LangChain 库 (LangChain Libraries)**: Python 和 JavaScript 库。包含各种组件的接口和集成、用于将这些组件组合成链 (chains) 和代理 (agents) 的基本运行时，以及现成的链和代理实现。
    *   **LangChain 模板 (LangChain Templates)**: 一系列易于部署的参考架构，适用于各种任务。
    *   **LangServe**: 一个用于将 LangChain 链部署为 REST API 的库。
    *   **LangSmith**: 一个开发者平台，允许调试、测试、评估和监控基于任何LLM框架构建的链，并与 LangChain 无缝集成。
*   **简化的应用生命周期**:
    
    *   **开发 (Develop)**: 使用 LangChain/LangChain.js 编写应用程序，并可利用模板快速上手。
    *   **生产化 (Productionize)**: 使用 LangSmith 检查、测试和监控链，以便持续改进并自信地部署。
    *   **部署 (Deploy)**: 使用 LangServe 将任何链转换为 API。
*   **LangChain 库的核心价值**:
    
    *   **组件 (Components)**: 用于处理语言模型的可组合工具和集成。这些组件是模块化的，易于使用，无论是否使用 LangChain 框架的其他部分。
    *   **现成的链 (Off-the-shelf chains)**: 内置的组件组合，用于完成更高级别的任务。
    *   现成的链便于快速开始，而组件则便于自定义现有链和构建新链。
*   **LangChain 库的包结构**:
    
    *   **`langchain-core`**: 包含基础抽象和 LangChain 表达式语言 (LCEL)。
    *   **`langchain-community`**: 包含第三方集成。
    *   **`langchain`**: 包含构成应用程序认知架构的链、代理和检索策略。

#### 使用 LangChain + Ollama 创建聊天机器人 (Create a ChatBot with LangChain + Ollama)

这部分的目标是逐步构建一个聊天机器人应用。

1.  **简单 LLM 链**: 仅依赖提示模板中的信息进行响应。
2.  **检索链**: 从单独的数据库中获取数据，并将其传递到提示模板中。
3.  **对话检索链**: 在检索链的基础上增加聊天历史记录，使其能够进行多轮对话并记住先前的问题。

##### 安装与设置 (Install & Setup)

首先，安装必要的 LangChain 包。 **关键代码：安装LangChain**

Python

    !pip install langchain
    !pip install langchain-community langchain-core # 安装核心和社区包

**详细说明：**

*   `langchain`: 核心 LangChain 包。
*   `langchain-community`: 包含许多第三方集成，例如与Ollama、各种向量数据库等的连接。
*   `langchain-core`: 提供LangChain的基础抽象和表达式语言 (LCEL)。

教程选择使用 **Ollama** 来运行本地的开源LLM。Ollama 允许用户在本地运行如 Llama 2, Gemma, Mistral 等模型。

**关键代码：安装和启动Ollama服务**

Python

    !curl -fsSL https://ollama.com/install.sh | sh # 下载并执行Ollama的安装脚本
    !setsid nohup ollama serve > run_serve.txt 2>&1 & # 在后台启动Ollama服务，并将日志输出到run_serve.txt

**详细说明：**

*   第一行命令从 Ollama 官网下载安装脚本并执行，这会在系统中安装 Ollama。
*   第二行命令使用 `setsid` 和 `nohup` 在后台启动 `ollama serve` 进程。`setsid` 会在一个新的会话中运行命令，使其与当前终端分离。`nohup` (no hang up) 确保即使用户登出，进程也会继续运行。输出和错误信息被重定向到 `run_serve.txt` 文件。

**关键代码：检查可用模型和运行指定模型**

Python

    !ollama list # 列出本地已下载的Ollama模型
    !setsid ollama run gemma:2b > run_gemma_2b.txt 2>&1 & # 在后台运行（并下载，如果尚未下载）gemma:2b模型

**详细说明：**

*   `!ollama list`: 显示当前Ollama环境中已经拉取（下载）的模型列表。初次运行时可能为空。
*   `!setsid ollama run gemma:2b ... &`: 这个命令会启动（如果模型不存在则先下载）名为 `gemma:2b` 的模型。Gemma 是 Google 开发的一系列轻量级、高性能的开放模型。`2b` 指的是20亿参数版本。同样，输出被重定向。

**关键代码：初始化Ollama LLM实例** 使用 `langchain_community.llms.Ollama` 来创建一个与本地运行的Ollama模型交互的LLM实例。

Python

    from langchain_community.llms import Ollama
    llm = Ollama(model="gemma:2b") # 指定使用gemma:2b模型

**详细说明：**

*   这行代码创建了一个`llm`对象，它是LangChain中与Ollama `gemma:2b`模型通信的接口。
*   教程中可能会出现一个 `LangChainDeprecationWarning`，提示 `Ollama` 类已弃用，并建议使用 `langchain-ollama` 包中的新类。这是因为LangChain正在将其生态系统模块化。

**关键代码：直接调用LLM** 可以直接使用`invoke`方法向LLM提问。

Python

    llm.invoke("how can langsmith help with testing?")

**详细说明：**

*   `invoke()` 是LangChain表达式语言(LCEL)中用于执行链或可运行对象的方法。
*   这里直接调用LLM，模型会根据其预训练知识回答问题。由于Langsmith是较新的工具，`gemma:2b` 的预训练数据可能不包含相关信息，所以回答可能不准确或泛泛而谈。
*   教程的输出显示模型给出了一些通用的测试辅助方法，这些方法可能适用于任何测试工具或概念，而不一定是Langsmith特有的。

##### 使用提示模板 (Prompt Template)

为了更好地引导LLM的响应，可以使用提示模板。 **关键代码：创建聊天提示模板**

Python

    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are world class technical documentation writer."), # 系统消息，设定AI的角色和风格
        ("user", "{input}") # 用户消息，包含一个占位符 {input} 用于实际的用户输入
    ])

**详细说明：**

*   `ChatPromptTemplate.from_messages`: 从一个消息列表创建提示模板。
*   `("system", "...")`: 系统消息，用于设定AI的身份、行为准则或整体对话的上下文。
*   `("user", "{input}")`: 用户消息。`{input}` 是一个变量，在实际调用时会被替换为用户的具体问题。

**关键代码：创建简单的LLM链** 使用LangChain表达式语言 (LCEL) 的管道符 `|` 将提示模板和LLM连接起来，形成一个链。

Python

    chain = prompt | llm

**详细说明：**

*   这个链的工作流程是：用户的输入首先通过`prompt`模板进行格式化，然后将格式化后的提示传递给`llm`进行处理。

**关键代码：调用链**

Python

    chain.invoke({"input": "how can langsmith help with testing?"})

**详细说明：**

*   调用这个链时，传入一个包含`input`键的字典，其值会被用来替换提示模板中的`{input}`占位符。
*   LLM的输出仍然是其原始格式（对于Ollama和许多聊天模型，通常是一个字符串，但对于某些封装的ChatModel，可能是一个`AIMessage`对象）。
*   模型现在会尝试以“世界级的技术文档编写者”的口吻来回答问题。

##### 输出解析器 (Output Parser)

LLM的输出通常是消息对象（如`AIMessage`）或字符串。为了方便后续处理，可以使用输出解析器将其转换为更易用的格式，如直接转换为字符串。 **关键代码：使用字符串输出解析器**

Python

    from langchain_core.output_parsers import StrOutputParser
    output_parser = StrOutputParser()

**关键代码：将输出解析器添加到链中**

Python

    chain = prompt | llm | output_parser

**详细说明：**

*   现在，链的输出会经过`StrOutputParser`处理，确保最终结果是一个字符串。

**关键代码：再次调用链**

Python

    chain.invoke({"input": "how can langsmith help with testing?"})

**详细说明：**

*   这次调用后，输出结果会是一个格式化后的字符串，内容上与之前类似，但类型是字符串。

#### 创建检索增强型聊天机器人 (Create a Retrieval Augmented ChatBot with LangChain + Ollama)

为了让LLM能够回答关于其训练数据之外的特定知识（例如关于LangSmith的详细信息），需要引入检索机制。

##### 1\. 加载外部数据 (Load External Data)

使用`WebBaseLoader`从网页加载数据作为外部知识源。 **关键代码：安装BeautifulSoup并加载网页内容**

Python

    !pip install beautifulsoup4 # 用于解析HTML
    from langchain_community.document_loaders import WebBaseLoader
    loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide") # 加载LangSmith用户指南页面
    docs = loader.load() # 返回一个Document对象列表

**详细说明：**

*   `WebBaseLoader` 会抓取指定URL的网页内容，并将其转换为LangChain的`Document`格式。

##### 2\. 创建嵌入模型 (Embedding Model)

使用Ollama提供的嵌入功能来为文本数据创建向量表示。 **关键代码：初始化Ollama嵌入模型**

Python

    from langchain_community.embeddings import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="gemma:2b") # 使用gemma:2b模型生成嵌入

**详细说明：**

*   `OllamaEmbeddings` 类允许使用本地Ollama模型来计算文本的嵌入向量。

##### 3\. 创建向量存储 (Vector Store)

将加载并分割后的文档嵌入并存储到向量数据库中，以便进行相似性搜索。教程中使用FAISS作为向量存储。 **关键代码：安装FAISS并构建索引**

Python

    !pip install faiss-cpu # FAISS的CPU版本 (也有GPU版本 faiss-gpu)
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    text_splitter = RecursiveCharacterTextSplitter() # 使用默认参数的递归字符分割器
    documents = text_splitter.split_documents(docs) # 将加载的文档分割成小块
    vector = FAISS.from_documents(documents, embeddings) # 从文档块和嵌入模型创建FAISS索引

**详细说明：**

*   `RecursiveCharacterTextSplitter`: 将长文档分割成语义上相关的块。
*   `FAISS.from_documents`: 这个方法会为每个文档块生成嵌入，并将这些嵌入存储在FAISS索引中，以便快速进行相似性搜索。

##### 4\. 创建文档处理链 (Document Chain)

定义一个链，用于接收问题和检索到的上下文文档，并生成答案。 **关键代码：定义提示模板和文档链**

Python

    from langchain.chains.combine_documents import create_stuff_documents_chain
    
    # 新的提示模板，明确指示LLM基于提供的上下文回答问题
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
    
    <context>
    {context}
    </context>
    
    Question: {input}""")
    
    document_chain = create_stuff_documents_chain(llm, prompt) # llm是Ollama gemma:2b实例

**详细说明：**

*   `create_stuff_documents_chain`: 这个函数创建一个链，它会将所有提供的上下文文档（`{context}`占位符）和问题（`{input}`占位符）“塞入”到提示模板中，然后传递给LLM。
*   提示模板中的`<context>{context}</context>`部分会由检索到的文档内容填充。

**关键代码：直接调用文档链进行测试 (可选)** 可以手动提供上下文来测试这个`document_chain`。

Python

    from langchain_core.documents import Document
    document_chain.invoke({
        "input": "how can langsmith help with testing?",
        "context": [Document(page_content="langsmith can let you visualize test results")] # 手动提供一个简单的上下文
    })

**详细说明：**

*   这里用一个非常简化的上下文来测试。LLM被限制只能基于这个上下文回答，所以如果上下文信息不足，它可能会说不知道或给出受限的答案。

##### 5\. 创建检索链 (Retrieval Chain)

将检索器（向量存储）与文档处理链结合起来，形成一个完整的检索链。 **关键代码：创建检索链**

Python

    from langchain.chains import create_retrieval_chain
    
    retriever = vector.as_retriever() # 从FAISS向量存储创建一个检索器对象
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

**详细说明：**

*   `vector.as_retriever()`: 将FAISS向量存储转换为一个标准的LangChain `Retriever`接口。
*   `create_retrieval_chain`: 这个函数将`retriever`和`document_chain`组合起来。当调用这个链时，它会：
    1.  使用用户的输入（问题）通过`retriever`检索相关文档。
    2.  将检索到的文档作为上下文，连同用户的输入一起传递给`document_chain`来生成答案。

**关键代码：调用检索链**

Python

    response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
    print(response["answer"]) # 答案在返回字典的 'answer' 键中

**详细说明：**

*   现在，当提问时，模型会首先从加载的LangSmith文档中检索相关信息，然后基于这些信息回答问题。因此，答案会更具体和准确。
*   输出会是LLM基于从LangSmith用户指南中检索到的上下文生成的关于LangSmith如何帮助测试的详细回答。

#### 练习：对话检索链 (Practice: Conversation Retrieval Chain)

目标是创建一个能够处理多轮对话的检索链，即聊天机器人需要记住之前的对话内容。

**关键点：**

1.  **检索方法更新**: 检索器不仅要考虑当前用户输入，还要考虑整个对话历史，以生成更相关的搜索查询。
2.  **最终LLM链更新**: LLM在生成答案时也需要考虑整个对话历史。

##### 1\. 更新检索器 (Updating Retrieval)

创建一个新的链，该链接收当前输入 (`input`) 和对话历史 (`chat_history`)，并使用LLM生成一个适合用于检索的搜索查询。

**关键代码：创建历史感知的检索器链**

Python

    from langchain.chains import create_history_aware_retriever
    from langchain_core.prompts import MessagesPlaceholder
    
    # 新的提示模板，用于根据对话历史和当前输入生成搜索查询
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"), # 占位符，用于插入聊天历史
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation") # 指示LLM生成搜索查询
    ])
    retriever_chain_with_history = create_history_aware_retriever(llm, retriever, prompt) # llm是Ollama, retriever是FAISS检索器

**详细说明：**

*   `MessagesPlaceholder(variable_name="chat_history")`: 允许在提示中动态插入一个消息列表（即聊天历史）。
*   `create_history_aware_retriever`: 这个函数创建一个链，该链首先使用LLM根据对话历史和当前用户输入重构一个查询，然后用这个重构后的查询去`retriever`中检索文档。

**关键代码：测试历史感知的检索器链**

Python

    from langchain_core.messages import HumanMessage, AIMessage
    
    chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
    retriever_chain_with_history.invoke({
        "chat_history": chat_history,
        "input": "Tell me how" # 后续问题
    })

**详细说明：**

*   这里模拟了一个简单的对话历史。
*   当用户问 "Tell me how" 时，这个历史感知的检索器会结合之前的对话（关于LangSmith能否帮助测试LLM应用）来理解后续问题的上下文，并生成一个更精确的搜索查询（例如，搜索 "how LangSmith helps test LLM applications"），然后返回相关的文档。
*   输出结果会是被检索到的`Document`对象列表。

##### 2\. 创建新的对话问答链

现在有了能够考虑对话历史的检索器，需要创建一个新的问答链，该链在生成答案时也会考虑对话历史。

**关键代码：创建包含聊天历史的问答链**

Python

    # 新的提示模板，包含上下文、聊天历史和当前用户输入
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"), # 系统消息，提供上下文
        MessagesPlaceholder(variable_name="chat_history"), # 插入聊天历史
        ("user", "{input}"), # 当前用户输入
    ])
    document_chain_with_history = create_stuff_documents_chain(llm, prompt)
    
    # 将历史感知的检索器链与新的文档处理链组合起来
    conversational_retrieval_chain = create_retrieval_chain(retriever_chain_with_history, document_chain_with_history)

**详细说明：**

*   新的提示模板现在同时包含 `{context}`（来自历史感知检索器的文档）和 `{chat_history}`。
*   `document_chain_with_history` 会将这些信息都传递给LLM。
*   `conversational_retrieval_chain` 的工作流程：
    1.  `retriever_chain_with_history` 部分：根据当前输入和聊天历史生成搜索查询，并检索相关文档。
    2.  `document_chain_with_history` 部分：将检索到的文档、聊天历史和当前用户输入组合成一个完整的提示，交由LLM生成最终答案。

**关键代码：测试完整的对话检索链**

Python

    chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
    response = conversational_retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": "Tell me how"
    })
    print(response["answer"]) # 打印最终答案

**详细说明：**

*   现在，LLM在回答 "Tell me how" 时，不仅会利用从LangSmith文档中检索到的具体信息，还会考虑到之前的对话内容（用户已经知道LangSmith可以提供帮助）。
*   输出的答案 (`response["answer"]`) 会是关于LangSmith如何帮助测试LLM应用的具体方法，并且其表述会考虑到这是一个对话的延续。
*   `response` 字典中除了 `answer`，通常还会包含 `context` (检索到的文档) 和 `chat_history` (更新后的聊天历史，如果链设计为自动更新的话，但这通常需要更复杂的管理)。

#### 总结

这个教程非常清晰地展示了LangChain的强大之处：

1.  **模块化**: 可以轻松组合不同的组件（LLM、提示模板、加载器、分割器、嵌入模型、向量存储、检索器、输出解析器）。
2.  **灵活性**: 支持本地模型（通过Ollama）和API模型。
3.  **核心功能**: 演示了如何从简单的LLM调用，到构建带提示的链，再到实现复杂的RAG（检索增强生成）和对话式RAG。
4.  **LCEL (LangChain Expression Language)**: 使用管道符 `|` 可以直观地将组件链接起来，构建复杂的数据处理和推理流程。

通过这个教程，学习者可以掌握使用LangChain和本地LLM（通过Ollama）构建具有上下文感知和推理能力的聊天机器人，特别是能够利用外部知识库并进行多轮对话的机器人。

信息来源和相关内容

---

