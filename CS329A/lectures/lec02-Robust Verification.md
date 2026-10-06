# CS329A Lecture 2: Robust Verification

[Training Verifiers to Solve Math Word Problems (arXiv 2021)](https://arxiv.org/abs/2110.14168)

大语言模型在数学推理中存在一个明显问题：它不仅可能产生错误答案，而且经常会以非常确定、非常连贯的方式给出错误的推理过程，也就是“看起来很合理，但实际上算错了”。当时已有的数据集也不完全适合研究这种数学推理能力。

- 一类数据集更偏向语言理解和普通问答，例如 SQuAD、OpenBookQA，并不重点考察多步数学推理；
- 另一类数据集，例如 MATH，又主要包含难度很高的竞赛级数学问题。
- 此外，一些已有数据并没有提供足够完整的中间推理步骤，因此很难研究模型究竟是在哪一步发生错误。

这就形成了工作的基本动机：不仅需要一个**适合多步数学推理的数据集**，也需要一种方法来**判断模型生成的完整解答到底是否可信**。

---

为了解决数据问题，工作提出了 **GSM8K** 数据集。GSM8K 包含大约 8,500 道小学水平的数学文字题。它的重要特点并不只是题目数量，而是数据质量和推理形式。

- 题目主要由人工创建，因此质量相对较高；
- 题目覆盖多种数学概念，具有较好的多样性；
- 更重要的是，它刻意强调 **multi-step reasoning（多步推理）**，要求模型通过若干中间步骤才能获得最终答案。
- 同时，GSM8K 收集的是自然语言形式的完整解题过程，而不仅仅是一个数学表达式或最终数值。这使研究者不仅能够观察“答案是否正确”，还能够研究模型生成的推理过程。

在此基础上，工作的第二个核心贡献是训练一个专门的 **verification model（验证模型）**。这个模型的任务不是自己生成答案，而是接收“题目 + 某个候选解答”，输出这个解答正确的概率。也就是说，Generator 负责“做题”，Verifier 负责“批改”。如果一道题生成了很多候选解答，Verifier 会分别判断这些解答的可信度，然后将得分最高的候选作为最终答案。因此整个系统不再要求 Generator 第一次就必须答对，而是允许它先探索多个可能的推理路径，再通过 Verifier 从中进行筛选。

---

整个方法可以理解为三个连续阶段。

![Lecture figure](../assets/Pasted%20image%2020260919214459.png)

- 首先利用训练集对一个语言模型进行微调，得到 **Generator**，使它能够针对数学题生成自然语言推理过程。
- 接下来，对于训练集中的每一道问题，都让 Generator 大量采样，论文中的设置是**每道题生成 100 个候选解答**。这些候选中自然会同时包含正确和错误的推理，因此可以根据最终答案是否正确，为每一个候选赋予一个二分类标签：

$$
Y_i^j = \begin{cases}
1, & \text{候选解答正确}\\
0, & \text{候选解答错误}
\end{cases}
$$
- 这样就构造出了大量“题目—候选解答—正确性标签”的训练数据。
- 最后，再利用这些数据**训练 Verifier**，让它学习判断一个模型生成的解答是否正确。在测试阶段，对一个新问题同样生成多个候选答案，Verifier 对所有候选进行评分，再选择得分最高的那个作为最终输出。这本质上就是一种 **generate-and-rank（先生成、再排序）** 的范式。
---

这里有一点非常重要：Verifier 的训练数据并不是人工重新编写大量“错误答案”，而是直接从 Generator 自己的采样结果中得到的。这样做的好处是，Verifier 在训练时看到的错误分布和真正测试时 Generator 容易产生的错误更加接近。例如 Generator 可能特别容易犯某种算术错误、漏掉一个条件，或者生成逻辑看似完整但最终答案错误的推理。**Verifier 正是在这些“模型真实会犯的错误”上进行学习**，因此它学习的并不是抽象意义上的数学正确性，而是学会区分 Generator 产生的正确和错误 completion。

---

![Lecture figure](../assets/Pasted%20image%2020260919214906.png)

Verifier 本身仍然是一个语言模型，只是在原语言模型基础上增加了一个较小的 **scalar head（标量预测头）**，用于预测正确性分数。

它的训练目标包含两部分：
- 一部分仍然保留原始语言模型的 next-token prediction，也就是根据前面的 token 预测下一个 token；
- 另一部分则是 Verifier 的正确性预测目标，即根据已经看到的题目和解答内容预测该解答正确或错误。

因此它实际上采用的是一种 **joint objective（联合训练目标）**。在计算正确性预测损失时，题目部分对应的 token 会被 mask 掉，不参与 verification loss，只有 solution 部分的 token 被用于计算验证目标。这样模型重点学习的是“给定问题之后，这段解答是否可信”。

---

Verifier 的评分方式还可以有不同设计。

- 最简单的是 **sentence-level verification**：等模型读完整个候选解答之后，只输出一个标量分数，即根据整个 generated solution 判断它正确的概率。
- 另一种是 **token-level verification**：模型在解答的每一个 token 之后都产生一个标量预测，即随着解题过程逐步推进，不断更新对整个解答正确性的判断。不过这里的 token-level 并不等于后来常见的“每一步都有人工过程标签”的 Process Reward Model。这里仍然只有整个解答最终正确或错误的标签，只是在每个 solution token 位置都训练预测这个标签。最终用于给整个 solution 排名的分数，则取最后一个 token 后得到的预测值。也就是说，它是在整个解答被完整读取后得到最终评分，而不是人工指出“具体哪一步错了”。

---

具体的训练流程是：

- 首先在 GSM8K 训练集上把 Generator **微调 2 个 epoch**；
- 然后针对每一道训练题从 Generator 中**采样 100 个 completions**，并根据最终答案将每一个 completion 标记为 correct 或 incorrect；
- 最后利用这一大批自动生成的二分类数据**训练 Verifier 1 个 epoch**。到了测试阶段，对于一道题生成多个候选答案：

$$
S_1, S_2, \ldots, S_k
$$

Verifier 分别计算：

$$
V(Q, S_1), V(Q, S_2), \ldots, V(Q, S_k)
$$

最终选择：

$$
S^* = \arg\max_{S_i} V(Q, S_i)
$$

因此 Verifier 实际上承担的是一个 **ranking / selection** 的作用：Generator 负责扩大候选解空间，而 Verifier 负责从这个候选空间中识别最有可能正确的答案。

---

![Lecture figure](../assets/Pasted%20image%2020260919215529.png)

实验说明了这种方法什么时候真正有效。

在**训练数据比较少的时候**，单纯训练 Verifier 并没有明显优势，甚至可能比直接继续微调 Generator 更差。原因是 Verifier 本身也需要大量数据才能学习到稳定的“正确与错误”的区别；训练集太小时容易发生过拟合。所以在数据规模为几百或一千左右时，直接 finetuning 往往更加有效。

但**随着训练数据规模扩大**，情况发生了明显变化。无论是 6B 还是 175B 模型，Verifier 的收益都会越来越明显。在 6B 模型上，数据量较小时 Verification 明显低于 Finetuning，但当训练集增加到几千题以后，Verification 开始超过 Finetuning；在接近完整 GSM8K 训练集规模时，Verifier 的 test solve rate 已经显著高于单纯微调。175B 模型上的趋势更加明显：数据规模足够大以后，“大量采样 + Verifier 选择”的性能提升远远高于继续对 Generator 进行普通微调。这说明 Verifier 本身也存在明显的数据规模效应：**小数据时容易过拟合，大数据时才能体现其优势。**

---

<img src="../assets/Pasted%20image%2020260919220620.png" alt="Lecture figure" width="318">

**Generator 和 Verifier 的规模应该怎么分配？**

**Verification Ablation（验证器消融实验）**。实验分别使用 6B 和 175B 两种规模的 Generator 与 Verifier，组合成四种情况：6B Generator + 6B Verifier、6B Generator + 175B Verifier、175B Generator + 6B Verifier，以及 175B Generator + 175B Verifier。横轴是训练集规模，纵轴是测试集解题成功率。实验结果非常明确：Generator 的规模对最终性能影响明显大于 Verifier 的规模。也就是说，**“大 Generator + 小 Verifier”明显优于“小 Generator + 大 Verifier”**。例如在训练数据充分时，175B Generator 即使只搭配 6B Verifier，性能仍然明显高于 6B Generator 搭配 175B Verifier。

这个结果很好理解，因为 Verifier 的作用只是从 Generator 已经生成出的候选答案中进行选择。如果 Generator 本身能力太弱，正确答案根本没有出现在候选集合里，那么再强的 Verifier 也无能为力。

而强 Generator 可以显著提高候选集合中出现正确答案的概率，也就是提高前面所说的 coverage / pass@k。一旦正确答案已经存在于候选集合中，即使 Verifier 比较小，也仍然有机会将它识别出来。因此整个系统实际上受到两个条件约束：

$$
\text{最终成功} = \text{正确答案被生成} \cap \text{正确答案被选中}
$$

其中第一个条件是第二个条件的前提。这也是为什么论文中的结论是：**扩大 Generator 比扩大 Verifier 更重要**。如果计算资源有限，与其使用一个很小的 Generator 再配一个特别大的 Verifier，不如优先保证 Generator 足够强，再搭配一个相对较小但有效的 Verifier。

---

<img src="../assets/Pasted%20image%2020260919220827.png" alt="Lecture figure" width="390">

**测试时是不是采样越多越好？**

这里增加 test-time compute 的方法非常直接：对每一道测试题生成越来越多的 candidate completions，然后使用 Verifier 从中选择最终答案。横轴是每道测试题生成的答案数量，从 25、50、100、200、400，一直增加到 3200；纵轴是 Test Solve Rate，也就是最终解题成功率。

**最开始增加采样数量确实非常有效**。例如从 25 个候选增加到 50、100、200 个候选时，最终正确率持续明显上升。这背后的原因和前面讲的 pass@k 完全一致：采样越多，正确答案至少出现一次的可能性越大。

当 k 从 25 增加到大约 400 时，最终 solve rate 从大约 35% 上升到接近 40%。

但这张图同时展示了一个更重要的现象：**采样并不是越多越好。** 在大约 400 个 completions 时，性能达到最高点；继续增加到 800、1600、3200 个候选以后，最终 solve rate 反而开始下降。也就是说：

$$
25 \rightarrow 400: \quad \text{更多采样} \Rightarrow \text{性能提高}
$$

但：

$$
400 \rightarrow 3200: \quad \text{更多采样} \Rightarrow \text{性能下降}
$$
这恰好说明，现实系统中的 Test-Time Scaling 与理想的 pass@k 并不是一回事。理论上的 pass@k 只关心“正确答案有没有出现”，因此随着 k 增大基本不会下降；但实际系统还必须依靠一个**不完美的 Verifier**来选择答案。候选数量越大，Verifier 面对的选择空间也越大。如果大量错误答案中出现一些“非常像正确答案”的高分错误样本，Verifier 就可能错误地把它们排到真正的正确答案前面。

---

[Let's Verify Step by Step (arXiv 2023)](https://arxiv.org/abs/2305.20050)

大语言模型在数学推理中不仅可能出现幻觉，还很容易在某一个中间推理步骤中犯逻辑错误，而数学推理具有明显的链式依赖关系：前面一步出错，后续即使推导形式看起来很合理，也可能全部建立在错误前提上。因此，只看最后答案是否正确，无法充分判断一条推理链究竟是否可信。**ORM 对整个解答只给一个总体正确性奖励，而 PRM 则进一步对解答中的每一个推理步骤分别进行评价**。

具体来说，对于一道数学题，**Generator** 首先生成包含 Step 1、Step 2、……、Final Answer 的**完整推理过程**。ORM 只关心最终结果是否正确，相当于学习一个概率：

$$
P(R_N = 1 \mid Q, S)
$$

其中 Q 是问题，S 是整个解答。它最终只输出一个整体分数，表示“这个完整 solution 最终是正确的概率”。这种监督相对便宜，因为只需要把模型最终答案和 ground truth 进行比较，就可以**自动得到 correct / incorrect 标签**，不需要人工逐步检查推理过程。

PRM 则不同。它会对每个推理步骤分别预测正确概率：

$$
P(R_1 = 1),\; P(R_2 = 1),\; \dots,\; P(R_N = 1)
$$

也就是说，它不仅回答“最终答案对不对”，还试图回答“第一步是否正确、第二步是否正确、第三步是否正确……”。最终对一整条推理链的 PRM 分数可以由各步骤正确概率组合得到，例如：

$$
\text{Score}_{PRM} = \prod_{i=1}^{N} P(R_i = 1)
$$

这样**只要某一个关键步骤的正确概率很低，整个 solution 的最终得分就会明显下降**。相比之下，ORM 主要根据完整解答结束位置上的预测给出一个整体分数，并不能精确指出错误究竟发生在哪一步。

---

过程监督之所以重要：

- 首先是因为它能够解决更精细的 **credit assignment（信用分配）** 问题。只看最终答案时，我们只能知道整条解答“成功还是失败”，却不知道是哪一个步骤导致了失败；PRM 则能够把反馈精确分配到具体推理步骤，因此可以定位错误出现的位置。

- 其次，它能够减少 **false positive（假阳性）**。有些解答虽然中间推理存在错误，但由于后面又碰巧进行了另一个错误操作，最后得到了正确答案。如果只做 Outcome Supervision，ORM 可能因为最终答案正确而给它较高奖励；但 PRM 会发现中间步骤本身是不成立的，因此不会把这种“错误推理碰巧得到正确结果”的 solution 当成高质量推理。
- 从 AI Alignment 的角度分析，过程监督更**鼓励模型产生人类能够理解并认可的推理过程**，而不只是追求一个碰巧正确的最终结果。

---

**PRM 的主要代价是训练数据非常昂贵**，因为最终答案是否正确可以自动判断，而每一个中间步骤是否正确通常需要人工检查。为此，工作收集了大量人类反馈来训练 PRM，并构建了 **PRM800K**：大约包含 **80 万个 step-level 标签，覆盖约 1.2 万道数学题**。每个推理步骤会被人工标注为 positive、negative 或 neutral，因此训练得到的 PRM 能够学习什么样的局部推理步骤是可靠的，而不仅仅是学习最终答案与标准答案是否一致。

为了降低如此昂贵的人工标注成本，作者使用了 **Active Learning（主动学习）**。Generator 会针对每一道数学题生成大量 candidate solutions，但并不是随机选择这些答案交给人工标注，而是优先挑选当前 PRM “最容易被骗”的样本。例如，某个 solution 的最终答案实际上是错的，但当前 PRM 却给出了很高的分数，那么这种样本就是非常有价值的 **convincing wrong answer**：它看起来很可信，却包含模型还没有学会识别的错误。让人工优先标注这些困难样本，可以针对性修复 PRM 的弱点。实验结果表明，这种主动学习策略相比随机选择样本进行人工标注，数据效率提高到了约 **2.6 倍**。

---

训练时，Generator 使用 GPT-4 base model；ORM 和 PRM 都是在 GPT-4 基础上进一步 fine-tune 得到的，但监督信号不同。ORM 的训练数据可以写成：

$$
(\text{generator sample}, \text{final-answer correctness})
$$

也就是一个完整生成结果加最终 correct / wrong 标签。PRM 的训练数据则是：

$$
(\text{generator sample}, \text{step-wise correctness})
$$
其中每个步骤的正确性标签来自 PRM800K。于是 ORM 学到的是“看完整个答案以后预测总体是否正确”，而 PRM 学到的是“随着推理一步一步进行，判断每一步是否合理”。

<img src="../assets/Pasted%20image%2020260919223724.png" alt="Lecture figure" width="625">

实验中最重要的结论是：**PRM 明显优于 ORM 和 Majority Voting，而且随着 Best-of-N 中 N 增大，这个优势反而越来越明显。** 这和前面讲的 inference-time scaling 正好连接起来。当只生成少量候选时，Verifier 的选择任务相对简单；但随着 N 增大，大量候选中会出现越来越多“看起来非常合理但实际错误”的答案。Majority Voting 只根据频率选择答案，ORM 又只有整体正确性判断，因此都比较容易被这些高质量错误答案干扰。而 PRM 能深入到推理步骤内部判断错误，因此候选集合越大，其精细验证能力越有价值。

特别值得注意的是，实验发现，在某些非常困难的问题上，Generator 生成的候选答案中**正确答案比例甚至低于 5%**，PRM 仍然能够把正确 solution 找出来。这非常重要，因为这种情形正是前面讨论的 Inference-Time Scaling 的典型困难场景。例如生成 100 个答案时：

$$
5\text{ 个正确} + 95\text{ 个错误}
$$

此时 Majority Voting 几乎不可能依赖“正确答案是多数”取得成功，但 PRM 并不依赖频率，而是通过检查推理过程本身进行排序。因此它可以在正确答案非常稀有的情况下依然把正确答案排到前面。这实际上说明 PRM 更接近我们真正需要的 robust verifier。

实验还比较了主动学习和数据规模的作用。使用 Active Learning 挑选高价值困难样本训练出来的 PRM，优于使用随机样本训练的 PRM，说明“标哪些数据”与“标多少数据”同样重要。同时，一个有意思的发现是：**少量高质量的 process supervision 可以达到与大量 outcome supervision 相近甚至更好的效果。** 这是因为每一个 step-level 标签包含的信息比单纯的最终正确/错误标签更加细粒度。Outcome label 只提供一个 bit 式的最终反馈，而 process supervision 可以告诉模型具体在哪一步发生错误，因此单位人工监督所传递的结构化信息更丰富。

![Lecture figure](../assets/Pasted%20image%2020260919223959.png)

最后，PRM 还表现出了较好的 **OOD（Out-of-Distribution）泛化能力**。也就是说，即使测试数据与 PRM 训练时的数学题分布并不完全一致，PRM 仍然可以用于判断这些新领域中的推理质量。在 AP Calculus、AP Chemistry、AP Physics 和 AMC10/12 等数据上，PRM 总体上都优于 ORM 和 Majority Voting。例如综合结果中，ORM 为 **63.8%**，Majority Voting 为 **61.3%**，而 PRM 达到 **72.9%**。这说明 PRM 学到的并不完全只是某一个数学数据集里的表面模式，而在一定程度上学习到了更通用的“推理步骤是否合理”的判断能力。不过论文表述也比较谨慎：PRM 能够容忍的是**一定程度的 distribution shift**，并不意味着任意远距离的领域迁移都能保持同样效果。

---

[Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations (arXiv 2023)](https://arxiv.org/abs/2312.08935)

