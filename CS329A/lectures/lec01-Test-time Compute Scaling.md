# CS329A Lecture 1: Test-time Compute Scaling

[Large Language Monkeys: Scaling Inference Compute with Repeated Sampling (arXiv 2024)](https://arxiv.org/pdf/2407.21787)
<img src="../assets/Pasted%20image%2020260919162043.png" alt="Lecture figure" width="640">
这篇论文的核心论点是：我们有一个LLM和一个Input Problem，我们让这个LLM反复回答这个问题，问的次数可以很多，然后，我们有一个Verifier，它会检验哪个回答是正确的回答，这个回答会成为系统的最终输出，这种方式我们称为 **Repeated Sampling**。

![Lecture figure](../assets/Pasted%20image%2020260919162611.png)

> Coverage（pass@k）：对一道题让模型独立生成 k 个答案，只要这 k 个答案里面**至少有 1 个正确答案**，这道题就算“被覆盖 / pass”。
>
> **k 越大，pass@k 越高**：假设模型对某道题**单次答对概率**是 p，如果简单假设每次采样相互独立，那么采样 k 次全部答错的概率是：(1−p)k，因此，<img src="../assets/Pasted%20image%2020260919164324.png" alt="Lecture figure" width="440">

通过 **Repeated Sampling** ，我们可以显著提升这些较弱模型的性能，甚至让它们超越那些更大、闭源的模型。

---

上面的公式是假设：每道题每次采样的成功概率都是固定的 p，而且各次采样近似独立。这是一个简单的概率模型，但是现实中的 benchmark 并不会严格服从上面的概率模型（有的题目非常简单/有的题目困难），所以整个 benchmark 的实际 Coverage 曲线不会严格服从：1−(1−p)k

研究者观察真实实验数据以后发现，用**exponentiated power law** 可以更好地拟合整体趋势。（**Scaling Laws for Inference**）

采样次数 k增加时，coverage / pass@k 会按照一条比较稳定的函数曲线上升。

<img src="../assets/Pasted%20image%2020260919163958.png" alt="Lecture figure" width="199">
其中：

- **k**：Number of Samples，也就是**一道题采样多少次**
- **c**：Coverage，也就是 **pass@k**
- **a,b**：通过实验数据拟合出来的参数

---

通过上面的知识，我们知道 **Repeated Sampling** 让正确答案出现的概率按规律增长，但是**真正困难的是如何从大量候选中可靠地找到它**。

这就需要我们讨论 **Verifier**

---

首先把问题分为两类：一些具有**自动验证机制**的任务特别适合 Inference-Time Scaling。例如：
- 代码生成可以通过单元测试判断程序是否正确
- 形式化数学证明可以交给 proof checker 验证
- CUDA 代码也可以与参考 PyTorch 实现进行结果比对

在这些任务中，Verifier 相对可靠，因此更多采样意味着更大的概率生成正确解，而正确解一旦出现，又能够被自动识别，所以采样数量的增加比较容易直接转化为能力提升。

对于**缺乏可靠自动验证器**的任务，情况就复杂得多。此时通常只能依赖 
- Majority Voting
- Reward Model
- LLM-as-a-Judge 
- 其他学习得到的 Ranker 来选择最终答案

其中 Majority Voting 的问题尤其明显，因为它隐含了“正确答案会成为多数”的假设，但这个假设经常并不成立。正确答案可能只是大量生成中的极少数，例如 100 个样本中只有 10 个正确，而某一种错误答案出现了 40 次。虽然 pass@100 已经是成功的，但多数投票最终仍然会选择错误答案。因此，高 pass@k 并不意味着 Majority Voting 的准确率也会很高。

![Lecture figure](../assets/Pasted%20image%2020260919171117.png)

Coverage，也就是 pass@k，可以理解为 Generator 的潜在能力上限，因为它只关心候选集合里是否存在正确答案；而 Majority Vote、Reward Model + Best-of-N 等方法则反映实际系统能否把正确答案选出来。实验中经常出现这样的情况：随着采样次数增加，pass@k 可以接近很高的水平，但使用 Reward Model 或 Majority Voting 后的最终成功率却停留在远低得多的位置。这说明很多题其实已经生成出了正确答案，只是在 Verification 和 Selection 阶段被丢失了。

因此，可以把这个现象理解为 **Generator–Verifier Gap**，即“生成能力”和“识别能力”之间的差距。当采样数量越来越大时，生成正确答案本身可能不再是主要瓶颈，真正的瓶颈会逐渐转向 Verification。尤其是在正确答案占比很低的情况下，Verifier 实际面对的是一个高度不平衡的排序问题：需要从成百上千个错误候选中找到少量正确候选。Reward Model 本身也可能会把某些看起来合理但实际上错误的答案排在正确答案之前，所以现实中的 Best-of-N 往往远低于 Oracle 情况下的理想表现。

![Lecture figure](../assets/Pasted%20image%2020260919171147.png)

---

[Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters (arXiv 2024)](https://arxiv.org/pdf/2408.03314)

![Lecture figure](../assets/Pasted%20image%2020260919173416.png)


在 **Parallel Sampling** ，我们看的是结果，使用 **Outcome Reward Model** 来进行打分
在 **Sequential Revisions** ，我们看的是过程，使用 **Process Reward Model**  来对构成解答的每一步进行打分。

![Lecture figure](../assets/Pasted%20image%2020260919203550.png)

- 对于 Best-of-N，我们使用 **Outcome Reward Model**
- 对于 Beam Search，我们使用 **Process Reward + Verifiers**

这篇论文提出 **Compute Optimal Strategy**

![Lecture figure](../assets/Pasted%20image%2020260919204040.png)


这篇论文的另一个观察是：对于简单和中等难度的问题，额外增加推理时间计算，可能比扩大模型的预训练更划算

![Lecture figure](../assets/Pasted%20image%2020260919204309.png)

---
[Archon: An Architecture Search Framework for Inference-Time Techniques (arXiv 2024)](https://arxiv.org/abs/2409.15254)

这篇论文要解决的问题是：我们怎么优化分配给不同问题的推理计算，然后设计出一套机制，既能得到高质量答案，又不浪费太多token和生成次数。

![Lecture figure](../assets/Pasted%20image%2020260919205732.png)
**Input:** 

- 需要优化的目标基准测试
- 推理调用预算
- 一组可用的 LLM
- 一些 Inference Time Techniques

**Optimizer:**

- 告诉我们如何把不同 LLM 与 不同的 Inference Time Techniques 结合起来（在给定的推理调用预算下），并获得高质量结果

---

首先看 Inference Time Techniques

![Lecture figure](../assets/Pasted%20image%2020260919210416.png)

![Lecture figure](../assets/Pasted%20image%2020260919210753.png)

---
提出 **Archon Architecture**

**![Lecture figure](../assets/Pasted%20image%2020260919211052.png)

![Lecture figure](../assets/Pasted%20image%2020260919211154.png)

