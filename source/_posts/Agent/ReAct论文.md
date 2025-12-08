---
date: 2025-10-14
title: ReAct论文
categories: [Agent, ReAct论文]
tag: Agent
---

#  ReAct论文

REACT: SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS

ReAct：实现语言模型中推理与行动协同的框架



ABSTRACT

While large language models (LLMs) have demonstrated impressive performance across tasks in language understanding and interactive decision making, their abilities for reasoning (e.g. chain-of-thought prompting) and acting (e.g. action plan generation) have primarily been studied as separate topics. In this paper, we explore the use of LLMs to generate both reasoning traces and task-speciﬁc actions in an interleaved manner, allowing for greater synergy between the two: reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while actions allow it to interface with and gather additional information from external sources such as knowledge bases or environments. We apply our approach, named ReAct, to a diverse set of language and decision making tasks and demonstrate its effectiveness over state-of-the-art baselines in addition to improved human interpretability and trustworthiness. Concretely, on question answering (HotpotQA) and fact veriﬁcation (Fever), ReAct overcomes prevalent issues of hallucination and error propagation in chain-of-thought reasoning by interacting with a simple Wikipedia API, and generating human-like task solving trajectories that are more interpretable than baselines without reasoning traces. Furthermore, on two interactive decision making benchmarks (ALFWorld and WebShop), ReAct outperforms imitation and reinforcement learning methods by an absolute success rate of 34% and 10% respectively, while being prompted with only one or two in-context examples.

尽管大型语言模型（LLMs）在语言理解和交互式决策任务中展现出令人印象深刻的性能，但其“推理能力”（如 chain-of-thought 提示）和“行动能力”（如生成行动计划）一直主要被作为两个彼此独立的研究方向。本论文探讨了让 LLM 同时、交替地生成推理轨迹（reasoning traces）与任务相关行动（task-specific actions）的方法，使两者之间形成更强的协同作用：

- **推理轨迹**帮助模型建立、跟踪并更新行动计划，以及处理异常情况；
- **行动**使模型能够与外部资源（如知识库或环境）交互以获取额外信息。

我们将这一方法命名为 **ReAct**，并将其应用于多种语言任务与决策任务，实验结果表明其在效果上明显优于当前最先进的基线方法，同时在可解释性与可信度方面也有提升。

更具体地说：
 在问题回答（HotpotQA）与事实验证（Fever）任务上，ReAct 通过与一个简单的 Wikipedia API 交互，有效克服了 chain-of-thought 推理中常见的幻觉（hallucination）与错误传播问题，并生成了更接近人类、且比无推理轨迹的基线方法更具可解释性的任务求解轨迹。

此外，在两个交互式决策基准（ALFWorld 与 WebShop）中，ReAct 的表现分别比模仿学习和强化学习方法高出 **34%** 与 **10%** 的绝对成功率，而模型仅使用了 **一到两个** in-context 示例即可达到这一效果。

INTRODUCTION

A unique feature of human intelligence is the ability to seamlessly combine task-oriented actions with verbal reasoning (or inner speech, Alderson-Day & Fernyhough, 2015), which has been theorized to play an important role in human cognition for enabling self-regulation or strategization (Vygotsky, 1987; Luria, 1965; Fernyhough, 2010) and maintaining a working memory (Baddeley, 1992). Consider the example of cooking up a dish in the kitchen. Between any two speciﬁc actions, we may reason in language in order to track progress (“now that everything is cut, I should heat up the pot of water”), to handle exceptions or adjust the plan according to the situation (“I don’t have salt, so let me use soy sauce and pepper instead”), and to realize when external information is needed (“how do I prepare dough? Let me search on the Internet”). We may also act (open a cookbook to read the recipe, open the fridge, check ingredients) to support the reasoning and to answer questions (“What dish can I make right now?”). This tight synergy between “acting” and “reasoning” allows humans
to learn new tasks quickly and perform robust decision making or reasoning, even under previously unseen circumstances or facing information uncertainties.

人类智能的一个独特特征，是能够将 **以任务为导向的行动** 与 **语言推理（或称内在言语，inner speech，Alderson-Day & Fernyhough, 2015）** 无缝结合在一起。理论研究认为，这种内在语言对于人类认知具有重要作用：它能够支持自我调节或策略规划（Vygotsky, 1987；Luria, 1965；Fernyhough, 2010），并维持工作记忆（Baddeley, 1992）。

以厨房里做一道菜为例：在任何两个具体动作之间，我们可能会通过语言进行推理，以便：

- **跟踪任务进度**（“现在所有食材都切好了，我应该把水烧开”），
- **处理异常情况或根据环境调整计划**（“我没有盐，那就用酱油和胡椒代替吧”），
- **意识到何时需要外部信息**（“面团怎么做？我去网上查一下”）。

同时，我们也会采取行动（打开食谱查看配方、打开冰箱检查食材等），以辅助推理并回答问题（例如：“我现在能做什么菜？”）。

这种“行动”与“推理”之间的紧密协同，使人类能够快速学习新任务，并在以前未见的情境或信息不确定的情况下，仍然具备稳健的决策能力与推理能力。

Recent results have hinted at the possibility of combining verbal reasoning with interactive decisionmaking in autonomous systems. On one hand, properly prompted large language models (LLMs) have demonstrated emergent capabilities to carry out several steps of reasoning traces to derive

![1765183356099](https://bob-blog-image.oss-cn-shanghai.aliyuncs.com/typora-user-images/1765183356099.png)

近期的研究结果已经暗示，在自主系统中将 **语言推理** 与 **交互式决策** 结合起来是可能的。一方面，如果给予恰当的提示，大型语言模型（LLMs）已经展现出涌现能力，能够执行多步推理轨迹，从而推导出：





Figure 1: (1) Comparison of 4 prompting methods, (a) Standard, (b) Chain-of-thought (CoT,Reason Only), (c) Act-only, and (d) ReAct (Reason+Act), solving a HotpotQA (Yang et al., 2018) question. (2) Comparison of (a) Act-only and (b) ReAct prompting to solve an AlfWorld (Shridharet al., 2020b) game. In both domains, we omit in-context examples in the prompt, and only show task solving trajectories generated by the model (Act, Thought) and the environment (Obs).

**图 1：**
 (1) 比较四种提示方法：(a) 标准提示；(b) 思维链（CoT，仅推理）；(c) 仅行动（Act-only）；以及 (d) ReAct（推理 + 行动），用于解决一个 HotpotQA（Yang 等，2018）任务问题。
 (2) 比较 (a) 仅行动提示 与 (b) ReAct 提示，在求解 AlfWorld（Shridhar 等，2020b）游戏任务中的表现。

在这两个任务领域中，我们均省略了提示中的 in-context 示例，只展示由模型生成的任务求解轨迹（Act、Thought）以及由环境生成的观测信息（Obs）。

answers from questions in arithmetic, commonsense, and symbolic reasoning tasks (Wei et al., 2022). However, this “chain-of-thought” reasoning is a static black box, in that the model uses its own internal representations to generate thoughts and is not grounded in the external world, which limits its ability to reason reactively or update its knowledge. This can lead to issues like fact hallucination and error propagation over the reasoning process (Figure 1 (1b)). On the other hand, recent work has explored the use of pre-trained language models for planning and acting in interactive environments (Ahn et al., 2022; Nakano et al., 2021; Yao et al., 2020; Huang et al., 2022a), with a focus on predicting actions via language priors. These approaches usually convert multi-modal observations into text, use a language model to generate domain-speciﬁc actions or plans, and then use a controller to choose or execute them. However, they do not employ language models to reason abstractly about high-level goals or maintain a working memory to support acting, barring Huang et al. (2022b) who perform a limited form of verbal reasoning to reiterate spatial facts about the current state. Beyond such simple embodied tasks to interact with a few blocks, there have not been studies on how reasoning and acting can be combined in a synergistic manner for general task solving, and if such a combination can bring systematic beneﬁts compared to reasoning or acting alone.

在算术、常识以及符号推理任务中，大型语言模型已经能够通过推理链生成答案（Wei 等，2022）。然而，这种“思维链（chain-of-thought）”推理本质上是一种静态的黑箱过程：模型依赖自身的内部表示来生成思考内容，而这些思考并未与外部世界建立连接，这限制了模型进行**反应式推理（reactive reasoning）**或更新其知识的能力。
 这种特性可能导致事实幻觉（hallucination）以及在推理过程中产生错误传播（如图 1 的 (1b) 所示）。

另一方面，近年来已有研究探索利用预训练语言模型在交互式环境中执行规划与行动（Ahn 等，2022；Nakano 等，2021；Yao 等，2020；Huang 等，2022a），这些研究主要关注通过语言先验来预测行动。此类方法通常将多模态观察转换为文本，使用语言模型生成特定领域的行动或计划，然后由控制器选择或执行这些行动。

然而，这些方法并未使用语言模型来对高层目标进行抽象推理，也没有维持一种可支撑行动的工作记忆——唯一的例外是 Huang 等（2022b），其方法中包含一种有限的语言推理，用于重复当前状态下的空间事实。

在这些与“操作少量物体”相关的简单具身任务之外，现有研究尚未探讨如何以一种**协同方式**将“推理”与“行动”结合用于通用任务求解，也尚不清楚这种结合是否能够带来比单独推理或单独行动更系统性的优势。

In this work, we present ReAct, a general paradigm to combine reasoning and acting with language models for solving diverse language reasoning and decision making tasks (Figure 1). ReAct prompts LLMs to generate both verbal reasoning traces and actions pertaining to a task in an interleaved manner, which allows the model to perform dynamic reasoning to create, maintain, and adjust high-level plans for acting (reason to act), while also interact with the external environments
(e.g. Wikipedia) to incorporate additional information into reasoning (act to reason).

在这项工作中，我们提出 **ReAct**，一种通用范式，用于将推理与行动结合到语言模型中，以解决多样化的语言推理与决策任务（见图 1）。
 ReAct 通过提示 LLM **以交替的方式**同时生成与任务相关的语言推理轨迹（reasoning traces）和行动（actions），从而使模型能够：

- 进行动态推理，以**创建、维护并调整用于行动的高层计划**（reason to act），
- 同时与外部环境（例如 Wikipedia）交互，将获取的额外信息融入推理过程（act to reason）。

We conduct empirical evaluations of ReAct and state-of-the-art baselines on four diverse benchmarks: question answering (HotPotQA, Yang et al., 2018), fact veriﬁcation (Fever, Thorne et al., 2018), text-based game (ALFWorld, Shridhar et al., 2020b), and webpage navigation (WebShop, Yao et al., 2022). For HotPotQA and Fever, with access to a Wikipedia API that the model can interact with, ReAct outperforms vanilla action generation models while being competitive with chain-ofthought reasoning (CoT) (Wei et al., 2022). The best approach overall is a combination of ReAct and CoT that allows for the use of both internal knowledge and externally obtained information during reasoning. On ALFWorld and WebShop, two or even one-shot ReAct prompting is able to outperform imitation or reinforcement learning methods trained with 10三次幂 ∼ 10五次幂task instances,with an absolute improvement of 34% and 10% in success rates respectively. We also demonstrate the importance of sparse, versatile reasoning in decision making by showing consistent advantages over controlled baselines with actions only. Besides general applicability and performance boost, the combination of reasoning and acting also contributes to model interpretability, trustworthiness, and diagnosability across all domains, as humans can readily distinguish information from model’s internal knowledge versus external environments, as well as inspect reasoning traces to understand the decision basis of model actions.

我们在四个多样化的基准任务上，对 ReAct 与当前最先进的基线方法进行了实证评估：
 **问题回答**（HotPotQA，Yang 等，2018）、
 **事实验证**（Fever，Thorne 等，2018）、
 **文字游戏**（ALFWorld，Shridhar 等，2020b）、
 以及 **网页导航**（WebShop，Yao 等，2022）。

在 HotPotQA 和 Fever 任务中，当模型能够与 Wikipedia API 进行交互时，ReAct 的表现优于普通的行动生成（action-only）模型，并与思维链推理（CoT，Wei 等，2022）具备竞争力。
 总体而言，性能最优的方法是 **将 ReAct 与 CoT 结合**，使模型能够在推理过程中同时利用其内部知识和从外部获取的信息。

在 ALFWorld 与 WebShop 这两个任务中，即便仅使用两个甚至一个 ReAct 的 in-context 示例，模型的表现仍能优于使用 **10³ 至 10⁵** 个任务实例训练的模仿学习或强化学习方法，成功率分别提升了 **34%** 和 **10%** 的绝对值。

我们还通过持续优于仅使用行动（actions only）的对照模型，展示了稀疏且多样化的推理在决策任务中的重要性。

除了通用性强与性能提升之外，将推理与行动结合也显著提高了模型在所有任务中的 **可解释性、可信度与可诊断性**：
 人类可以轻松区分模型内部知识与外部环境信息的来源，并通过检查推理轨迹来理解模型行动决策的依据。

To summarize, our key contributions are the following: (1) we introduce ReAct, a novel promptbased
paradigm to synergize reasoning and acting in language models for general task solving; (2) we
perform extensive experiments across diverse benchmarks to showcase the advantage of ReAct in a
few-shot learning setup over prior approaches that perform either reasoning or action generation in
isolation; (3) we present systematic ablations and analysis to understand the importance of acting in
reasoning tasks, and reasoning in interactive tasks; (4) we analyze the limitations of ReAct under the
prompting setup (i.e. limited support of reasoning and acting behaviors), and perform initial ﬁnetuning
experiments showing the potential of ReAct to improve with additional training data. Scaling up
ReAct to train and operate on more tasks and combining it with complementary paradigms like
reinforcement learning could further unlock the potential of large language models.

总结而言，我们的主要贡献如下：

**(1)** 我们提出 **ReAct**，一种新颖的基于提示（prompt-based）的范式，用于在语言模型中协同推理与行动，以解决通用任务；

**(2)** 我们在多个多样化的基准任务上进行了广泛实验，展示了在少样本学习（few-shot）设定下，ReAct 相较于仅执行推理或仅执行行动生成的以往方法所具备的优势；

**(3)** 我们进行了系统性的消融实验与分析，以理解在推理任务中行动的重要性，以及在交互式任务中推理的重要性；

**(4)** 我们分析了在提示设定下 ReAct 的局限性（例如对推理与行动行为的支持程度有限），并进行了初步的微调实验，展示 ReAct 随着额外训练数据有进一步提升的潜力。

将 ReAct 扩展到更多任务上进行训练与应用，并将其与强化学习等互补范式结合，可能会进一步释放大型语言模型的潜能。

2 REACT: SYNERGIZING REASONING + ACTING

Consider a general setup of an agent interacting with an environment for task solving. At time
step t, an agent receives an observation ot ∈ O from the environment and takes an action at ∈ A
following some policy π(at|ct), where ct = (o1, a1, · · · , ot−1, at−1, ot) is the context to the agent.
Learning a policy is challenging when the mapping ct 7→ at is highly implicit and requires extensive
computation. For example, the agent shown in Figure 1(1c) is unable to generate the correct ﬁnal
action (Act 4) to ﬁnish the QA task as it requires complex reasoning over the trajectory context
(Question, Act 1-3, Obs 1-3). Similarly, the agent shown in Figure 1(2a) fails to comprehend from the
context that sinkbasin 1 does not contain peppershaker 1, thus keep producing hallucinating actions.

![1765186033650](https://bob-blog-image.oss-cn-shanghai.aliyuncs.com/typora-user-images/1765186033650.png)

The idea of ReAct is simple: we augment the agent’s action space to Aˆ = A ∪ L, where L is the
space of language. An action aˆt ∈ L in the language space, which we will refer to as a thought or a
reasoning trace, does not affect the external environment, thus leading to no observation feedback.
Instead, a thought aˆt aims to compose useful information by reasoning over the current context ct,
and update the context ct+1 = (ct, aˆt) to support future reasoning or acting. As shown in Figure 1,
there could be various types of useful thoughts, e.g. decomposing task goals and create action plans
(2b, Act 1; 1d, Thought 1), injecting commonsense knowledge relevant to task solving (2b, Act 1),
extracting important parts from observations (1d, Thought2, 4), track progress and transit action plans
(2b, Act 8), handle exceptions and adjust action plans (1d, Thought 3), and so on.



![1765186362213](https://bob-blog-image.oss-cn-shanghai.aliyuncs.com/typora-user-images/1765186362213.png)

However, as the language space L is unlimited, learning in this augmented action space is difﬁcult
and requires strong language priors. In this paper, we mainly focus on the setup where a frozen
large language model, PaLM-540B (Chowdhery et al., 2022)
1
, is prompted with few-shot in-context
examples to generate both domain-speciﬁc actions and free-form language thoughts for task solving
(Figure 1 (1d), (2b)). Each in-context example is a human trajectory of actions, thoughts, and
environment observations to solve a task instance (see Appendix C). For the tasks where reasoning is
of primary importance (Figure 1(1)), we alternate the generation of thoughts and actions so that the
task-solving trajectory consists of multiple thought-action-observation steps. In contrast, for decision
making tasks that potentially involve a large number of actions (Figure 1(2)), thoughts only need to appear sparsely in the most relevant positions of a trajectory, so we let the language model decide the
asynchronous occurrence of thoughts and actions for itself.

![1765186595782](https://bob-blog-image.oss-cn-shanghai.aliyuncs.com/typora-user-images/1765186595782.png)

