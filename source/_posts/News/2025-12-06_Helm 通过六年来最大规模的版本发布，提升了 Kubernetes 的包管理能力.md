---
date: 2025-12-06
title: Helm 通过六年来最大规模的版本发布，提升了 Kubernetes 的包管理能力
categories: [News, Helm 通过六年来最大规模的版本发布，提升了 Kubernetes 的包管理能力]
tag: News
---

# Helm 通过六年来最大规模的版本发布，提升了 Kubernetes 的包管理能力

![Helm通过六年来最大规模的版本发布，提升了Kubernetes的包管理能力](https://static001.infoq.cn/resource/image/b0/8e/b02e85f4646bd569d0a205822639738e.jpg)

# 注解

## Kubernetes $ Helm

```
1、Kubernetes和Helm 最初是 Google 开发的产品，后来捐赠给 CNCF（Cloud Native Computing Foundation，云原生计算基金会） 进行社区化托管。所以现在 Kubernetes和Helm 属于 开源社区项目（CNCF），不属于任何单一公司。

2、总结:
最初开发者：Google
当前归属：云原生基金会 CNCF（Linux Foundation 下属）
性质：开源容器编排系统、软件安装器

3、Kubernetes 与 Helm /helm/的关系
helmsman 美/ˈhelmzmən/ n.舵手
helm 美/helm/ n.舵轮；舵柄、 vt. 指挥；给…掌舵；
Kubernetes /ˌkuːbərˈneɪtiːz/ 舵手（helmsman）（希腊语）
Kubernetes 别称：K8s。这是一种技术界常见的命名缩写方式，叫做 “数字缩写法”（numeronym）。
缩写规则是：
保留开头的 K
保留结尾的 s
中间 8 个字母（ubernete）用数字 8 替代
所以：K + 8 个字母 + s = K8s

Kubernetes 是“操作系统”，Helm 是“Kubernetes 的软件包管理器”。
**1.Kubernetes 是“容器编排平台”**
- 部署、扩容、更新你的应用
- 操作的对象是 Deployment、Service、Ingress、ConfigMap、Secret 等 YAML 文件
- 所有部署都需要你提供一堆 YAML

**2. Helm 是“Kubernetes 的 apt / yum / pip”**
Helm 的作用就是：
把一大堆 Kubernetes YAML 文件打包成一个 Chart（软件包）**
比如一个应用可能需要：
- deployment.yaml
- service.yaml
- ingress.yaml
- configmap.yaml
- secret.yaml
- values.yaml（变量）
Helm 会把这些文件统一管理、模板化，进行一键部署、一键升级。


**简单类比**

| 角色           | 类比                  |
| -------------- | --------------------- |
| Kubernetes     | iOS / Android 系统    |
| Helm           | App Store（应用商店）  |
| Chart          | App 应用包            |
你用 Kubernetes 就像拿到一个手机系统，
 但没有 Helm 就像手机没有应用商店，需要你手动安装每个组件。
 
**Helm 解决了 Kubernetes 的哪些痛点？**

❌ Kubernetes 原生问题：

- YAML 文件太多
- 版本管理困难
- 配置修改容易出错
- 升级/回滚复杂

✔️ Helm 提供的能力：

- 模板化 YAML（减少重复）
- 一键部署（helm install）
- 一键升级（helm upgrade）
- 一键回滚（helm rollback）
- 官方 Chart 仓库（Redis、MySQL、Prometheus 等都能一键部署）

**他们的关系一句话总结**

> **Kubernetes 是平台，Helm 是让你更方便地在 Kubernetes 上部署应用的软件包管理工具。**
```

## SSA & kubectl apply 

```
**SSA = Server-Side Apply** 服务器端应用

SSA 的本质含义：

> **把原本由 kubectl (kube + ctl “控制 Kubernetes 的命令行工具”)客户端在本地处理的 apply 合并逻辑，交给 Kubernetes API 服务器在服务端执行。**

也就是说：
- 以前：kubectl apply 在你本地做“合并、对比、修改” → 再提交给服务器
- 现在：你把意图（YAML）发给 API Server → **由 API Server 负责执行合并、冲突检测、字段拥有权管理**

**为什么叫“Server-Side Apply（服务器端应用）”？**
因为所有 *apply* 的处理逻辑都从 **客户端迁移到了服务端**。
以前叫 **Client-Side Apply（CSA）**，现在改为 **Server-Side Apply（SSA）**。
 
 **SSA 有什么作用？（重点）**
 **1. 解决了 YAML 合并冲突问题（最大价值）**

Kubernetes 对象可能被多个控制器/工具修改，比如：

- Helm
- Kubectl
- Kustomize
- Operator
- Admission webhook

以前（CSA）：

- 客户端无法知道谁拥有某个字段
- 修改可能互相覆盖 → **导致不可预测冲突**

现在（SSA）：

- 服务器维护 **字段拥有者（field ownership）**
- 每个字段知道“谁修改了它”
- 冲突会被 API Server 明确拒绝，避免隐性覆盖

✔ 更稳定
 ✔ 更安全
 ✔ 更可控

**2. 增强声明式配置的“可重复性”**

SSA 强调：

> “最终状态由服务器严格管理，而不是客户端本地逻辑决定。”
好处：

- Render 出来的 YAML 多次 apply 结果一致
- 更像 Terraform 的声明式效果
- 更符合 GitOps 体系（ArgoCD、Flux）
```

## WebAssembly （WASM）

```
assembly 美/əˈsembli/ n.汇编

WebAssembly = 网站可执行的二进制格式；一种轻量、快速、安全的通用运行时。
更常见的中文说法：
“网页汇编”
“Web 二进制程序格式”
“可移植的运行时模块”

Helm 4 为什么支持 WASM 插件？
因为 WASM 有 3 大优势：
1. 可移植性极高（跨平台）
写一次 → Linux / Windows / Mac / ARM / x86 都能运行
（不像 Go 插件那样受本地 ABI 限制）

2. 安全（沙盒机制）
WASM 在沙盒内执行，不会随意访问文件系统、网络
→ 对 Helm 这类包管理工具非常重要

3. 性能高
接近原生语言速度。
```

## GitOps

```
GitOps = Git + Operations 
GitOps 工具（如 ArgoCD、FluxCD）
意思：
“基于 Git 的运维方式” 或 “以 Git 为中心的运维模型”
更通俗：
用 Git 来管理和自动化 Kubernetes 的配置、部署和运维。

GitOps 的作用是什么？为什么这么重要？
GitOps 的本质作用是：让集群的最终状态由 Git 仓库决定，而不是人工执行 kubectl 或手工部署。
1. 声明式配置统一存放在 Git
2. 自动部署（Continuous Delivery）
3. 自动回滚（Rollback 简单且可靠）
4. 安全、高可审计
5. 与 Helm、SSA 完美结合

总结：
GitOps = 用 Git 来管理 Kubernetes 的声明式配置，让部署、更新、回滚全部自动化，由系统而不是人工维护集群状态。
```

## kstatus

```
定义：
kstatus 是 Kubernetes SIG CLI 推出的一个“资源状态解析库（status library）”，用于判断 Kubernetes 资源的实际状态。

它不是一个单纯“查看部署状态的命令”，而是：
一个可供 Helm、GitOps 工具（如 ArgoCD、FluxCD）或其他控制器使用的库，用来标准化资源状态判定。
kstatus 是 Kubernetes 官方为各种工具提供的统一“资源状态判断标准”，确保它们都能一致、准确地判断 K8s 对象是否成功部署。
kstatus 通常不会被终端用户直接使用，可在 Helm 4、GitOps 工具和自定义控制器中发挥作用。

作用：
Helm 4 使用 kstatus 来判断 chart 是否成功安装 / 升级
GitOps 工具使用 kstatus 判断同步是否完成


```



# 正文

Helm（Kubernetes 应用程序包管理器）[正式发布4.0.0版本](https://www.cncf.io/announcements/2025/11/12/helm-marks-10-years-with-release-of-version-4/)。Helm 4 是六年来的首次重大升级，也标志着 Helm 在云原生计算基金会（CNCF）的指引下迎来了其十周年纪念。本次更新旨在解决可扩展性、安全性和开发工作流等方面的多项挑战。

Helm 4 的 SDK 有几项功能增强，旨在增强集成和开发体验。它采用现代 Go 日志接口支持多日志记录器，并通过可嵌入命令将 Helm 的功能直接嵌入到其他应用程序中。现在，Helm 4 还原生支持 server-side apply（SSA）（这是 Kubernetes 的一个特性，将 kubectl apply 命令的逻辑移到 API 服务器中）。这一变化反映了 Kubernetes 生态系统中广泛存在的一个发展趋势，确保基于 Helm 构建的集成方案既功能强大又符合现代集群行为。

![img](https://static001.geekbang.org/infoq/ec/ec455db5931a78a01e3de435e9436a73.png)

插件系统也已经重新构建：传统 Helm 插件仍然有效，但现在用户还可以编写 WebAssembly（WASM=WebAssembly+Module: WebAssembly 模块）插件，实现了更广泛的可移植性。此外，图表分发、性能以及图表签名和自动化测试机制也有所改进。

Helm 联合创始人 Rimantas Mocevicius 在[博文](https://rimusz.net/exploring-the-helm-v4-development-process-and-potential-features/)中指出，这些变化既反映了新特性，也反映了对 Helm v3 时代积累的设计债务的重新审视。

作为指引 Helm 4 发展的提案，[HIP-0012](https://helm.sh/community/hips/hip-0012)（Helm 改进提案）制定了明确的时间表：2024 年末启动规划阶段，2025 年中完成工程开发，2025 年 11 月正式发布。该提案特别强调，团队应在合理的时间内交付开发的功能，同时谨慎引入破坏性更改。路线图涵盖了针对 Kubernetes 集成的改进，包括 server-side apply（SSA）、现代化插件架构及图表 API 优化。同时，通过改进流程，明确并解决了贡献者参与度不足与维护积压等问题。

从更广泛的用户视角来看，Jimmy Song 在其博客中[指出](https://jimmysong.io/blog/helm-4-delivery-and-plugin-rebuild/)，此次发布使 Helm“超越了模板工具的范畴”，并实现了现代化升级。他认为，添加 SSA 使其与 GitOps 方法高度契合，而可重现构建、WASM 插件支持以及 kstatus 等状态解析工具的引入，则使其与现代 Kubernetes 范式保持了同步。Song 认为，这些变革意味着 Helm 正从单纯的图表渲染器转型为真正的部署编排器。

Helm 生态系统中颇具争议的其中一个问题是支持自定义资源定义（CRD）。有一项旨在增强 CRD 更新行为的[提案](https://github.com/helm/community/pull/379/files)已经提交，其中包括合并新版本、追加到版本列表、保留元数据以及确保回滚路径向后兼容。然而，截至 Helm 4 初始版本发布时，这些提案尚未被采纳。在安装图表时，Helm 仍会将 CRD 放置于专用的”crds/“目录中，但不会通过常规升级流程升级或删除 CRD。现有文档已明确指出：升级位于 crds 文件夹中的 CRD 仅会触发警告提示，不会被实际应用。

社区反馈反映出人们对这一遗漏的失望。在发布后不久的[一项Reddit讨论](https://www.reddit.com//r/kubernetes/comments/1ova60o/release_helm_v400_helmhelm/)中，有用户询问 CRD 行为是否有所改进。一位用户评论说：“CRD 方面仍然没有改进？“这说明 Helm 仍然无法安全地管理 CRD 生命周期。另一位用户报告说，他们组织里的工具和 CLI 工作流依赖于基于注解的 CRD，适应 Helm CRD 逻辑中的任何变化都将非同小可。

有其他评论（如[来自Heinan Cabouly的评论](https://blog.stackademic.com/helm-4-is-finally-here-but-argocd-solved-its-biggest-problem-3-years-ago-15352bcb70ad)）指出，像 Argo CD 这样的 GitOps 工具早在数年前就已经弥补了 Helm 最显著的工作流缺陷，暗示 Helm 4 的更新更像是追赶潮流而非开创性变革。然而，Helm 4 仍然被公认为是一个重要的里程碑，它提升了项目的长期可行性，而不是一个增量补丁。

从业者者和博主们对 Helm 4 部署安全性的提升表示欢迎，特别是新增的基于就绪状态的控制机制，这能减少发布过程中相互依赖的组件间的竞争条件。[Enix撰稿人](https://enix.io/en/blog/helm-4/,)Pierre-Louis Gueugnon 盛赞其更加智能的基于内容的图表缓存和性能优化，对于频繁进行大规模部署的团队而言，这些改进将大幅提升他们的生活质量。

展望未来，Helm 维护者已明确表示，未被 v4 采纳的功能可能在次要版本甚至 Helm 5 中进行考虑。社区将密切关注 CRD 升级何时能达到足以广泛采用的程度——安全、稳定且文档完善。

**原文链接：**

<https://www.infoq.com/news/2025/11/helm-4/>