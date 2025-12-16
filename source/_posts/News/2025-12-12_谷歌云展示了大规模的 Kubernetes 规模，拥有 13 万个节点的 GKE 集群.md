---
date: 2025-12-12
title: 谷歌云展示了大规模的 Kubernetes 规模，拥有 13 万个节点的 GKE 集群
categories: [News, 谷歌云展示了大规模的 Kubernetes 规模，拥有 13 万个节点的 GKE 集群]
tag: News
---

# 谷歌云展示了大规模的 Kubernetes 规模，拥有 13 万个节点的 GKE 集群

# 注解

## etcd 是什么？

```
etcd 是一个分布式、一致性的键值存储系统，专为云原生应用和分布式系统设计，是 Kubernetes 集群的核心组件之一。它由 CoreOS（后被 Red Hat 收购）开发，基于 Raft 共识算法实现数据的高可用、强一致性和容错性，常被用于存储集群的关键配置、状态信息和元数据。

etcd 的单词组成
etcd 名称并非由四个独立单词缩写而成，其命名来源有明确的背景：
早期开发者将其命名为 “etc directory”（即 Unix/Linux 系统中存放配置文件的 /etc 目录）的缩写，寓意它是分布式系统的 “配置中心”；
后缀 “d” 取自 Unix/Linux 中 “守护进程（daemon）” 的惯例（如 sshd、nginxd`），表示 etcd 是一个后台运行的服务进程。

etcd 的核心作用
在 Kubernetes 集群中，etcd 是唯一的集群状态数据源，具体作用包括：
存储集群配置与元数据
保存 Kubernetes 所有资源对象的信息（如 Pod、Service、Deployment、Node 等），以及集群的网络策略、RBAC 权限规则、命名空间配置等。例如，当你执行 kubectl create pod 时，Pod 的定义会被持久化到 etcd 中。
2、提供一致性与高可用保障
基于 Raft 算法实现多节点数据同步，确保即使部分 etcd 节点故障，集群仍能正常读写数据（默认需至少 3 个节点组成集群，容忍 1 个节点故障；5 个节点可容忍 2 个故障）。
3、支持实时监听与变更通知
Kubernetes 的控制器（如 Deployment 控制器、Node 控制器）通过 Watch API 监听 etcd 中资源的变化，实时触发调度、扩容、故障恢复等操作。例如，当某个 Node 宕机，etcd 中该节点的状态更新会被控制器感知，进而将该节点上的 Pod 调度到其他节点。
4、作为集群的 “唯一真相源”
Kubernetes 所有组件（kube-apiserver、kube-scheduler、kube-controller-manager）均通过 kube-apiserver 访问 etcd，确保整个集群的状态一致，避免出现数据冲突或不一致。
5、适配大规模集群的扩展
原生 etcd 在超大规模集群（如十万级节点）下可能面临性能瓶颈，因此文中提到用基于 Spanner 的系统替代 etcd，以支持更大规模的集群并降低负载。
```

## Spanner

```

```

## pod

```
Pod 是 Kubernetes 中最小的部署和调度单元，不是单个容器，而是一个或多个紧密关联的容器的组合（这些容器共享网络命名空间、存储卷和生命周期）。简单来说，Pod 就像一个 “容器组”，里面的容器会被调度到同一个节点上，协同工作（比如一个容器运行应用，另一个容器负责日志收集）。
Pod 是 Kubernetes 操作的基础，所有资源（如 Deployment、StatefulSet）最终都会落地为 Pod 来运行，它也是 Kubernetes 保证应用高可用、弹性伸缩的核心载体。

Pod 的单词组成
Pod 并非由三个单词缩写而成，它的命名源自生物学概念 ——豆荚（pod），寓意里面的多个容器像豆荚里的豆子一样紧密结合、共享资源。
```



# 正文

谷歌 Kubernetes 引擎（Google Kubernetes Engine，GKE）背后的团队透露，他们成功构建并运行了一个拥有 13 万个节点的Kubernetes集群，使其成为迄今为止最大的公开披露的 Kubernetes 集群。这一里程碑突显了云原生基础设施的进步程度以及其对于 AI 和数据时代大规模、计算密集型工作负载的准备情况。

这一壮举是通过重构 Kubernetes 控制平面和存储后端的关键组件来实现的，用一个基于自定义[Spanner](https://cloud.google.com/blog/products/containers-kubernetes/gke-65k-nodes-and-counting)的系统取代传统的 etcd 数据存储，以支持大规模，并优化集群 API 和调度逻辑，以减少恒定节点和 Pod 更新带来的负载。工程团队还引入了新的工具，用于自动化、并行化的节点池配置和更快的调整大小，帮助克服了在这种规模下通常会阻碍响应性的典型瓶颈。

随着 AI 训练和推理工作负载的增加，通常需要数百或数千个 GPU 或高吞吐量 CPU 集群，运行庞大、统一的[Kubernetes](https://kubernetes.io/)集群的能力成为一个关键的推动因素。通过 13 万个节点的集群，大规模模型训练、分布式数据处理或全局微服务集群等工作负载可以在单个控制平面下进行管理，简化了编排和资源共享。

规模突破的核心是 Google 用定制的、基于[Spanner](https://cloud.google.com/spanner?hl=en)的存储层取代了[etcd](https://etcd.io/)作为主要的控制平面数据存储。。传统的 Kubernetes 依赖 etcd 来进行强一致性状态管理，但由于[写扩增](https://en.wikipedia.org/wiki/Write_amplification)、[监控扇出](https://en.wikipedia.org/wiki/Fan-out)和[领导者选举](https://medium.com/@contactomyna/leader-election-with-leases-in-distributed-systems-6fc46ea84b30)开销，etcd 在非常高的节点和 Pod 数量下成为扩展瓶颈。通过将集群状态卸载到 Spanner 中，谷歌获得了水平可扩展性、全局一致性以及节点、Pod、资源租赁等 API 对象的[自动分片](https://en.wikipedia.org/wiki/Shard_(database_architecture))。这极大地降低了 API 服务器的压力，并消除了通常将 Kubernetes 集群限制在数万个节点的共识瓶颈。对 API 服务器进行重构，以批量处理和压缩监控流量，防止因节点心跳和 Pod 状态更新而导致控制平面饱和。

对于基础设施资源，谷歌引入了高度并行化的节点配置和调度优化，以避免当数万个节点同时加入集群时发生的踩踏问题。节点池的创建使用了激进的并行性，而 kube-scheduler 则被调优以减少每个 Pod 的调度延迟并最小化全局锁定。网络和 IP 地址管理也被重新设计，以避免[CIDR](https://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing)耗尽和极端规模的路由表限制。至关重要的是，工程师将“集群规模”视为一个全栈系统问题，涵盖了 API 效率、数据库架构、调度算法和网络控制平面，而仅仅不是简单地增加资源配额。这种架构转变是 Kubernetes 能够从数万个节点进入真正的超大规模领域的原因。



这一里程碑也代表了对 GKE 限制的巨大飞跃。就在几年前，[GKE](https://cloud.google.com/blog/products/containers-kubernetes/gke-65k-nodes-and-counting)还支持多达 65,000 个节点的集群。虽然这些限制已经支持了大规模的工作负载，但 13 万个节点是该容量的两倍多，这强烈表明谷歌云支持其所谓的“AI 吉瓦时代”的雄心壮志。

 

尽管如此，谷歌警告说，这个集群是在“实验模式”下构建的，主要是作为验证可扩展性的概念验证。在大多数实际部署中，像自动扩展、网络策略、资源配额和调度限制这样的限制可能需[要更保守的配置](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/planning-large-clusters)。

 

对于追求大规模 AI 或数据工作负载的组织来说，这一公告表明，曾经被认为只适用于中小型服务的云原生基础设施现在可以扩展到数十万个节点。它表明，Kubernetes 在适当优化后，即使是最苛刻的计算需求下，仍然是一个可行的支柱。

 

然而，这种集群扩展并不是谷歌所独有的。在 2025 年 7 月，AWS[宣布](https://www.infoq.com/news/2025/09/aws-eks-kubernetes-ultrascale)EKS 现在支持多达 10 万个工作节点的集群，这比通常的限制有了显著的增长。这种增强是针对超大型 AI/ML 工作负载的：根据 AWS 的说法，这种规模的单个 EKS 集群可以支持多达 160 万个 Trainium 芯片或 80 万个 NVIDIA GPU，从而实现“超大规模的 AI/ML 工作负载，如最先进的模型训练、微调和智能推理。”

 

AWS 记录了他们如何通过在底层进行广泛的再工程来实现这种规模。他们通过调整 Kubernetes API 服务器、扩展控制平面容量、改进网络和镜像分发管道来优化数据平面以处理极端负载。在测试期间，集群处理了数百万个 Kubernetes 对象（节点、Pod、服务），即使在高扰动和高调度负载下，也能保持响应式的 API 延迟。

 

最近的 EKS 声明表明，GKE 所展示的可扩展性雄心并不是一家云供应商所独有的。一家主要的云提供商承诺提供具有 100K 节点的 Kubernetes 集群，这一事实加强了 Kubernetes 已经为“AI 吉瓦时代”做好了准备的论点。它还为公司提供了一个选择，以评估是否投资于定制的大规模工程（如 GKE 的 130K 节点构建）或通过 EKS 采用托管的大规模服务。

















