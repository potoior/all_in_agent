# RAG From Scratch (从零开始的RAG教程)

<!-- 
项目标题，表明这是一个从零开始构建RAG系统的教程
-->

<!-- 
LLMs are trained on a large but fixed corpus of data, limiting their ability to reason about private or recent information. 
大语言模型（LLMs）是在大量但固定的数据语料库上训练的，这限制了它们对私有或最新信息进行推理的能力。
-->
<!-- 大语言模型的局限性说明：由于LLM是在固定的训练数据上进行训练的，因此它们无法获取训练数据之外的信息，特别是私有信息或最新的信息 -->
大语言模型（LLMs）是在大量但固定的数据语料库上训练的，这限制了它们对私有或最新信息进行推理的能力。

<!-- 
Fine-tuning is one way to mitigate this, but is often [not well-suited for factual recall](https://www.anyscale.com/blog/fine-tuning-is-for-form-not-facts) and [can be costly](https://www.glean.com/blog/how-to-build-an-ai-assistant-for-the-enterprise).
微调是缓解这一问题的一种方法，但通常[不适合用于事实回忆](https://www.anyscale.com/blog/fine-tuning-is-for-form-not-facts)，并且[可能成本高昂](https://www.glean.com/blog/how-to-build-an-ai-assistant-for-the-enterprise)。
-->
<!-- 微调的局限性：虽然微调可以部分解决LLM的知识局限问题，但它主要用于改善模型的行为模式而非增加事实知识，并且实施成本较高 -->
微调是缓解这一问题的一种方法，但通常[不适合用于事实回忆](https://www.anyscale.com/blog/fine-tuning-is-for-form-not-facts)，并且[可能成本高昂](https://www.glean.com/blog/how-to-build-an-ai-assistant-for-the-enterprise)。

<!-- 
Retrieval augmented generation (RAG) has emerged as a popular and powerful mechanism to expand an LLM's knowledge base, using documents retrieved from an external data source to ground the LLM generation via in-context learning. 
检索增强生成（RAG）已成为一种流行且强大的机制来扩展大语言模型的知识库，它通过从外部数据源检索文档，利用上下文学习来支撑大语言模型的生成。
-->
<!-- RAG技术的优势：RAG通过检索外部数据源的相关文档，并结合上下文学习来增强大语言模型的生成能力，有效扩展了模型的知识边界 -->
检索增强生成（RAG）已成为一种流行且强大的机制来扩展大语言模型的知识库，它通过从外部数据源检索文档，利用上下文学习来支撑大语言模型的生成。

<!-- 
These notebooks accompany a [video playlist](https://youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x&feature=shared) that builds up an understanding of RAG from scratch, starting with the basics of indexing, retrieval, and generation. 
这些笔记本电脑 accompanies 一个[视频播放列表](https://youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x&feature=shared)，从索引、检索和生成的基础知识开始，逐步建立对RAG的理解。
-->
<!-- 教程说明：这些Jupyter Notebook文件配合视频播放列表使用，帮助学习者从基础概念开始逐步掌握RAG技术 -->
这些笔记本电脑 accompanies 一个[视频播放列表](https://youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x&feature=shared)，从索引、检索和生成的基础知识开始，逐步建立对RAG的理解。

<!-- 图片展示了RAG的详细架构 -->
<!-- RAG架构示意图：展示RAG系统的核心组件及其相互关系 -->
![rag_detail_v2](https://github.com/langchain-ai/rag-from-scratch/assets/122662504/54a2d76c-b07e-49e7-b4ce-fc45667360a1)

<!-- 视频播放列表链接 -->
<!-- 视频教程链接：提供与本教程配套的在线视频资源 -->
[Video playlist](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x)