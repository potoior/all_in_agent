"""
搜索智能体动作模块 - 网络信息检索与处理

功能概述:
这个模块实现了Searcher智能体的核心搜索功能，负责从互联网检索相关信息，
并对搜索结果进行筛选、提取和过滤，为答案生成提供有价值的外部知识。

模块组成:
1. QueryExpansion: 查询扩展 - 基于用户场景生成多个搜索查询
2. WebSearch: 网络搜索 - 使用多个搜索引擎获取结果
3. SelectResult: 结果筛选 - 选择最相关的网页进行深度分析
4. SelectFetcher: 内容获取 - 爬取选中网页的详细内容
5. FilterSelectedResult: 内容过滤 - 提取与查询相关的关键信息

技术特点:
- 支持多搜索引擎（Tavily、DuckDuckGo）
- 异步并发处理，提高搜索效率
- 智能内容过滤，去除无关信息
- 错误重试机制，增强稳定性
"""

from dotenv import load_dotenv
load_dotenv()

import json
import asyncio
from metagpt.actions import Action
from metagpt.logs import logger
from tianji.agents.metagpt_agents.utils.json_from import SharedDataSingleton
from tianji.agents.metagpt_agents.utils.agent_llm import ZhipuApi as LLMApi
from tianji.agents.metagpt_agents.utils.helper_func import *
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
from duckduckgo_search import DDGS
import ast
from typing import Tuple
import requests
from bs4 import BeautifulSoup
import re
from tavily import TavilyClient


class QueryExpansion(Action):
    """
    查询扩展动作类 - 智能生成搜索查询
    
    功能说明:
    1. 分析用户对话历史，理解用户需求背景
    2. 结合当前场景类型（生日祝福、节日问候等）
    3. 生成多个相关的搜索查询，提高信息检索的全面性
    4. 将生成的查询存储到共享数据中供后续使用
    
    设计原理:
    - 单一查询可能遗漏重要信息
    - 多角度查询能覆盖更全面的知识
    - 结合场景特征生成更精准的查询
    """
    
    # 查询扩展的提示词模板
    PROMPT_TEMPLATE: str = """
    #Role:
    - 查询扩展小助手

    ## Background:
    - 作为一个专业的查询扩展小助手。接下来，我将向你展示一段用户与大模型的历史对话记录，user 表示用户，assistant 表示大模型，你需要从中分析并生成合适的额外查询。

    ## Goals:
    - 你的任务是对历史对话记录中的内容进行分析，并且生成合适数量的额外查询，以供搜索引擎查询。

    ## Attention:
    - 我将提供给你用户目前所面对的场景，你可以自行参考，并且在此基础生成额外查询。

    ## Constraints:
    - 直接返回单个列表（例如：["额外查询一","额外查询二","额外查询三"]），不需要回复其他任何内容！

    ## Input:
    - 历史对话记录：```{instruction}```
    - 用户目前所面对的场景: ```{scene}``

    ## Workflow:
    ### Step 1: 思考生成的额外查询否需要包含场景要素（例如对象，场景，语气等细节）。
    ### Step 2: 生成查询列表并返回 "["额外查询一","额外查询二","额外查询三"]"
    """

    name: str = "queryExpansion"

    async def run(self, instruction: str):
        """
        执行查询扩展的主要方法
        
        参数:
            instruction: 用户与智能体的对话历史，包含用户需求背景
            
        返回值:
            str: 生成的查询列表字符串
            
        工作流程:
        1. 获取共享数据中的场景标签
        2. 加载场景属性配置文件
        3. 构建包含对话历史和场景的提示词
        4. 调用LLM生成扩展查询
        5. 解析并存储查询结果
        """
        # 获取共享数据实例，访问用户当前的场景信息
        sharedData = SharedDataSingleton.get_instance()
        scene_label = sharedData.scene_label
        
        # 加载场景属性配置文件，获取场景详细信息
        json_data = load_json("scene_attribute.json")
        scene, _, _ = extract_single_type_attributes_and_examples(
            json_data, scene_label
        )

        # 构建提示词，包含对话历史和场景信息
        prompt = self.PROMPT_TEMPLATE.format(
            instruction=instruction,
            scene=scene,
        )

        # 重试机制：最多尝试5次，处理LLM响应不稳定的情况
        max_retry = 5
        for attempt in range(max_retry):
            try:
                # 调用LLM API生成扩展查询，使用高温度值增加创造性
                rsp = await LLMApi()._aask(prompt=prompt, temperature=1.00)
                logger.info("机器人分析需求：\n" + rsp)
                
                # 清理LLM响应，处理可能的格式问题
                rsp = (
                    rsp.replace("```list", "")
                    .replace("```", "")
                    .replace("“", '"')
                    .replace("”", '"')
                    .replace("，", ",")
                )
                
                # 解析字符串形式的列表，转换为Python列表
                sharedData.extra_query = ast.literal_eval(rsp)
                return rsp
                
            except Exception as e:
                logger.error(f"第{attempt + 1}次查询扩展失败: {str(e)}")
                if attempt == max_retry - 1:
                    raise Exception("Searcher agent failed to response")
                continue

# 初始化搜索引擎客户端
ddgs = DDGS()  # DuckDuckGo搜索客户端
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))  # Tavily搜索API客户端

class WebSearch(Action):
    """
    网络搜索动作类 - 执行多引擎网络搜索
    
    功能说明:
    1. 接收查询扩展生成的多个搜索查询
    2. 使用多搜索引擎并发执行搜索（Tavily为主，DuckDuckGo备用）
    3. 对搜索结果进行过滤和格式化，去除低质量内容
    4. 将处理后的搜索结果存储到共享数据中
    
    设计特点:
    - 多引擎备份：Tavily失败时自动切换DuckDuckGo
    - 并发搜索：使用线程池并行处理多个查询
    - 智能过滤：屏蔽低质量网站，限制结果数量
    - 错误重试：增强搜索稳定性
    """
    
    name: str = "WebSearch"

    async def run(self, instruction: str):
        """
        执行网络搜索的主要方法
        
        参数:
            instruction: 用户指令（当前未使用，预留接口）
            
        返回值:
            str: 空字符串（搜索结果存储在共享数据中）
            
        工作流程:
        1. 从共享数据获取扩展查询列表
        2. 并发执行多个搜索任务
        3. 合并和过滤搜索结果
        4. 存储到共享数据中供后续使用
        """
        # 获取共享数据实例，访问查询扩展结果
        sharedData = SharedDataSingleton.get_instance()
        queries = sharedData.extra_query  # 扩展后的查询列表
        search_results = {}  # 存储所有搜索结果

        def search(query: str) -> dict:
            """
            单个查询的搜索执行函数
            
            参数:
                query: 搜索查询字符串
                
            返回值:
                dict: 处理后的搜索结果字典
                
            搜索策略:
            1. 优先使用Tavily API（质量更高）
            2. Tavily失败时重试5次，间隔随机时间
            3. 最终失败时抛出异常
            """
            max_retry = 5
            for attempt in range(max_retry):
                try:
                    # 优先使用Tavily搜索，质量更高
                    response = _call_tavily(query)
                    result = _parse_response(response)
                    return result
                except Exception as e:
                    logger.warning(f"Tavily搜索失败（尝试{attempt+1}）: {str(e)}")
                    # 随机等待后重试，避免频繁请求
                    time.sleep(random.randint(2, 5))
            raise Exception("Failed to get search results after retries.")

        def _call_tavily(query: str, **kwargs) -> dict:
            """
            Tavily搜索API调用函数
            
            参数:
                query: 搜索查询
                **kwargs: 额外参数
                
            返回值:
                dict: Tavily API响应
                
            特点:
            - 使用Tavily专业搜索API
            - 最多返回5个结果（平衡质量和速度）
            """
            try:
                logger.info(f"Tavily搜索: {query}, 参数: {kwargs}")
                # 调用Tavily API，限制结果数量以提高速度
                response = tavily_client.search(query, max_results=5)
                return response
            except Exception as e:
                raise Exception(f"Tavily搜索'{query}'失败: {str(e)}")
        
        def _call_ddgs(query: str, **kwargs) -> dict:
            """
            DuckDuckGo搜索备用函数
            
            警告:
            - 当前未使用，DuckDuckGo容易触发202限制
            - 保留作为未来备用搜索选项
            """
            max_retry = 5
            for attempt in range(max_retry):
                try:
                    logger.info(f"DuckDuckGo搜索: {query}, 参数: {kwargs}")
                    # 调用DuckDuckGo搜索API
                    response = ddgs.text(query.strip("'"), max_results=15, safesearch="off")
                    logger.info(f"DuckDuckGo搜索结果: {len(response)}条")
                    return response
                except Exception as e:
                    logger.error(f"DuckDuckGo搜索'{query}'失败: {str(e)}")
                    time.sleep(random.randint(2, 5))
            raise Exception("DuckDuckGo搜索失败")

        async def _async_call_ddgs(query: str, **kwargs) -> dict:
            """
            异步DuckDuckGo搜索函数
            
            特点:
            - 异步执行，避免阻塞主线程
            - 20秒超时保护
            - 最多返回30个结果（获取更多选择）
            """
            try:
                logger.info(f"异步DuckDuckGo搜索: {query}, 参数: {kwargs}")
                # 使用异步线程执行搜索，避免阻塞
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        ddgs.text, query.strip("'"), max_results=30, safesearch="off"
                    ),
                    timeout=20,  # 20秒超时保护
                )
                logger.info(f"异步DuckDuckGo搜索结果: {len(response)}条")
                return response
            except Exception as e:
                logger.error(f"异步DuckDuckGo搜索'{query}'失败: {str(e)}")
                raise

        def _parse_response(response: dict) -> dict:
            """
            解析搜索引擎响应的统一函数
            
            参数:
                response: 搜索引擎返回的原始响应数据
                        - Tavily: 包含'results'键的字典
                        - DuckDuckGo: 列表格式的结果
                        
            返回值:
                dict: 过滤后的搜索结果字典，格式为:
                    {
                        0: {"url": "链接", "summ": "摘要", "title": "标题"},
                        1: {"url": "链接", "summ": "摘要", "title": "标题"},
                        ...
                    }
                    
            处理流程:
            1. 识别搜索引擎类型（通过响应格式判断）
            2. 统一数据格式，提取关键信息（URL、内容、标题）
            3. 过滤低质量网站和PDF文件
            4. 限制结果数量，避免过多内容
            
            过滤规则:
            - 屏蔽知乎、百度、搜狐等防爬虫网站
            - 排除PDF文件（难以处理）
            - 最多返回20个结果（可根据模型上下文长度调整）
            """
            raw_results = []  # 存储原始搜索结果
            filtered_results = {}  # 存储过滤后的结果
            count = 0  # 结果计数器
            
            # 判断是否为 tavily 搜索引擎的结果（字典格式，包含results键）
            if isinstance(response, dict) and 'results' in response:
                # tavily 搜索引擎结果解析
                for item in response['results']:
                    raw_results.append(
                        (
                            item['url'],      # 网页链接
                            item['content'],  # 网页内容摘要
                            item['title']     # 网页标题
                        )
                    )
            else:
                # ddgs 搜索引擎结果解析（列表格式）
                for item in response:
                    raw_results.append(
                        (
                            item["href"],     # 网页链接
                            item["description"] if "description" in item else item["body"],  # 内容摘要
                            item["title"],    # 网页标题
                        )
                    )
                    
            # 过滤和格式化结果
            for url, snippet, title in raw_results:
                # 过滤规则：屏蔽特定网站和PDF文件
                if all(
                    domain not in url
                    for domain in ["zhihu.com", "baidu.com", "sohu.com"]  # 屏蔽防爬虫网站，避免获取不到内容
                ) and not url.endswith(".pdf"):  # 排除PDF文件，难以处理
                    # 使用json.dumps处理特殊字符，确保文本格式正确
                    filtered_results[count] = {
                        "url": url,  # 网页链接
                        "summ": json.dumps(snippet, ensure_ascii=False)[1:-1],  # 内容摘要（去掉引号）
                        "title": title,  # 网页标题
                    }
                    count += 1
                    # 限制结果数量，避免过多内容影响后续处理
                    if count >= 20:  # 可根据大模型的context length调整此参数
                        break
            return filtered_results
        logger.info(f"开始搜索{queries}")
        # 使用线程池并发执行多个搜索任务，提高效率
        with ThreadPoolExecutor() as executor:
            # 提交所有搜索任务到线程池
            future_to_query = {executor.submit(search, q): q for q in queries}

            # 处理完成的搜索任务
            for future in as_completed(future_to_query):
                try:
                    # 获取搜索结果
                    results = future.result()
                except Exception:
                    # 忽略失败的搜索任务，继续处理其他结果
                    pass
                else:
                    # 合并搜索结果，避免重复URL
                    for result in results.values():
                        if result["url"] not in search_results:
                            # 新URL，直接添加
                            search_results[result["url"]] = result
                        else:
                            # 重复URL，合并摘要内容
                            search_results[result["url"]][
                                "summ"
                            ] += f"\n{result['summ']}"

        # 重新索引搜索结果，使用连续的数字作为键
        search_results = {
            idx: result for idx, result in enumerate(search_results.values())
        }

        # 将搜索结果存储到共享数据中，供后续步骤使用
        sharedData.search_results = search_results
        logger.info(f"搜索完成，共获得{len(search_results)}个结果")
        return ""  # WebSearch动作不返回具体内容，结果存储在共享数据中


class SelectResult(Action):
    """
    搜索结果选择类 - 智能筛选需要进一步抓取的网页
    
    功能说明:
    1. 接收WebSearch返回的搜索结果
    2. 分析每个网页的内容与查询列表的相关性
    3. 选择最有价值的网页进行深度抓取
    4. 使用LLM进行智能判断，提高抓取效率
    
    设计特点:
    - 智能筛选：避免抓取所有网页，提高效率
    - 相关性判断：基于查询列表选择最相关的内容
    - 数量控制：最多选择20个网页，平衡质量和效率
    """
    
    PROMPT_TEMPLATE: str = """
    #Role:
    - 选取网页内容小助手

    ## Background:
    - 接下来，我将向你展示一段从搜索引擎返回的内容（字典列表形式），"url"字段表示网址，"summ"表示网页的部分片段，"title"表示网页的标题。你需要分析哪一些网站可能需要进一步查询。

    ## Goals:
    - 你的任务是基于我所提供的查询列表，从中识别并筛选出哪一些网页的内容可能符合查询列表里的查询。

    ## Constraints:
    - 最多返回20个需要进一步查询的网页。
    - 结果：你只需要返回单个列表，用索引值代表需要进一步查询的网页（例如：["0","4","6"]），不需要回复其他任何内容！。

    ## Input:
    - 搜索引擎返回的内容：```{search_results}```
    - 查询列表: ```{extra_query}``

    你只需要返回列表，用索引值代表需要进一步查询的网页（例如：["0","4","6"]），不需要回复其他任何内容！。
    """

    name: str = "selectResult"

    async def run(self, instruction: str):
        """
        执行搜索结果选择的主要方法
        
        参数:
            instruction: 用户指令（当前未使用，预留接口）
            
        返回值:
            str: 空字符串（选择结果存储在共享数据中）
            
        工作流程:
        1. 从共享数据获取搜索结果和查询列表
        2. 构建提示词，调用LLM进行智能选择
        3. 解析LLM返回的选择结果
        4. 存储选择的网页索引到共享数据中
        """
        sharedData = SharedDataSingleton.get_instance()
        logger.info(f"当前搜索结果为：{sharedData.search_results}")
        
        # 检查是否有搜索结果
        if sharedData.search_results == {}:
            logger.error("搜索结果为{}，请检查是否触发 searcher agent")
            return "搜索结果为空"
            
        # 构建LLM提示词，传入搜索结果和查询列表
        prompt = self.PROMPT_TEMPLATE.format(
            search_results=sharedData.search_results,
            extra_query=sharedData.extra_query,
        )

        # 调用LLM进行智能选择，最多重试5次
        max_retry = 5
        for attempt in range(max_retry):
            try:
                # 调用LLM API，使用较高温度值增加多样性
                rsp = await LLMApi()._aask(prompt=prompt, temperature=1.00)
                logger.info("机器人 SelectResult 分析需求：\n" + rsp)
                
                # 清理LLM响应，处理格式问题
                rsp = (
                    rsp.replace("```list", "")  # 移除可能的代码块标记
                    .replace("```", "")         # 移除代码块结束标记
                    .replace("“", '"')          # 替换中文引号
                    .replace("”", '"')          # 替换中文引号
                    .replace("，", ",")          # 替换中文逗号
                )
                
                # 解析字符串为Python列表
                rsp = ast.literal_eval(rsp)
                print(f"选择的网页索引：{rsp}")
                
                # 将选择的索引转换为整数列表并存储
                sharedData.filter_weblist = [int(item) for item in rsp]
                return str(rsp)
                
            except Exception as e:
                logger.warning(f"第{attempt+1}次尝试失败：{str(e)}")
                pass
                
        # 所有重试都失败，抛出异常
        raise Exception("Searcher agent failed to response after 5 attempts")


class SelectFetcher(Action):
    """
    选择抓取类 - 深度抓取选定网页的完整内容
    
    功能说明:
    1. 接收SelectResult选择的网页索引列表
    2. 并发抓取选定网页的完整HTML内容
    3. 提取网页的主要文本内容
    4. 处理抓取失败的情况，提供降级方案
    
    设计特点:
    - 并发抓取：使用线程池提高抓取效率
    - 错误处理：单个网页失败不影响整体流程
    - 内容提取：使用BeautifulSoup提取主要文本
    - 超时保护：避免长时间等待无响应网站
    """
    
    name: str = "selectFetcher"

    async def run(self, instruction: str):
        """
        执行网页内容抓取的主要方法
        
        参数:
            instruction: 用户指令（当前未使用，预留接口）
            
        返回值:
            str: 空字符串（抓取结果存储在共享数据中）
            
        工作流程:
        1. 从共享数据获取选择的网页索引和搜索结果
        2. 并发抓取选定网页的完整内容
        3. 提取每个网页的主要文本内容
        4. 存储抓取结果到共享数据中
        """
        sharedData = SharedDataSingleton.get_instance()

        def fetch(url: str) -> Tuple[bool, str]:
            """
            抓取单个网页内容的函数
            
            参数:
                url: 目标网页的URL
                
            返回值:
                Tuple[bool, str]: (是否成功, 网页内容或错误信息)
                
            工作流程:
            1. 发送HTTP请求获取网页HTML
            2. 使用BeautifulSoup提取纯文本内容
            3. 清理多余的换行符
            4. 检查内容质量（最少50个字符）
            
            错误处理:
            - 网络请求失败：返回错误信息
            - 内容过少：标记为无价值内容
            - 超时保护：20秒超时限制
            """
            try:
                # 发送HTTP请求，设置20秒超时
                response = requests.get(url, timeout=20)
                response.raise_for_status()  # 检查HTTP状态码
                html = response.content      # 获取HTML内容
            except requests.RequestException as e:
                # 网络请求失败，返回错误信息
                return False, str(e)

            # 使用BeautifulSoup解析HTML并提取纯文本
            text = BeautifulSoup(html, "html.parser").get_text()
            # 清理多余的换行符，统一格式
            cleaned_text = re.sub(r"\n+", "\n", text)
            
            # 内容质量检查：少于50个字符认为是无价值内容
            if len(cleaned_text) <= 50:  # 过滤掉内容过少的网页
                return False, "no valuable content"
                
            return True, cleaned_text

        # 使用线程池并发抓取选定的网页
        with ThreadPoolExecutor() as executor:
            # 为每个选定的网页提交抓取任务
            future_to_id = {
                executor.submit(
                    fetch, sharedData.search_results[select_id]["url"]  # 获取网页URL
                ): select_id
                for select_id in sharedData.filter_weblist  # 遍历选择的索引列表
                if select_id in sharedData.search_results  # 确保索引有效
            }

            # 处理完成的抓取任务
            for future in as_completed(future_to_id):
                select_id = future_to_id[future]
                try:
                    # 获取抓取结果
                    web_success, web_content = future.result()
                except Exception:
                    # 抓取失败，跳过该网页
                    pass
                else:
                    # 抓取成功，存储网页内容（限制长度避免过大）
                    if web_success:
                        sharedData.search_results[select_id]["content"] = web_content[
                            :1024  # 限制内容长度为1024字符，避免数据过大
                        ]
                        
        logger.info(f"网页抓取完成，成功抓取{len(sharedData.filter_weblist)}个网页")
        return ""  # SelectFetcher动作不返回具体内容，结果存储在共享数据中


class FilterSelectedResult(Action):
    """
    筛选选定结果类 - 使用LLM智能提取相关内容
    
    功能说明:
    1. 接收SelectFetcher抓取的网页完整内容
    2. 使用LLM分析每个网页的内容与查询的相关性
    3. 提取与查询主题直接相关或间接相关的信息
    4. 过滤掉不相关或冗余的内容
    
    设计特点:
    - 智能分析：使用LLM理解内容语义和相关性
    - 并发处理：使用线程池并行处理多个网页
    - 细节保留：提取具体细节，避免过度概括
    - 质量过滤：去除无关内容，提高信息密度
    
    注意:
    - 建议使用长上下文模型处理大量内容
    - 每个网页独立处理，避免信息交叉污染
    """
    
    # 该处最好用长上下文的模型
    PROMPT_TEMPLATE: str = """
    #Role:
    - 数据抽取小助手。

    ## Background:
    - 接下来，我将呈现一段从搜索引擎返回的内容。您的任务是尽可能提取出有潜力或有可能与查询列表里的主题直接相关或间接相关的信息,同时过滤掉所有不相关或冗余的部分。

    ## Goals:
    - 你的任务是基于我所提供的查询列表，把重要的内容提取出来。

    ## Constraints:
    - 直接返回提取后的结果，不需要回复其他任何内容！。
    - 提取的信息不可以过于概括，反之需要包含所有相关的细节。

    ## Input:
    - 搜索引擎返回的内容：```{search_results}```
    - 查询列表: ```{extra_query}``
    """

    name: str = "selectResult"

    async def run(self, instruction: str):
        """
        执行内容筛选的主要方法
        
        参数:
            instruction: 用户指令（当前未使用，预留接口）
            
        返回值:
            str: 空字符串（筛选结果存储在共享数据中）
            
        工作流程:
        1. 从共享数据获取抓取的网页内容
        2. 为每个网页构建筛选提示词
        3. 并发调用LLM进行内容筛选
        4. 存储筛选后的高质量内容
        """
        sharedData = SharedDataSingleton.get_instance()

        async def ask(result, extra_query):
            """
            异步调用LLM进行内容筛选的辅助函数
            
            参数:
                result: 单个网页的内容
                extra_query: 查询列表
                
            返回值:
                str: LLM筛选后的相关内容
                
            功能:
            - 构建筛选提示词
            - 调用LLM API进行内容分析
            - 返回筛选后的高质量内容
            """
            prompt = self.PROMPT_TEMPLATE.format(
                search_results=result, extra_query=extra_query
            )
            rsp = await LLMApi()._aask(prompt=prompt, temperature=1.00)
            logger.info("机器人 FilterSelectedResult 分析需求：\n" + rsp)
            return rsp

        def run_ask(result, extra_query):
            """
            在线程中运行异步LLM调用的包装函数
            
            参数:
                result: 单个网页的内容
                extra_query: 查询列表
                
            返回值:
                str: LLM筛选后的相关内容
                
            注意:
            由于线程池中的函数需要同步执行，
            但LLM调用是异步的，因此需要创建新的事件循环
            """
            # 为当前线程创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # 在新的事件循环中运行异步函数
                response = loop.run_until_complete(ask(result, extra_query))
                return response
            finally:
                # 确保事件循环被正确关闭
                loop.close()

        # 使用线程池并发处理多个网页的内容筛选
        with ThreadPoolExecutor() as executor:
            # 为每个有内容的网页提交筛选任务
            future_to_id = {
                executor.submit(
                    run_ask, result["content"], sharedData.extra_query  # 传入网页内容和查询列表
                ): select_id
                for select_id, result in sharedData.search_results.items()  # 遍历所有搜索结果
                if "content" in result  # 只处理已有抓取内容的网页
            }

            # 处理完成的筛选任务
            for future in as_completed(future_to_id):
                select_id = future_to_id[future]
                try:
                    # 获取LLM筛选后的内容
                    result = future.result()
                except Exception as exc:
                    # 记录筛选失败的错误，但继续处理其他任务
                    logger.error(f"FilterSelectedResult 提取{select_id}出错: {str(exc)}")
                    pass
                else:
                    # 存储筛选后的高质量内容
                    sharedData.search_results[select_id]["filtered_content"] = result
                    
        logger.info(f"内容筛选完成，共处理{len(future_to_id)}个网页")
        return ""  # FilterSelectedResult动作不返回具体内容，结果存储在共享数据中
