import type { ChatResponse, ChatRequest } from '../types';

export interface Document {
  id: string;
  title: string;
  content: string;
  tags: string[];
}

export class DocumentService {
  private documents: Document[] = [
    {
      id: 'doc1',
      title: '项目概述',
      content: '这是一个基于Vue3和TypeScript的文档聊天应用。该应用允许用户通过自然语言查询文档内容，获取相关信息。项目使用Vite作为构建工具，提供了快速的开发体验。',
      tags: ['概述', '项目', 'Vue3', 'TypeScript']
    },
    {
      id: 'doc2',
      title: '功能特性',
      content: '1. 实时聊天功能：用户可以与AI助手进行实时对话。\n2. 文档查询：支持通过关键词查询文档内容。\n3. 智能回复：AI助手会根据文档内容生成相关回复。\n4. 响应式设计：适配不同屏幕尺寸的设备。',
      tags: ['功能', '特性', '聊天', '查询']
    },
    {
      id: 'doc3',
      title: '技术栈',
      content: '前端框架：Vue 3.5.24\n开发语言：TypeScript 5.9.3\n构建工具：Vite 7.2.4\n样式：CSS3\n组件：Vue 单文件组件',
      tags: ['技术', '栈', 'Vue3', 'TypeScript', 'Vite']
    },
    {
      id: 'doc4',
      title: '使用指南',
      content: '1. 启动项目：npm run dev\n2. 构建项目：npm run build\n3. 预览构建：npm run preview\n4. 在聊天窗口中输入问题，AI助手会根据文档内容进行回复。',
      tags: ['使用', '指南', '命令', '开发']
    }
  ]

  /**
   * 根据关键词查询文档内容
   * @param query 关键词
   * @returns 相关文档列表
   */
  searchDocuments(query: string): Document[] {
    if (!query.trim()) return []

    const lowerQuery = query.toLowerCase()
    
    return this.documents.filter(doc => {
      // 检查标题、内容和标签是否包含关键词
      const titleMatch = doc.title.toLowerCase().includes(lowerQuery)
      const contentMatch = doc.content.toLowerCase().includes(lowerQuery)
      const tagMatch = doc.tags.some(tag => tag.toLowerCase().includes(lowerQuery))
      
      return titleMatch || contentMatch || tagMatch
    })
  }

  /**
   * 根据文档ID获取文档详情
   * @param id 文档ID
   * @returns 文档详情或undefined
   */
  getDocumentById(id: string): Document | undefined {
    return this.documents.find(doc => doc.id === id)
  }

  /**
   * 获取所有文档列表
   * @returns 所有文档
   */
  getAllDocuments(): Document[] {
    return this.documents
  }

  /**
   * 根据查询结果生成回复内容 - 发送请求到后端API
   * @param query 用户查询
   * @param session_id 会话ID
   * @returns 生成的回复内容
   */
  async generateResponse(query: string, session_id: string): Promise<string> {
    try {
      console.log('发送请求到后端:', { query, session_id });
      
      const requestBody: ChatRequest = {
        query,
        session_id
      };
      
      const response = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
      })

      console.log('后端响应状态:', response.status);

      // if (!response.ok) {
      //   throw new Error(`HTTP error! status: ${response.status}`)
      // }

      const data: ChatResponse = await response.json()
      console.log('后端返回的数据:', data);
      
      // 直接返回后端的response字段
      if (data.response) {
        console.log('返回response字段:', data.response);
        return data.response
      }
      
      // 如果没有response字段，尝试其他字段或返回默认消息
      console.warn('后端返回数据缺少response字段');
      return `后端返回数据格式异常，缺少response字段`
    } catch (error) {
      console.error('Error calling chat API:', error)
      // 如果后端请求失败，回退到本地模拟数据
      const results = this.searchDocuments(query)
      
      if (results.length === 0) {
        return `未找到与"${query}"相关的文档内容。请尝试其他关键词。(当前使用本地模拟数据)`
      }

      let response = `根据文档内容，关于"${query}"的信息如下：

(当前使用本地模拟数据)

`
      
      results.forEach((doc, index) => {
        response += `${index + 1}. ${doc.title}\n`
        response += `${doc.content}\n\n`
      })
      
      return response
    }
  }
}