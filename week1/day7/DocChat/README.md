# DocChat 项目文档

## 概述

DocChat 是一个基于 FastAPI 构建的文档聊天应用程序，允许用户上传文档并通过 AI 技术与它们进行对话。

## 项目结构

```
docchat/
├── app/                 # 应用主目录
│   ├── __init__.py      # Python 包初始化文件
│   ├── main.py          # 应用程序入口点
│   ├── schemas.py       # Pydantic 数据模型/模式定义
│   ├── services.py      # 业务逻辑实现
│   └── routes.py        # API 路由定义
├── main.py              # 原始入口点
└── pyproject.toml       # 项目依赖配置文件
```

## API 接口

### 健康检查
- `GET /health` - 检查 API 是否正常运行

### 文档管理
- `POST /api/v1/documents/` - 上传新文档
- `GET /api/v1/documents/{document_id}` - 获取特定文档
- `GET /api/v1/documents/` - 列出所有文档

### 聊天接口
- `POST /api/v1/chat` - 与文档进行对话

## 快速开始

1. 安装依赖:
   ```
   pip install -e .
   ```

2. 运行应用:
   ```
   python -m app.main
   ```

3. 访问 API 文档 `http://localhost:8000/docs`

## 主要依赖

- FastAPI - 现代高性能 Web 框架
- Uvicorn - ASGI 服务器实现
- Langchain - AI 应用编排框架
- OpenAI - AI 模型接口
- PyMuPDF - PDF 文档处理库