export interface Message {
  id: number;
  content: string;
  sender: 'user' | 'ai';
  timestamp: Date;
}

// 后端API返回数据类型
export interface ChatResponse {
  content: string;
  response: string;
}

// 发送到后端的请求类型
export interface ChatRequest {
  query: string;
  session_id: string;
}