<script setup lang="ts">
import { ref } from "vue";
import MessageBubble from "./MessageBubble.vue";
import ChatInput from "./ChatInput.vue";
import type { Message } from "../types";
import { DocumentService } from "../services/DocumentService";

const documentService = new DocumentService();

// 生成UUID的函数
const generateUUID = (): string => {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
};

// 会话ID
const session_id = ref<string>(generateUUID());

// 新建对话
const handleNewConversation = () => {
  session_id.value = generateUUID();
  messages.value = [
    {
      id: 1,
      content: "你好！我是文档助手，请问你想了解关于文档的什么内容？",
      sender: "ai",
      timestamp: new Date(),
    },
  ];
};

const messages = ref<Message[]>([
  {
    id: 1,
    content: "你好！我是文档助手，请问你想了解关于文档的什么内容？",
    sender: "ai",
    timestamp: new Date(),
  },
]);

const handleSendMessage = async (content: string) => {
  // 添加用户消息
  const userMessage: Message = {
    id: messages.value.length + 1,
    content,
    sender: "user",
    timestamp: new Date(),
  };
  messages.value.push(userMessage);

  // 使用文档服务生成AI回复（调用后端API）
  const aiMessage: Message = {
    id: messages.value.length + 1,
    content: "正在查询中...",
    sender: "ai",
    timestamp: new Date(),
  };
  messages.value.push(aiMessage);

  try {
    console.log("开始调用generateResponse...");
    const responseContent = await documentService.generateResponse(
      content,
      session_id.value
    );
    console.log("收到响应内容:", responseContent);

    // 更新AI消息内容 - 通过索引直接修改数组元素
    const aiMessageIndex = messages.value.findIndex(
      (msg) => msg.id === aiMessage.id
    );
    if (aiMessageIndex !== -1) {
      messages.value[aiMessageIndex].content = responseContent;
      messages.value[aiMessageIndex].timestamp = new Date();
      console.log("消息已更新:", messages.value[aiMessageIndex]);
    }
  } catch (error) {
    console.error("Error generating response:", error);
    const aiMessageIndex = messages.value.findIndex(
      (msg) => msg.id === aiMessage.id
    );
    if (aiMessageIndex !== -1) {
      messages.value[aiMessageIndex].content =
        "抱歉，查询文档时出现错误。请稍后重试。";
      messages.value[aiMessageIndex].timestamp = new Date();
    }
  }
};
</script>

<template>
  <div class="chat-container">
    <div class="chat-header">
      <div class="header-content">
        <div class="header-title">
          <h1>文档助手</h1>
          <p>查询文档内容</p>
        </div>
        <button
          class="new-conversation-btn"
          @click="handleNewConversation"
        >
          新建对话
        </button>
      </div>
    </div>

    <div class="chat-messages">
      <MessageBubble
        v-for="message in messages"
        :key="message.id"
        :message="message"
      />
    </div>

    <ChatInput @send="handleSendMessage" />
  </div>
</template>

<style scoped>
.chat-container {
  width: 100%;
  max-width: 1200px;
  height: 100vh;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  background-color: #ffffff;
  box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
  border-radius: 8px;
  overflow: hidden;
}

/* PC端响应式优化 */
@media (min-width: 1024px) {
  .chat-container {
    margin: 1rem auto;
    height: calc(100vh - 2rem);
  }

  .chat-header {
    padding: 1.5rem 2rem;
  }

  .chat-messages {
    padding: 2rem;
  }
}

@media (max-width: 768px) {
  .chat-container {
    max-width: 100%;
    border-radius: 0;
    box-shadow: none;
  }

  .message-bubble {
    max-width: 85%;
  }
}

.chat-header {
  background-color: #4caf50;
  color: white;
  padding: 1rem;
  text-align: center;
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  max-width: 1200px;
  margin: 0 auto;
}

.header-title {
  text-align: center;
  flex: 1;
}

.chat-header h1 {
  margin: 0;
  font-size: 1.5rem;
}

.chat-header p {
  margin: 0.25rem 0 0 0;
  font-size: 0.9rem;
  opacity: 0.9;
}

.new-conversation-btn {
  background-color: white;
  color: #4caf50;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 0.5rem;
  font-size: 0.9rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
  white-space: nowrap;
}

.new-conversation-btn:hover {
  background-color: #f0f0f0;
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}

.new-conversation-btn:active {
  transform: translateY(0);
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}
</style>