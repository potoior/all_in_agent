<script setup lang="ts">
import type { Message } from "../types";

interface Props {
  message: Message;
}

const props = defineProps<Props>();

const formatTime = (date: Date) => {
  return date.toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
  });
};
</script>

<template>
  <div :class="['message-wrapper', `message-wrapper-${message.sender}`]">
    <div :class="['message-bubble', `message-bubble-${message.sender}`]">
      <div class="message-content">{{ message.content }}</div>
      <div class="message-time">{{ formatTime(message.timestamp) }}</div>
    </div>
  </div>
</template>

<style scoped>
.message-wrapper {
  display: flex;
  margin-bottom: 1rem;
}

.message-wrapper-user {
  justify-content: flex-end;
}

.message-wrapper-ai {
  justify-content: flex-start;
}

.message-bubble {
  max-width: 600px;
  padding: 1rem 1.25rem;
  border-radius: 1rem;
  position: relative;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
  font-size: 1rem;
}

.message-bubble-user {
  background-color: #4caf50;
  color: white;
  border-bottom-right-radius: 0.25rem;
}

.message-bubble-ai {
  background-color: #f9f9f9;
  color: #333;
  border-bottom-left-radius: 0.25rem;
  border: 1px solid #e0e0e0;
}

.message-content {
  margin-bottom: 0.5rem;
  line-height: 1.6;
  word-wrap: break-word;
  text-align: left;
}

.message-time {
  font-size: 0.8rem;
  opacity: 0.7;
  text-align: right;
}

.message-bubble-ai .message-time {
  color: #666;
}
</style>