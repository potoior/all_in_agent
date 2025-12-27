<script setup lang="ts">
import { ref, onMounted, watch } from "vue";

const emit = defineEmits<{
  send: [content: string];
}>();

const input = ref("");
const textareaRef = ref<HTMLTextAreaElement | null>(null);

const handleSend = () => {
  if (input.value.trim()) {
    emit("send", input.value.trim());
    input.value = "";
    adjustTextareaHeight();
  }
};

const handleKeyPress = (event: KeyboardEvent) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    handleSend();
  }
};

const adjustTextareaHeight = () => {
  if (textareaRef.value) {
    textareaRef.value.style.height = "auto";
    textareaRef.value.style.height =
      Math.min(textareaRef.value.scrollHeight, 120) + "px";
  }
};

watch(input, () => {
  adjustTextareaHeight();
});

onMounted(() => {
  adjustTextareaHeight();
});
</script>

<template>
  <div class="chat-input-container">
    <textarea
      ref="textareaRef"
      v-model="input"
      @keydown="handleKeyPress"
      placeholder="请输入您的问题..."
      class="chat-input"
      rows="1"
    ></textarea>
    <button
      @click="handleSend"
      class="send-button"
      :disabled="!input.trim()"
    >
      发送
    </button>
  </div>
</template>

<style scoped>
.chat-input-container {
  display: flex;
  gap: 1rem;
  padding: 1.5rem 2rem;
  background-color: white;
  border-top: 1px solid #e0e0e0;
  box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.05);
}

.chat-input {
  flex: 1;
  padding: 1rem 1.5rem;
  border: 2px solid #e0e0e0;
  border-radius: 2.5rem;
  resize: none;
  font-size: 1.1rem;
  font-family: inherit;
  transition: all 0.3s ease;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
}

.chat-input:focus {
  outline: none;
  border-color: #4caf50;
  box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.1);
}

.send-button {
  padding: 1rem 2.5rem;
  background-color: #4caf50;
  color: white;
  border: none;
  border-radius: 2.5rem;
  font-size: 1.1rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 2px 8px rgba(76, 175, 80, 0.3);
}

.send-button:hover:not(:disabled) {
  background-color: #45a049;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(76, 175, 80, 0.4);
}

.send-button:active:not(:disabled) {
  transform: translateY(0);
}

.send-button:disabled {
  background-color: #cccccc;
  cursor: not-allowed;
}
</style>