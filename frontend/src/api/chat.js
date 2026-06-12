const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '/api';

export async function sendChatMessage(message, history) {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ message, history }),
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.detail || '聊天请求失败，请稍后重试。');
  }

  return data;
}

export async function requestEmotionInsight({ message, history, emotion, includeAiReview = false }) {
  const response = await fetch(`${API_BASE_URL}/insight`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message,
      history,
      emotion,
      include_ai_review: includeAiReview,
    }),
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.detail || '情绪画像生成失败，请稍后重试。');
  }

  return data.insight;
}
