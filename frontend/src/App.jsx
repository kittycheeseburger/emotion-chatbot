import { Activity, Bot, Brain, Compass, Flag, RefreshCcw, Send, Sparkles, Target, UserRound } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { requestEmotionInsight, sendChatMessage } from './api/chat.js';
import { demoMessages } from './data/demoMessages.js';
import './styles/app.css';

const emotionTips = {
  焦虑: '建议先降低任务颗粒度，把注意力放在一个可完成动作上。',
  平静: '当前表达较稳定，适合继续推进计划或整理思路。',
  积极: '可以趁状态较好处理需要创造力或决策的任务。',
  负面: '当前情绪压力较高，建议先暂停一下，把问题拆成一个最小可执行步骤。',
  低落: '先承认当下的疲惫感，再选择一件低成本的小事恢复掌控感。',
  愤怒: '先给情绪降温，等表达更稳定后再处理冲突或做决定。',
};

function App() {
  const [messages, setMessages] = useState(demoMessages);
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [isReviewing, setIsReviewing] = useState(false);
  const [error, setError] = useState('');
  const [reviewError, setReviewError] = useState('');
  const [currentInsight, setCurrentInsight] = useState(null);
  const messageListRef = useRef(null);

  const userEmotionRecords = useMemo(
    () => messages.filter((message) => message.role === 'user' && message.emotion),
    [messages],
  );

  const latestEmotion = userEmotionRecords.at(-1)?.emotion;
  const latestUserMessage = userEmotionRecords.at(-1);

  useEffect(() => {
    const element = messageListRef.current;
    if (element) {
      element.scrollTop = element.scrollHeight;
    }
  }, [messages]);

  async function handleSubmit(event) {
    event.preventDefault();
    const text = input.trim();

    if (!text || isSending) return;

    setError('');

    const userMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      time: '现在',
    };

    const apiHistory = messages.map(({ role, content }) => ({ role, content }));

    setMessages((current) => [...current, userMessage]);
    setInput('');
    setIsSending(true);

    try {
      const result = await sendChatMessage(text, apiHistory);
      const assistantMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: result.reply,
        time: '现在',
      };

      setMessages((current) => {
        const nextMessages = [...current];
        const index = nextMessages.findIndex((message) => message.id === userMessage.id);

        if (index >= 0) {
          nextMessages[index] = {
            ...nextMessages[index],
            emotion: result.emotion,
          };
        }

        return [...nextMessages, assistantMessage];
      });
      setCurrentInsight(result.insight);
    } catch (requestError) {
      setError(requestError.message);
      setMessages((current) => current.filter((message) => message.id !== userMessage.id));
    } finally {
      setIsSending(false);
    }
  }

  async function handleGenerateReview() {
    if (!latestUserMessage?.content || !latestEmotion || isReviewing) return;

    setReviewError('');
    setIsReviewing(true);

    try {
      const history = messages.map(({ role, content }) => ({ role, content }));
      const insight = await requestEmotionInsight({
        message: latestUserMessage.content,
        history,
        emotion: latestEmotion,
        includeAiReview: true,
      });
      setCurrentInsight(insight);
    } catch (requestError) {
      setReviewError(requestError.message);
    } finally {
      setIsReviewing(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <header className="topbar">
          <div className="brand">
            <span className="brand-mark" aria-hidden="true">
              <Brain size={22} />
            </span>
            <div>
              <h1>情绪分析聊天机器人</h1>
              <p>RoBERTa 情绪识别 · 本地 RAG 策略检索</p>
            </div>
          </div>
          <div className={`status-pill ${isSending ? 'working' : ''}`}>
            <span className="status-dot" />
            {isSending ? '生成中' : 'RAG 本地'}
          </div>
        </header>

        <div className="content-grid">
          <section className="chat-panel" aria-label="聊天记录">
            <div className="panel-heading">
              <div>
                <h2>对话</h2>
                <p>上下文聊天</p>
              </div>
              <button className="icon-button" type="button" aria-label="智能建议">
                <Sparkles size={18} />
              </button>
            </div>

            <div className="message-list" ref={messageListRef}>
              {messages.map((message) => (
                <article className={`message-row ${message.role}`} key={message.id}>
                  <div className="avatar" aria-hidden="true">
                    {message.role === 'assistant' ? <Bot size={18} /> : <UserRound size={18} />}
                  </div>
                  <div className="message-bubble">
                    <div className="message-meta">
                      <span>{message.role === 'assistant' ? '机器人' : '我'}</span>
                      <time>{message.time}</time>
                    </div>
                    <p>{message.content}</p>
                  </div>
                </article>
              ))}
              {isSending && (
                <article className="message-row assistant">
                  <div className="avatar" aria-hidden="true">
                    <Bot size={18} />
                  </div>
                  <div className="message-bubble">
                    <div className="message-meta">
                      <span>机器人</span>
                      <time>现在</time>
                    </div>
                    <p>正在结合上下文和情绪分析生成回复...</p>
                  </div>
                </article>
              )}
            </div>

            <form className="composer" onSubmit={handleSubmit}>
              {error && <p className="composer-error">{error}</p>}
              <textarea
                aria-label="输入聊天内容"
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder="输入你想说的话..."
                rows="2"
                disabled={isSending}
              />
              <button className="send-button" type="submit" aria-label="发送消息" disabled={isSending}>
                <Send size={18} />
              </button>
            </form>
          </section>

          <aside className="insight-panel" aria-label="情绪分析">
            <div className="panel-heading">
              <div>
                <h2>情绪分析</h2>
                <p>当前状态</p>
              </div>
              <Activity size={20} />
            </div>

            <div className="emotion-summary">
              <span className="summary-emoji">{latestEmotion?.emoji ?? '🙂'}</span>
              <div>
                <p>当前情绪</p>
                <strong>{latestEmotion?.label ?? '暂无'}</strong>
              </div>
              <div
                className="score-ring"
                style={{
                  '--score': `${latestEmotion?.score ?? 0}%`,
                }}
              >
                <span>{latestEmotion?.score ?? 0}</span>
              </div>
            </div>

            {!latestEmotion && (
              <div className="empty-emotion">
                <span>🧠</span>
                <p>发送第一条消息后，这里会显示当前情绪类型和指数。</p>
              </div>
            )}

            <div className="suggestion-box">
              <h3>基础回应策略</h3>
              <p>{emotionTips[latestEmotion?.label] ?? '等待真实情绪模型返回结果后，自动生成针对性建议。'}</p>
            </div>

            <div className="insight-card">
              <div className="insight-title">
                <div>
                  <h3>AI 情绪画像</h3>
                  <p>{currentInsight?.source === 'rag' ? 'RAG 检索增强' : '实时画像'}</p>
                </div>
                <button
                  className="review-button"
                  type="button"
                  onClick={handleGenerateReview}
                  disabled={!latestEmotion || isReviewing}
                >
                  <RefreshCcw size={16} />
                  {isReviewing ? '检索中' : 'RAG 复盘'}
                </button>
              </div>

              {reviewError && <p className="review-error">{reviewError}</p>}

              <div className="insight-grid">
                <div className="insight-item">
                  <Flag size={17} />
                  <span>可能触发点</span>
                  <strong>{currentInsight?.trigger ?? '等待对话'}</strong>
                </div>
                <div className="insight-item">
                  <Target size={17} />
                  <span>对话目标</span>
                  <strong>{currentInsight?.goal ?? '发送消息后生成'}</strong>
                </div>
                <div className="insight-item">
                  <Compass size={17} />
                  <span>建议语气</span>
                  <strong>{currentInsight?.tone ?? '平和、开放'}</strong>
                </div>
              </div>

              <div className="next-action">
                <h4>下一步行动</h4>
                <p>{currentInsight?.next_action ?? '这里会结合情绪和上下文生成一个可执行的小步骤。'}</p>
              </div>

              <div className="strategy-source">
                <h4>检索策略</h4>
                <p>
                  {currentInsight?.strategy
                    ? `${currentInsight.strategy} · 匹配度 ${Math.round((currentInsight.retrieval_score ?? 0) * 100)}%`
                    : '发送消息后会从情绪策略知识库中检索最相关方案。'}
                </p>
              </div>

              <div className="review-summary">
                <h4>本轮复盘</h4>
                <p>{currentInsight?.summary ?? '点击 RAG 复盘后，会基于上下文和检索策略生成更完整的情绪摘要。'}</p>
              </div>
            </div>
          </aside>
        </div>
      </section>
    </main>
  );
}

export default App;
