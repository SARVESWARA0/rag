'use client';
import React, { useState, useEffect, useRef } from 'react';
import { useChat } from 'ai/react';
import { Send, Loader2, ChevronDown, Menu, Settings, Sun, Moon } from 'lucide-react';
import Markdown from 'markdown-to-jsx';

const formatTime = (date) => {
  return new Date(date).toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    hour12: true
  });
};

const TypingIndicator = () => (
  <div className="flex space-x-2 p-3 bg-gray-50 rounded-lg max-w-[100px]">
    <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
    <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
    <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce"></div>
  </div>
);

export default function AdvancedChat() {
  const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat();
  const [theme, setTheme] = useState('light');
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);

  const toggleTheme = () => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleScroll = () => {
    if (chatContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = chatContainerRef.current;
      setShowScrollButton(scrollHeight - scrollTop - clientHeight > 100);
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <div className={`flex h-screen ${theme === 'dark' ? 'dark bg-gray-900' : 'bg-gray-50'}`}>
      {/* Sidebar */}
      <div className={`
        fixed lg:static lg:flex flex-col w-64 h-full bg-white dark:bg-gray-800 border-r 
        transition-transform duration-300 ease-in-out z-50
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
      `}>
        <div className="p-4 border-b dark:border-gray-700">
          <h2 className="text-xl font-bold dark:text-white">Chat History</h2>
        </div>
        <div className="flex-1 overflow-y-auto p-4">
          {/* Chat history items would go here */}
          <div className="space-y-2">
            {[1, 2, 3].map((i) => (
              <div key={i} className="p-3 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer">
                <div className="text-sm font-medium dark:text-white">Chat {i}</div>
                <div className="text-xs text-gray-500 dark:text-gray-400">Last message...</div>
              </div>
            ))}
          </div>
        </div>
        <div className="p-4 border-t dark:border-gray-700">
          <button
            onClick={toggleTheme}
            className="flex items-center space-x-2 text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white"
          >
            {theme === 'light' ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
            <span>{theme === 'light' ? 'Dark Mode' : 'Light Mode'}</span>
          </button>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col h-full">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b bg-white dark:bg-gray-800 dark:border-gray-700">
          <div className="flex items-center space-x-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="lg:hidden p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg"
            >
              <Menu className="w-5 h-5 dark:text-white" />
            </button>
            <h1 className="text-xl font-semibold dark:text-white">AI Assistant</h1>
          </div>
          <div className="flex items-center space-x-4">
            <div className="text-sm text-gray-500 dark:text-gray-400">
              {messages.length} messages
            </div>
            <button className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg">
              <Settings className="w-5 h-5 dark:text-white" />
            </button>
          </div>
        </div>

        {/* Messages Container */}
        <div 
          ref={chatContainerRef}
          onScroll={handleScroll}
          className="flex-1 overflow-y-auto p-4 space-y-4 dark:bg-gray-900"
        >
          {messages.map((m, index) => (
            <div
              key={m.id}
              className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] rounded-lg p-4 ${
                  m.role === 'user'
                    ? 'bg-blue-500 text-white'
                    : 'bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold">
                    {m.role === 'user' ? 'You' : 'AI'}
                  </span>
                  <span className="text-xs opacity-70">
                    {formatTime(new Date())}
                  </span>
                </div>
                <div className={`prose ${
                  m.role === 'user' ? 'prose-invert' : 'dark:prose-invert'
                } max-w-none`}>
                  <Markdown>{m.content}</Markdown>
                </div>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <TypingIndicator />
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Scroll to Bottom Button */}
        {showScrollButton && (
          <button
            onClick={scrollToBottom}
            className="absolute bottom-20 right-8 p-2 bg-gray-900 dark:bg-gray-700 text-white rounded-full shadow-lg hover:bg-gray-800 dark:hover:bg-gray-600 transition-all"
          >
            <ChevronDown className="w-5 h-5" />
          </button>
        )}

        {/* Input Form */}
        <div className="border-t bg-white dark:bg-gray-800 dark:border-gray-700 p-4">
          <form onSubmit={handleSubmit} className="flex gap-2">
            <input
              className="flex-1 p-2 border border-gray-300 dark:border-gray-600 rounded-lg 
                focus:outline-none focus:ring-2 focus:ring-blue-500 
                dark:bg-gray-700 dark:text-white dark:placeholder-gray-400"
              value={input}
              placeholder="Type your message..."
              onChange={handleInputChange}
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="p-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 
                disabled:opacity-50 disabled:cursor-not-allowed
                transition-colors duration-200"
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}