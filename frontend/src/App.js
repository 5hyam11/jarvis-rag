import React, { useState, useRef, useEffect } from 'react';
import { 
  Send, Mic, MicOff, Volume2, VolumeX, Upload, Trash2,
  Bot, User, FileText, Loader2, Menu, X
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [selectedVoice, setSelectedVoice] = useState('nova');
  const [documents, setDocuments] = useState([]);
  const [useRag, setUseRag] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isConnected, setIsConnected] = useState(true);
  
  const messagesEndRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const fileInputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    checkHealth();
    fetchDocuments();
  }, []);

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_URL}/health`);
      setIsConnected(res.ok);
    } catch {
      setIsConnected(false);
    }
  };

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_URL}/documents`);
      const data = await res.json();
      setDocuments(data.documents || []);
    } catch (err) {
      console.error('Failed to fetch documents:', err);
    }
  };

  const sendMessage = async (text) => {
    if (!text.trim()) return;

    const userMessage = { role: 'user', content: text };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, use_rag: useRag, top_k: 5 }),
      });

      const data = await res.json();
      const assistantMessage = { 
        role: 'assistant', 
        content: data.answer,
        sources: data.sources || []
      };
      
      setMessages(prev => [...prev, assistantMessage]);

      if (voiceEnabled && data.answer) {
        playTTS(data.answer);
      }
    } catch (err) {
      console.error('Chat error:', err);
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error. Please check if the API is running.',
        error: true
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const playTTS = async (text) => {
    try {
      const res = await fetch(`${API_URL}/tts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, voice: selectedVoice }),
      });
      const audioBlob = await res.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      audio.play();
    } catch (err) {
      console.error('TTS error:', err);
    }
  };

  const transcribeAudio = async (audioBlob) => {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');
    try {
      const res = await fetch(`${API_URL}/transcribe`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      return data.text;
    } catch (err) {
      console.error('Transcription error:', err);
      return null;
    }
  };

  const uploadDocument = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    try {
      setIsLoading(true);
      const res = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData,
      });
      if (res.ok) {
        await fetchDocuments();
        return true;
      }
      return false;
    } catch (err) {
      console.error('Upload error:', err);
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        audioChunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        stream.getTracks().forEach(track => track.stop());
        
        setIsLoading(true);
        const text = await transcribeAudio(audioBlob);
        setIsLoading(false);

        if (text) {
          sendMessage(text);
        }
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (err) {
      console.error('Recording error:', err);
      alert('Could not access microphone. Please check permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    sendMessage(input);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (file) {
      const success = await uploadDocument(file);
      if (success) {
        alert(`${file.name} uploaded successfully!`);
      } else {
        alert('Upload failed. Please try again.');
      }
    }
    e.target.value = '';
  };

  return (
    <div className="app">
      <button className="mobile-menu-btn" onClick={() => setSidebarOpen(!sidebarOpen)}>
        {sidebarOpen ? <X size={24} /> : <Menu size={24} />}
      </button>

      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <Bot size={28} />
          <h1>Jarvis</h1>
        </div>

        <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
          <span className="status-dot"></span>
          {isConnected ? 'API Connected' : 'API Disconnected'}
        </div>

        <div className="sidebar-section">
          <h3><FileText size={16} /> Knowledge Base</h3>
          
          <button className="upload-btn" onClick={() => fileInputRef.current?.click()}>
            <Upload size={16} /> Upload Document
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.txt,.md,.docx"
            onChange={handleFileUpload}
            hidden
          />

          <div className="document-list">
            {documents.length === 0 ? (
              <p className="no-docs">No documents uploaded</p>
            ) : (
              documents.map((doc, i) => (
                <div key={i} className="document-item">
                  <FileText size={14} />
                  <span>{doc}</span>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="sidebar-section">
          <h3>Settings</h3>
          
          <label className="toggle-setting">
            <span>Use Knowledge Base</span>
            <input type="checkbox" checked={useRag} onChange={(e) => setUseRag(e.target.checked)} />
          </label>

          <label className="toggle-setting">
            <span>Voice Responses</span>
            <input type="checkbox" checked={voiceEnabled} onChange={(e) => setVoiceEnabled(e.target.checked)} />
          </label>

          {voiceEnabled && (
            <select value={selectedVoice} onChange={(e) => setSelectedVoice(e.target.value)} className="voice-select">
              <option value="alloy">Alloy</option>
              <option value="echo">Echo</option>
              <option value="fable">Fable</option>
              <option value="onyx">Onyx</option>
              <option value="nova">Nova</option>
              <option value="shimmer">Shimmer</option>
            </select>
          )}
        </div>

        <button className="clear-btn" onClick={() => setMessages([])}>
          <Trash2 size={16} /> Clear Chat
        </button>
      </aside>

      {sidebarOpen && <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)} />}

      <main className="chat-container">
        <div className="messages">
          {messages.length === 0 && (
            <div className="welcome">
              <Bot size={48} />
              <h2>Hello, I'm Jarvis</h2>
              <p>Your AI assistant with voice and knowledge base capabilities.</p>
              <p className="api-url">API: {API_URL}</p>
              <div className="suggestions">
                <button onClick={() => sendMessage("What can you help me with?")}>What can you help me with?</button>
                <button onClick={() => sendMessage("Tell me about yourself")}>Tell me about yourself</button>
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`message ${msg.role} ${msg.error ? 'error' : ''}`}>
              <div className="message-icon">
                {msg.role === 'user' ? <User size={20} /> : <Bot size={20} />}
              </div>
              <div className="message-content">
                <ReactMarkdown>{msg.content}</ReactMarkdown>
                {msg.sources && msg.sources.length > 0 && (
                  <div className="sources">
                    <span className="sources-label">Sources:</span>
                    {msg.sources.map((src, j) => (
                      <span key={j} className="source-tag">{src.file}</span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="message assistant loading">
              <div className="message-icon"><Bot size={20} /></div>
              <div className="message-content">
                <Loader2 className="spinner" size={20} />
                <span>Thinking...</span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <form className="input-area" onSubmit={handleSubmit}>
          <div className="input-wrapper">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type a message or click the mic..."
              rows={1}
              disabled={isLoading || isRecording}
            />
            
            <div className="input-actions">
              <button
                type="button"
                className={`voice-btn ${isRecording ? 'recording' : ''}`}
                onClick={isRecording ? stopRecording : startRecording}
                disabled={isLoading}
              >
                {isRecording ? <MicOff size={20} /> : <Mic size={20} />}
              </button>

              <button type="button" className="tts-toggle" onClick={() => setVoiceEnabled(!voiceEnabled)}>
                {voiceEnabled ? <Volume2 size={20} /> : <VolumeX size={20} />}
              </button>

              <button type="submit" className="send-btn" disabled={!input.trim() || isLoading}>
                <Send size={20} />
              </button>
            </div>
          </div>
          
          {isRecording && (
            <div className="recording-indicator">
              <span className="pulse"></span>
              Recording... Click mic to stop
            </div>
          )}
        </form>
      </main>
    </div>
  );
}

export default App;