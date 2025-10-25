// src/pages/MentalHealthCheckup.js
import React, { useState, useRef, useCallback } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios'; // <-- THIS LINE IS CRUCIAL
import GAD7Questions from '../components/GAD7Questions';

// Create an instance of axios to talk to our backend
const api = axios.create({
  baseURL: 'http://localhost:5000',
});

function MentalHealthCheckup() {
  const webcamRef = useRef(null);
  
  // States for results
  const [faceEmotion, setFaceEmotion] = useState(null);
  const [textEmotion, setTextEmotion] = useState(null);
  const [speechEmotion, setSpeechEmotion] = useState(null);
  
  // States for inputs
  const [textInput, setTextInput] = useState("");
  const [audioFile, setAudioFile] = useState(null);
  
  // States for loading/error messages
  const [faceLoading, setFaceLoading] = useState(false);
  const [textLoading, setTextLoading] = useState(false);
  const [speechLoading, setSpeechLoading] = useState(false);
  const [faceError, setFaceError] = useState('');
  const [textError, setTextError] = useState('');
  const [speechError, setSpeechError] = useState('');
  
  // States for chatbot
  const [chatMessages, setChatMessages] = useState([
    { sender: 'bot', text: 'Hello! I am your empathic AI assistant. Feel free to talk about your day.' }
  ]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);


  // --- 1. Facial Emotion ---
  const captureFace = useCallback(() => {
    setFaceLoading(true);
    setFaceError('');
    setFaceEmotion(null);

    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) {
      setFaceError("Could not capture image from webcam.");
      setFaceLoading(false);
      return;
    }
    
    api.post('/predict_face', { image: imageSrc })
      .then(response => {
        setFaceEmotion(response.data.emotion);
      })
      .catch(err => {
        setFaceError(err.response?.data?.error || "Error analyzing face.");
      })
      .finally(() => {
        setFaceLoading(false);
      });
  }, [webcamRef]);

  // --- 2. Text Emotion ---
  const analyzeText = () => {
    if (!textInput.trim()) {
      setTextError("Please type a message first.");
      return;
    }
    setTextLoading(true);
    setTextError('');
    setTextEmotion(null);
    
    api.post('/predict_text', { text: textInput })
      .then(response => {
        setTextEmotion(response.data.emotion);
      })
      .catch(err => {
        setTextError(err.response?.data?.error || "Error analyzing text.");
      })
      .finally(() => {
        setTextLoading(false);
      });
  };

  // --- 3. Speech Emotion ---
  const handleAudioUpload = () => {
    if (!audioFile) {
      setSpeechError("Please select an audio file first.");
      return;
    }
    setSpeechLoading(true);
    setSpeechError('');
    setSpeechEmotion(null);

    const formData = new FormData();
    formData.append('audio', audioFile);

    api.post('/predict_speech', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    .then(response => {
      setSpeechEmotion(response.data.emotion);
    })
    .catch(err => {
      setSpeechError(err.response?.data?.error || "Error analyzing speech.");
    })
    .finally(() => {
      setSpeechLoading(false);
    });
  };

  // --- 4. Chatbot ---
  const handleChatSubmit = async (e) => {
    e.preventDefault();
    const userMsg = chatInput;
    if (!userMsg.trim()) return;

    setChatMessages(prev => [...prev, { sender: 'user', text: userMsg }]);
    setChatInput("");
    setChatLoading(true);

    try {
      // This is the line (125) that was failing
      const response = await api.post('/chat', {
        message: userMsg,
        face_emotion: faceEmotion,
        text_emotion: textEmotion,
        speech_emotion: speechEmotion
      });
      setChatMessages(prev => [...prev, { sender: 'bot', text: response.data.reply }]);
    } catch (err) {
      setChatMessages(prev => [...prev, { sender: 'bot', text: 'Sorry, I had trouble connecting.' }]);
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <div>
      <h2>Mental Health Checkup</h2>
      
      {/* --- Section 1: Facial --- */}
      <section className="checkup-section">
        <h3>1. Facial Emotion Check</h3>
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          width={480}
          style={{ borderRadius: '8px' }}
        />
        <br />
        <button onClick={captureFace} disabled={faceLoading}>
          {faceLoading ? 'Analyzing...' : 'Analyze Face'}
        </button>
        {faceEmotion && <p>Detected Face Emotion: <strong>{faceEmotion}</strong></p>}
        {faceError && <p className="auth-error">{faceError}</p>}
      </section>
      
      {/* --- Section 2: Text --- */}
      <section className="checkup-section">
        <h3>2. Text Emotion Check</h3>
        <textarea
          rows="4"
          cols="50"
          style={{ width: '100%', padding: '10px', borderRadius: '6px', border: '1px solid #ccc' }}
          placeholder="How are you feeling right now? (e.g., 'I had a great day!' or 'I'm feeling very stressed.')"
          value={textInput}
          onChange={(e) => setTextInput(e.target.value)}
        />
        <br />
        <button onClick={analyzeText} disabled={textLoading} style={{ marginTop: '10px' }}>
          {textLoading ? 'Analyzing...' : 'Analyze Text'}
        </button>
        {textEmotion && <p>Detected Text Emotion: <strong>{textEmotion}</strong></p>}
        {textError && <p className="auth-error">{textError}</p>}
      </section>

      {/* --- Section 3: Speech --- */}
      <section className="checkup-section">
        <h3>3. Speech Emotion Check</h3>
        <p>Upload an audio file (.wav, .mp3) of you speaking.</p>
        <input 
          type="file" 
          accept="audio/*" 
          onChange={(e) => setAudioFile(e.target.files[0])} 
        />
        <button onClick={handleAudioUpload} disabled={speechLoading} style={{ marginLeft: '10px' }}>
          {speechLoading ? 'Analyzing...' : 'Analyze Speech'}
        </button>
        {speechEmotion && <p>Detected Speech Emotion: <strong>{speechEmotion}</strong></p>}
        {speechError && <p className="auth-error">{speechError}</p>}
      </section>
      
      {/* --- Section 4: GAD-7 --- */}
      <section className="checkup-section">
        <h3>4. GAD-7 Anxiety Questionnaire</h3>
        <GAD7Questions />
      </section>

      {/* --- Section 5: Empathic Chatbot --- */}
      <section className="checkup-section">
        <h3>5. Empathic AI Assistant</h3>
        <p>Now, let's talk. Your emotions detected above will be sent to the AI for a more empathic response.</p>
        <div className="chat-.window">
          {chatMessages.map((msg, index) => (
            <div key={index} className={`chat-message ${msg.sender}`}>
              {msg.text}
            </div>
          ))}
        </div>
        <form onSubmit={handleChatSubmit} className="chat-form">
          <input
            type="text"
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            placeholder="Talk to the assistant..."
            disabled={chatLoading}
          />
          <button type="submit" disabled={chatLoading}>
            {chatLoading ? '...' : 'Send'}
          </button>
        </form>
      </section>
    </div>
  );
}

export default MentalHealthCheckup;