// src/components/GAD7Questions.js
import React, { useState } from 'react';

const questions = [
  "Feeling nervous, anxious, or on edge",
  "Not being able to stop or control worrying",
  "Worrying too much about different things",
  "Trouble relaxing",
  "Being so restless that it is hard to sit still",
  "Becoming easily annoyed or irritable",
  "Feeling afraid, as if something awful might happen"
];

const options = [
  { label: "Not at all", value: 0 },
  { label: "Several days", value: 1 },
  { label: "More than half the days", value: 2 },
  { label: "Nearly every day", value: 3 }
];

function GAD7Questions() {
  const [answers, setAnswers] = useState(Array(questions.length).fill(0));
  const [totalScore, setTotalScore] = useState(null);

  const handleAnswerChange = (qIndex, value) => {
    const newAnswers = [...answers];
    newAnswers[qIndex] = parseInt(value);
    setAnswers(newAnswers);
  };

  const calculateScore = () => {
    const score = answers.reduce((total, current) => total + current, 0);
    setTotalScore(score);
  };

  const getInterpretation = (score) => {
    if (score === null) return "";
    if (score <= 4) return "Minimal anxiety";
    if (score <= 9) return "Mild anxiety";
    if (score <= 14) return "Moderate anxiety";
    return "Severe anxiety";
  };

  return (
    <div className="gad-7-form">
      <p>Over the last 2 weeks, how often have you been bothered by the following problems?</p>
      {questions.map((question, qIndex) => (
        <div key={qIndex} className="gad-question">
          <p><strong>{qIndex + 1}. {question}</strong></p>
          {options.map((option, oIndex) => (
            <label key={oIndex}>
              <input
                type="radio"
                name={`q${qIndex}`}
                value={option.value}
                checked={answers[qIndex] === option.value}
                onChange={() => handleAnswerChange(qIndex, option.value)}
              />
              {option.label}
            </label>
          ))}
        </div>
      ))}
      <button onClick={calculateScore} style={{ marginTop: '20px' }}>
        Calculate Score
      </button>
      
      {totalScore !== null && (
        <div className="gad-results">
          <h3>Your GAD-7 Score: {totalScore}</h3>
          <p>Interpretation: <strong>{getInterpretation(totalScore)}</strong></p>
        </div>
      )}
    </div>
  );
}

export default GAD7Questions;