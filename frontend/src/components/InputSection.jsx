import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FaRandom, FaCheckCircle } from 'react-icons/fa';

const InputSection = ({ onPredict, loading }) => {
  const [input, setInput] = useState('');
  const [error, setError] = useState('');
  const [isValid, setIsValid] = useState(false);

  const validateInput = (inputString) => {
    const trimmed = inputString.trim();
    
    if (!trimmed) {
      return { valid: false, error: 'Input cannot be empty' };
    }

    const values = trimmed.split(',').map(v => v.trim()).filter(v => v !== '');

    if (values.length !== 60) {
      return { 
        valid: false, 
        error: `Expected exactly 60 values, but got ${values.length}` 
      };
    }

    const numericValues = [];
    for (let i = 0; i < values.length; i++) {
      const num = parseFloat(values[i]);
      if (isNaN(num)) {
        return { 
          valid: false, 
          error: `Invalid value at position ${i + 1}: "${values[i]}" is not a number` 
        };
      }
      numericValues.push(num);
    }

    return { valid: true, values: numericValues };
  };

  useEffect(() => {
    const validation = validateInput(input);
    setIsValid(validation.valid);
  }, [input]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    const validation = validateInput(input);

    if (!validation.valid) {
      setError(validation.error);
      return;
    }

    try {
      await onPredict(validation.values);
      setInput('');
    } catch (err) {
      setError(err.message || 'Prediction failed');
    }
  };

  const loadSampleRock = () => {
    const sampleData = [
      0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,
      0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,
      0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,
      0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,
      0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,
      0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032
    ].join(',');
    setInput(sampleData);
    setError('');
  };

  const loadSampleMine = () => {
    const sampleData = [
      0.0094,0.0333,0.0306,0.0376,0.1296,0.1795,0.1909,0.1692,0.187,0.1725,0.2228,0.3106,0.4144,0.5157,0.5369,0.5107
      ,0.6441,0.7326,0.8164,0.8856,0.9891,1,0.875,0.8631,0.9074,0.8674,0.775,0.66,0.5615,0.4016,0.2331,0.1164,0.1095,0.0431,0.0619,0.1956,0.212
      ,0.3242,0.4102,0.2939,0.1911,0.1702,0.101,0.1512,0.1427,0.1097,0.1173,0.0972,0.0703,0.0281,0.0216,0.0153,0.0112,0.0241,0.0164,0.0055,0.0078,0.0055,0.0091,0.0067
    ].join(',');
    setInput(sampleData);
    setError('');
  };

  const loadRandomData = () => {
    const randomData = Array.from({ length: 60 }, () => 
      (Math.random() * 1).toFixed(4)
    ).join(',');
    setInput(randomData);
    setError('');
  };

  const currentCount = input.split(',').filter(v => v.trim() !== '').length;
  const progress = (currentCount / 60) * 100;

  return (
    <motion.div
      initial={{ opacity: 0, x: -50 }}
      animate={{ opacity: 1, x: 0 }}
      className="bg-white dark:bg-dark-card rounded-2xl shadow-lg p-6 card-shadow"
    >
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold text-gray-800 dark:text-white flex items-center">
          <span className="text-3xl mr-3 animate-bounce-slow">🔍</span>
          Enter Sonar Data
        </h2>
        {isValid && (
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            className="flex items-center text-green-500"
          >
            <FaCheckCircle className="text-2xl" />
          </motion.div>
        )}
      </div>

      {/* Instructions Card */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-4 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900 dark:to-blue-900 border-l-4 border-blue-500 p-4 rounded-lg"
      >
        <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
          <strong className="text-blue-600 dark:text-blue-400">📋 Instructions:</strong> Enter exactly{' '}
          <strong className="text-blue-600 dark:text-blue-400">60 numeric values</strong> separated by commas.
          Each value represents a sonar frequency reading (0.0 - 1.0).
        </p>
        
        {/* Sample Data Buttons */}
        <div className="flex flex-wrap gap-2">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={loadSampleRock}
            className="px-3 py-2 bg-blue-500 hover:bg-blue-600 text-white text-xs rounded-lg font-medium flex items-center gap-2 transition"
          >
            <span></span> Load Sample 1
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={loadSampleMine}
            className="px-3 py-2 bg-blue-500 hover:bg-blue-600 text-white text-xs rounded-lg font-medium flex items-center gap-2 transition"
          >
            <span></span> Load Sample 2
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={loadRandomData}
            className="px-3 py-2 bg-blue-500 hover:bg-blue-600 text-white text-xs rounded-lg font-medium flex items-center gap-2 transition"
          >
            <FaRandom /> Random Data
          </motion.button>
        </div>
      </motion.div>

      <form onSubmit={handleSubmit}>
        <div className="mb-4">
          <label htmlFor="input" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Sonar Readings
          </label>
          <textarea
            id="input"
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              setError('');
            }}
            className={`w-full h-40 px-4 py-3 border rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent font-mono text-sm transition-all dark:bg-dark-bg dark:text-gray-300 dark:border-dark-border ${
              error ? 'input-error border-red-500' : isValid ? 'border-blue-500' : 'border-gray-300'
            }`}
            placeholder="0.0200,0.0371,0.0428,0.0207,0.0954,..."
          />
          
          {/* Progress Bar */}
          <div className="mt-3">
            <div className="flex justify-between items-center mb-1">
              <p className={`text-xs font-medium ${
                currentCount === 60 ? 'text-blue-500 dark:text-blue-400' : 
                currentCount > 60 ? 'text-red-600 dark:text-red-400' : 
                'text-gray-500 dark:text-gray-400'
              }`}>
                Count: {currentCount} / 60 
                {currentCount === 60 && <span className="ml-2">✓ Perfect!</span>}
              </p>
              {currentCount > 0 && currentCount < 60 && (
                <p className="text-xs text-orange-600 dark:text-orange-400">
                  {60 - currentCount} more needed
                </p>
              )}
              {currentCount > 60 && (
                <p className="text-xs text-red-600 dark:text-red-400 animate-pulse">
                  Remove {currentCount - 60} values
                </p>
              )}
            </div>
            
            {/* Animated Progress Bar */}
            <div className="h-2 bg-gray-200 dark:bg-dark-border rounded-full overflow-hidden">
              <motion.div
                className={`h-full ${
                  currentCount === 60 ? 'bg-blue-500' :
                  currentCount > 60 ? 'bg-red-500' :
                  'bg-blue-500'
                }`}
                initial={{ width: 0 }}
                animate={{ width: `${Math.min(progress, 100)}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="mb-4 bg-red-50 dark:bg-red-900 dark:bg-opacity-30 border-l-4 border-red-500 p-4 rounded-lg animate-shake"
          >
            <p className="text-sm text-red-700 dark:text-red-400 flex items-center">
              <span className="text-xl mr-2">❌</span>
              <strong>Error:</strong>&nbsp;{error}
            </p>
          </motion.div>
        )}

        {/* Submit Button */}
        <motion.button
          whileHover={{ scale: loading || currentCount !== 60 ? 1 : 1.02 }}
          whileTap={{ scale: loading || currentCount !== 60 ? 1 : 0.98 }}
          type="submit"
          disabled={loading || currentCount !== 60}
          className={`w-full py-4 px-6 rounded-xl font-bold text-white text-lg transition-all btn-ripple ${
            loading || currentCount !== 60
              ? 'bg-blue-400 cursor-not-allowed'
            :'!bg-gradient-to-r !from-blue-500 !via-blue-600 !to-blue-800 hover:from-blue-700 hover:via-blue-800 hover:to-blue-700 shadow-lg hover:shadow-2xl'
          }`}
        >
          {loading ? (
            <span className="flex items-center justify-center gap-3">
              <svg className="animate-spin h-6 w-6" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              Analyzing Sonar Data...
            </span>
          ) : (
            <span className="flex items-center justify-center gap-3">
              <span className="text-3xl"></span>
              Predict: Rock or Mine?
            </span>
          )}
        </motion.button>
      </form>
    </motion.div>
  );
};

export default InputSection;
