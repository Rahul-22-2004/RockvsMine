import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title, LineElement, PointElement } from 'chart.js';
import { Pie, Bar, Line } from 'react-chartjs-2';
import { FaChartPie, FaChartBar, FaChartLine, FaChartArea, FaExpand, FaSyncAlt, FaInfoCircle } from 'react-icons/fa';
import axios from 'axios';
import toast from 'react-hot-toast';


ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title, LineElement, PointElement);


const PredictionChart = ({ history }) => {
  const [chartType, setChartType] = useState('pie');
  const [rocImage, setRocImage] = useState(null);
  const [loadingROC, setLoadingROC] = useState(false);
  const [rocModelInfo, setRocModelInfo] = useState(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';  // Use same env var as in App.jsx

  const rockCount = history.filter(h => h.prediction === 'Rock').length;
  const mineCount = history.filter(h => h.prediction === 'Mine').length;


  // Fix useEffect with useCallback
  const fetchROCCurve = useCallback(async () => {
    try {
      setLoadingROC(true);
      const response = await axios.get(`${API_URL}/roc-curve`);
      setRocImage(response.data.roc_curve_image);
      setRocModelInfo({
        modelName: response.data.model_name,
        timestamp: new Date(response.data.timestamp).toLocaleString()
      });
      toast.success('ROC Curve loaded!', { icon: '📊', duration: 2000 });
    } catch (error) {
      console.error('Error fetching ROC curve:', error);
      toast.error('Failed to load ROC curve', { duration: 2000 });
    } finally {
      setLoadingROC(false);
    }
  }, [API_URL]);


  useEffect(() => {
    if (chartType === 'roc' && !rocImage) {
      fetchROCCurve();
    }
  }, [chartType, rocImage, fetchROCCurve]);


  // Pie Chart Data
  const pieData = {
    labels: ['🪨 Rock', '💣 Mine'],
    datasets: [
      {
        label: 'Predictions',
        data: [rockCount, mineCount],
        backgroundColor: [
          'rgba(59, 130, 246, 0.8)',
          'rgba(239, 68, 68, 0.8)'
        ],
        borderColor: [
          '#2563EB',
          '#DC2626'
        ],
        borderWidth: 3,
        hoverOffset: 10,
      },
    ],
  };


  // Bar Chart Data
  const barData = {
    labels: ['🪨 Rock', '💣 Mine'],
    datasets: [
      {
        label: 'Number of Predictions',
        data: [rockCount, mineCount],
        backgroundColor: [
          'rgba(59, 130, 246, 0.8)',
          'rgba(239, 68, 68, 0.8)'
        ],
        borderColor: [
          '#2563EB',
          '#DC2626'
        ],
        borderWidth: 2,
        borderRadius: 8,
      },
    ],
  };


  // Trend Line Chart Data (Confidence over time)
  const trendData = {
    labels: history.slice(-10).map((_, i) => `#${i + 1}`),
    datasets: [
      {
        label: 'Confidence %',
        data: history.slice(-10).map(h => h.confidence),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.4,
        fill: true,
        pointRadius: 5,
        pointHoverRadius: 7,
      },
    ],
  };


  const options = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          font: {
            size: 12,
            weight: 'bold'
          },
          padding: 15,
          usePointStyle: true,
        }
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        padding: 12,
        titleFont: {
          size: 14,
        },
        bodyFont: {
          size: 13,
        },
        cornerRadius: 8,
      }
    },
    animation: {
      animateScale: true,
      animateRotate: true,
    }
  };


  return (
    <motion.div
      initial={{ opacity: 0, x: 50 }}
      animate={{ opacity: 1, x: 0 }}
      className="bg-white dark:bg-dark-card rounded-2xl shadow-lg p-6 card-shadow h-full"
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-bold text-gray-800 dark:text-white flex items-center">
          <span className="text-2xl mr-2">📈</span>
          Statistics & Analysis
        </h3>


        {/* Chart Type Selector */}
        {history.length > 0 && (
          <div className="flex gap-2">
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={() => setChartType('pie')}
              className={`p-2 rounded-lg transition ${
                chartType === 'pie' 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-gray-200 dark:bg-dark-bg text-gray-600 dark:text-gray-400'
              }`}
              title="Pie Chart"
            >
              <FaChartPie />
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={() => setChartType('bar')}
              className={`p-2 rounded-lg transition ${
                chartType === 'bar' 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-gray-200 dark:bg-dark-bg text-gray-600 dark:text-gray-400'
              }`}
              title="Bar Chart"
            >
              <FaChartBar />
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={() => setChartType('trend')}
              className={`p-2 rounded-lg transition ${
                chartType === 'trend' 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-gray-200 dark:bg-dark-bg text-gray-600 dark:text-gray-400'
              }`}
              title="Trend Line"
            >
              <FaChartLine />
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={() => setChartType('roc')}
              className={`p-2 rounded-lg transition ${
                chartType === 'roc' 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-gray-200 dark:bg-dark-bg text-gray-600 dark:text-gray-400'
              }`}
              title="ROC Curve"
            >
              <FaChartArea />
            </motion.button>
          </div>
        )}
      </div>


      {history.length === 0 ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center py-12"
        >
          <motion.p
            animate={{ rotate: [0, 10, -10, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="text-7xl mb-4"
          >
            📊
          </motion.p>
          <p className="text-gray-500 dark:text-gray-400 text-sm">
            No data to display.<br/>Make some predictions first!
          </p>
        </motion.div>
      ) : (
        <div className="space-y-6">
          {/* Chart Display */}
          <motion.div
            key={chartType}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3 }}
            className={chartType === 'roc' ? 'min-h-[400px]' : 'h-64'}
          >
            {chartType === 'pie' && (
              <div className="h-full flex items-center justify-center">
                <div className="w-full max-w-[250px]">
                  <Pie data={pieData} options={options} />
                </div>
              </div>
            )}
            {chartType === 'bar' && (
              <Bar data={barData} options={options} />
            )}
            {chartType === 'trend' && (
              <Line data={trendData} options={options} />
            )}
            {chartType === 'roc' && (
              <AnimatePresence>
                {loadingROC ? (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="h-full flex flex-col items-center justify-center text-center"
                  >
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                      className="text-6xl mb-4"
                    >
                      🔄
                    </motion.div>
                    <p className="text-gray-600 dark:text-gray-400 font-medium">Loading ROC Curve...</p>
                  </motion.div>
                ) : rocImage ? (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="relative h-full flex flex-col items-center justify-center"
                  >
                    <div className="relative w-full max-w-md">
                      <img
                        src={rocImage}
                        alt="ROC Curve"
                        className="w-full h-auto rounded-lg shadow-xl border-2 border-blue-200 dark:border-blue-700"
                      />
                      <motion.button
                        whileHover={{ scale: 1.1 }}
                        whileTap={{ scale: 0.9 }}
                        onClick={() => setIsFullscreen(true)}
                        className="absolute top-2 right-2 bg-white/90 dark:bg-gray-800/90 p-2 rounded-full shadow-md hover:bg-white dark:hover:bg-gray-800 transition"
                        title="Fullscreen"
                      >
                        <FaExpand className="text-gray-600 dark:text-gray-300" />
                      </motion.button>
                    </div>

                    {rocModelInfo && (
                      <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="mt-4 text-center space-y-1 text-sm"
                      >
                        <div className="flex items-center justify-center gap-2 text-gray-600 dark:text-gray-400">
                          <FaInfoCircle />
                          <span>Model: {rocModelInfo.modelName}</span>
                        </div>
                        <p className="text-xs text-gray-500 dark:text-gray-500">
                          Generated: {rocModelInfo.timestamp}
                        </p>
                      </motion.div>
                    )}
                  </motion.div>
                ) : (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="h-full flex flex-col items-center justify-center text-center"
                  >
                    <motion.div
                      animate={{ rotate: [0, 10, -10, 0] }}
                      transition={{ duration: 2, repeat: Infinity }}
                      className="text-7xl mb-4"
                    >
                      📊
                    </motion.div>
                    <p className="text-gray-600 dark:text-gray-400 mb-4 font-medium text-center">
                      ROC curve unavailable<br />
                      <span className="text-sm text-gray-500">Model may not be trained yet</span>
                    </p>
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={fetchROCCurve}
                      className="px-6 py-3 bg-gradient-to-r from-blue-400 via-blue-500 to-blue-600 text-white rounded-lg font-semibold shadow-lg hover:shadow-xl transition-all flex items-center gap-2"
                    >
                      <FaSyncAlt />
                      Load ROC Curve
                    </motion.button>
                  </motion.div>
                )}
              </AnimatePresence>
            )}
          </motion.div>


          {/* Summary Stats Cards - Only show for non-ROC charts */}
          {chartType !== 'roc' && (
            <>
              <div className="grid grid-cols-3 gap-3 mb-8">
                <motion.div
                  whileHover={{ scale: 1.05, y: -5 }}
                  className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900 dark:to-blue-800 rounded-xl p-4 text-center shadow-md"
                >
                  <p className="text-3xl font-bold text-blue-600 dark:text-blue-300">{rockCount}</p>
                  <p className="text-xs text-gray-600 dark:text-gray-300 mt-1 font-medium">🪨 Rocks</p>
                </motion.div>
                
                <motion.div
                  whileHover={{ scale: 1.05, y: -5 }}
                  className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900 dark:to-blue-800 rounded-xl p-4 text-center shadow-md"
                >
                  <p className="text-3xl font-bold text-blue-600 dark:text-blue-300">{mineCount}</p>
                  <p className="text-xs text-gray-600 dark:text-gray-300 mt-1 font-medium">💣 Mines</p>
                </motion.div>
                
                <motion.div
                  whileHover={{ scale: 1.05, y: -5 }}
                  className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900 dark:to-blue-800 rounded-xl p-4 text-center shadow-md"
                >
                  <p className="text-3xl font-bold text-blue-600 dark:text-blue-300">{history.length}</p>
                  <p className="text-xs text-gray-600 dark:text-gray-300 mt-1 font-medium">📊 Total</p>
                </motion.div>
              </div>


              {/* Average Confidence */}
              <motion.div
                whileHover={{ scale: 1.02 }}
                className="bg-gradient-to-r from-blue-50 to-blue-150 dark:from-blue-900 dark:to-blue-800 rounded-xl p-4 border-2 border-blue-200 dark:border-blue-700"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-xs text-gray-600 dark:text-gray-300 mb-1 font-medium">Average Confidence</p>
                    <p className="text-3xl font-bold text-blue-600 dark:text-blue-300">
                      {(history.reduce((acc, h) => acc + h.confidence, 0) / history.length).toFixed(1)}%
                    </p>
                  </div>
                  <div className="text-5xl opacity-50">✨</div>
                </div>
              </motion.div>


              {/* Percentage Breakdown */}
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-600 dark:text-gray-400 flex items-center gap-2">
                    <span className="w-3 h-3 bg-blue-500 rounded-full"></span>
                    Rock Percentage
                  </span>
                  <span className="font-bold text-gray-800 dark:text-white">
                    {history.length > 0 ? ((rockCount / history.length) * 100).toFixed(1) : 0}%
                  </span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-600 dark:text-gray-400 flex items-center gap-2">
                    <span className="w-3 h-3 bg-red-500 rounded-full"></span>
                    Mine Percentage
                  </span>
                  <span className="font-bold text-gray-800 dark:text-white">
                    {history.length > 0 ? ((mineCount / history.length) * 100).toFixed(1) : 0}%
                  </span>
                </div>
              </div>


              {/* High Confidence Badge */}
              {history.length > 0 && (
                (() => {
                  const highConfidence = history.filter(h => h.confidence >= 90).length;
                  const percentage = (highConfidence / history.length) * 100;
                  return (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="bg-gradient-to-r from-blue-50 to-blue-100 dark:from-blue-800 dark:to-blue-900 rounded-xl p-3 border border-blue-200 dark:border-blue-700"
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-2xl">🎯</span>
                        <div className="flex-1">
                          <p className="text-xs text-gray-600 dark:text-gray-300">High Confidence Rate</p>
                          <p className="text-lg font-bold text-blue-600 dark:text-blue-300" >
                            {percentage.toFixed(0)}% ({highConfidence}/{history.length})
                          </p>
                        </div>
                      </div>
                    </motion.div>
                  );
                })()
              )}
            </>
          )}
        </div>
      )}

      {/* Fullscreen Modal */}
      <AnimatePresence>
        {isFullscreen && rocImage && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setIsFullscreen(false)}
            className="fixed inset-0 bg-black/95 backdrop-blur-sm z-50 flex items-center justify-center p-4 cursor-zoom-out"
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="relative max-w-4xl w-full cursor-default"
            >
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => setIsFullscreen(false)}
                className="absolute -top-14 right-0 bg-red-500 hover:bg-red-600 text-white px-6 py-3 rounded-lg font-semibold shadow-xl transition flex items-center gap-2"
              >
                ✕ Close
              </motion.button>
              <div className="bg-white dark:bg-gray-900 rounded-xl p-4 shadow-2xl">
                <img
                  src={rocImage}
                  alt="ROC Curve Fullscreen"
                  className="w-full h-auto rounded-lg"
                />
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};


export default PredictionChart;