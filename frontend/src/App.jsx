import React, { useState, useEffect, useCallback } from "react";
// import axios from "axios";
import toast from "react-hot-toast";
import { motion, AnimatePresence } from "framer-motion";
import { Toaster } from "react-hot-toast";

import InputSection from "./components/InputSection";
import ResultHistory from "./components/ResultHistory";
import PredictionChart from "./components/PredictionChart";
import ContactForm from "./components/ContactForm";
import DarkModeToggle from "./components/DarkModeToggle";
import StatsCard from "./components/StatsCard";
import ToastNotification from "./components/ToastNotification";
import useDarkMode from "./hooks/useDarkMode";
import LatestPrediction from "./components/LatestPrediction";

import { fireConfetti } from "./utils/confetti";
import { FaTrophy } from "react-icons/fa";

import Login from "./auth/Login";
import Register from "./auth/Register";
import api from "./services/api";

const App = () => {
  const [latestResult, setLatestResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [isDark, setIsDark] = useDarkMode();

  // Show animation every time app loads (3 seconds)
  const [showWelcome, setShowWelcome] = useState(true);

  const [stats, setStats] = useState({
    totalPredictions: 0,
    avgConfidence: 0,
    rockCount: 0,
    mineCount: 0,
    accuracy: 0,
  });

  const [isAuthenticated, setIsAuthenticated] = useState(
    !!localStorage.getItem("access_token")
  );

  const [showRegister, setShowRegister] = useState(false);

  const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

  const CONTACT_FORM_ACTION = process.env.REACT_APP_CONTACT_FORM_ACTION || "";

  // -------------------------------------
  // CALCULATE STATS
  // -------------------------------------
  const calculateStats = useCallback(() => {
    if (history.length === 0) {
      setStats({
        totalPredictions: 0,
        avgConfidence: 0,
        rockCount: 0,
        mineCount: 0,
        accuracy: 0,
      });
      return;
    }

    const rockCount = history.filter((h) => h.prediction === "Rock").length;
    const mineCount = history.filter((h) => h.prediction === "Mine").length;
    const avgConfidence =
      history.reduce((acc, h) => acc + h.confidence, 0) / history.length;

    const highConfidence = history.filter((h) => h.confidence >= 80).length;
    const accuracy = (highConfidence / history.length) * 100;

    setStats({
      totalPredictions: history.length,
      avgConfidence: Number(avgConfidence).toFixed(1),
      rockCount,
      mineCount,
      accuracy: Number(accuracy).toFixed(1),
    });
  }, [history]);

  // -------------------------------------
  // LOAD HISTORY + ALWAYS show welcome animation (5s)
  // -------------------------------------

  useEffect(() => {
    const loadHistory = async () => {
      try {
        const res = await api.get("/history");
        setHistory(res.data.history || []);
      } catch (err) {
        console.error("Failed to load history", err);
      }
    };

    if (isAuthenticated) {
      loadHistory();
    }
  }, [isAuthenticated]);

  useEffect(() => {
    setShowWelcome(true);
    const timer = setTimeout(() => setShowWelcome(false), 3000);
    return () => clearTimeout(timer);
  }, []);

  // -------------------------------------
  // SAVE HISTORY + UPDATE STATS
  // -------------------------------------
  useEffect(() => {
    calculateStats();
  }, [history, calculateStats]);

  // -------------------------------------
  // HANDLE PREDICTION
  // -------------------------------------
  const handlePredict = async (features) => {
    setLoading(true);
    const loadingToast = toast.loading("🔍 Analyzing sonar data...");

    try {
      // const response = await axios.post(`${API_URL}/predict`, {
      //   values: features,
      // });
      const response = await api.post("/predict", {
        values: features,
      });

      const result = {
        prediction: response.data.prediction,
        confidence: response.data.confidence,
        timestamp: response.data.timestamp,
        input: features.join(","),
      };

      setLatestResult(result);
      setHistory((prevHistory) => [...prevHistory, result]);

      toast.success(
        `${result.prediction === "Rock" ? "🪨" : "💣"} Prediction: ${
          result.prediction
        }!`,
        { id: loadingToast }
      );

      if (result.confidence >= 90) fireConfetti();

      return result;
    } catch (error) {
      console.error("Prediction error:", error);

      toast.error(
        error.response?.data?.detail ||
          error.response?.data?.error ||
          "Failed to get prediction. Is the backend running?",
        { id: loadingToast }
      );

      throw new Error(
        error.response?.data?.detail ||
          error.response?.data?.error ||
          "Failed to get prediction from server"
      );
    } finally {
      setLoading(false);
    }
  };

  // -------------------------------------
  // CLEAR HISTORY
  // -------------------------------------
  const handleClearHistory = async () => {
    if (window.confirm("Clear all prediction history?")) {
      try {
        await api.delete("/clear-history");
        setHistory([]);
        setLatestResult(null);
        toast.success("History cleared!");
      } catch {
        toast.error("Failed to clear history");
      }
    }
  };

  // -------------------------------------
  // KEYBOARD SHORTCUTS - FIXED
  // -------------------------------------
  useEffect(() => {
    const handleKeyPress = (e) => {
      // normalize key
      const key = (e.key || "").toLowerCase();

      // don't react when typing in inputs/textareas/selects or contenteditable
      const target = e.target;
      const tag =
        target && target.tagName ? target.tagName.toLowerCase() : null;
      const isEditing =
        tag === "input" ||
        tag === "textarea" ||
        tag === "select" ||
        target?.isContentEditable;

      if (isEditing) return; // ignore shortcuts while typing

      // Ctrl/Cmd + D => Toggle Dark Mode
      if (
        (e.ctrlKey || e.metaKey) &&
        e.key.toLowerCase() === "d" &&
        !e.repeat
      ) {
        e.preventDefault();

        setIsDark((prev) => {
          const nextIsDark = !prev;

          toast.success(
            `${nextIsDark ? "🌙 Dark" : "☀️ Light"} mode activated`
          );

          return nextIsDark;
        });

        return;
      }

      // Ctrl/Cmd + K => Clear History
      if ((e.ctrlKey || e.metaKey) && key === "k") {
        e.preventDefault();
        handleClearHistory();
        return;
      }
    };

    window.addEventListener("keydown", handleKeyPress);
    return () => window.removeEventListener("keydown", handleKeyPress);
    // no deps: handler uses functional setIsDark and functions declared above
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // -------------------------------------
  // Handle logout
  // -------------------------------------
  const handleLogout = () => {
    localStorage.removeItem("access_token");
    setIsAuthenticated(false);
    setHistory([]);
    setLatestResult(null);
    toast.success("Logged out successfully");
  };

  // -------------------------------------
  // AUTH GATE
  // -------------------------------------

  if (!isAuthenticated) {
    return (
      <>
        {/* ✅ ALWAYS MOUNTED */}
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: { fontSize: "14px" },
          }}
        />

        {/* AUTH GATE */}
        <div className="min-h-screen flex items-center justify-center">
          {showRegister ? (
            <Register
              onSuccess={() => setShowRegister(false)}
              switchToLogin={() => setShowRegister(false)}
            />
          ) : (
            <Login
              onLogin={() => setIsAuthenticated(true)}
              switchToRegister={() => setShowRegister(true)}
            />
          )}
        </div>
      </>
    );
  }

  // -------------------------------------
  // UI
  // -------------------------------------

  return (
      <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8 page-transition">
        <ToastNotification />
        <DarkModeToggle
          isDark={isDark}
          toggleDark={() => setIsDark((prev) => !prev)}
        />

        {isAuthenticated && (
          <button
            onClick={handleLogout}
            className="absolute top-6 right-20 px-4 py-2 rounded-lg
               bg-red-500 hover:bg-red-600 text-white text-sm font-semibold
               transition"
          >
            Logout
          </button>
        )}

        {/* WELCOME ANIMATION — ALWAYS SHOW (5s) */}
        <AnimatePresence>
          {showWelcome && (
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 backdrop-blur-sm"
              onClick={() => setShowWelcome(false)}
            >
              <motion.div
                className="bg-white dark:bg-dark-card rounded-2xl p-12 text-center shadow-2xl max-w-md"
                initial={{ y: -100 }}
                animate={{ y: 0 }}
                transition={{ type: "spring", stiffness: 300 }}
              >
                <motion.div
                  animate={{ rotate: [0, 360] }}
                  transition={{
                    duration: 3,
                    repeat: Infinity,
                    ease: "linear",
                  }}
                  className="text-8xl mb-4"
                >
                  🪨💣
                </motion.div>

                <h2 className="text-3xl font-bold mb-2 text-black dark:text-white">
                  Welcome!
                </h2>

                <p className="text-gray-600 dark:text-gray-400 mb-4">
                  Advanced ML-powered Sonar Classification
                </p>

                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => setShowWelcome(false)}
                  className="px-6 py-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-full font-semibold"
                >
                  Get Started 🚀
                </motion.button>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="max-w-7xl mx-auto">
          {/* HEADER */}
          <motion.div
            initial={{ opacity: 0, y: -50 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-12"
          >
            <motion.h1
              className="text-4xl sm:text-5xl md:text-6xl font-extrabold text-white mb-4 drop-shadow-lg"
              animate={{ scale: [1, 1.02, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <span className="dark:text-red-400  text-black">
                🪨 Rock vs Mine 💣
              </span>
            </motion.h1>

            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="text-lg sm:text-xl dark:text-white  text-black opacity-90 mb-4"
            >
              Advanced ML-powered Sonar Data Classification
            </motion.p>

            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.5 }}
              className="inline-flex items-center px-6 py-3 glass rounded-full text-black dark:text-white text-sm font-semibold"
            >
              <span className="w-3 h-3 bg-green-500 rounded-full mr-3 animate-pulse"></span>
              Backend Connected: {API_URL}
            </motion.div>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.7 }}
              className="mt-4 text-xs dark:text-white text-black opacity-60"
            >
              💡 Shortcuts:
              <kbd className="px-2 py-1 bg-white bg-opacity-20 rounded">
                Ctrl+D
              </kbd>{" "}
              Dark Mode |
              <kbd className="px-2 py-1 bg-white bg-opacity-20 rounded ml-2">
                Ctrl+K
              </kbd>{" "}
              Clear History
            </motion.div>
          </motion.div>

          {/* STATS */}
          {history.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8 "
            >
              <StatsCard
                icon="🎯"
                title="Total Predictions"
                value={stats.totalPredictions}
                color="from-blue-500 to-blue-700"
                delay={0.1}
              />
              <StatsCard
                icon="📊"
                title="Avg Confidence"
                value={`${stats.avgConfidence}%`}
                color="from-blue-500 to-blue-700"
                delay={0.2}
              />
              <StatsCard
                icon="🪨"
                title="Rocks Found"
                value={stats.rockCount}
                color="from-blue-500 to-blue-700"
                delay={0.3}
              />
              <StatsCard
                icon="💣"
                title="Mines Found"
                value={stats.mineCount}
                color="from-blue-500 to-blue-700"
                delay={0.4}
              />
            </motion.div>
          )}

          {/* ACHIEVEMENT */}
          {history.length >= 10 && (
            <motion.div
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              className="mb-8 text-center"
            >
              <div className="inline-flex items-center gap-3 px-6 py-3 bg-gradient-to-r from-yellow-400 to-orange-500 rounded-full text-white font-bold shadow-lg badge-glow">
                <FaTrophy className="text-2xl" />
                <span>
                  🏆 Achievement Unlocked:{" "}
                  {history.length >= 50 ? "Master Analyzer" : "Explorer"}
                </span>
              </div>
            </motion.div>
          )}

          {/* INPUT + CONTACT */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
            <div className="lg:col-span-2">
              <InputSection onPredict={handlePredict} loading={loading} />
            </div>
            <div>
              <ContactForm formAction={CONTACT_FORM_ACTION} />
            </div>
          </div>

          {/* RESULTS + CHART */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2 space-y-8">
              {/* ✅ Latest Prediction Card */}
              <LatestPrediction result={latestResult} />

              <ResultHistory
                latestResult={latestResult}
                history={history}
                onClearHistory={handleClearHistory}
              />
            </div>
            <div>
              <PredictionChart history={history} />
            </div>
          </div>

          {/* FOOTER */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1 }}
            className="mt-12 text-center text-white opacity-80"
          >
            <p className="text-sm mb-2">
              ⚡ Powered by Logistic Regression ML Model + Random Forest Model |
              Data from Google Cloud Storage
            </p>
            {/* <p className=" text-xs">
            FastAPI Backend + React Frontend | Builded BY RMJ developers |{" "}
            {new Date().getFullYear()}
          </p> */}
            <div className="mt-4 flex justify-center gap-4 text-xs">
              <a href="http" className="hover:text-blue-300 transition">
                About
              </a>
              <span>•</span>
              <a href="http" className="hover:text-blue-300 transition">
                GitHub
              </a>
              <span>•</span>
              <a href="http" className="hover:text-blue-300 transition">
                Documentation
              </a>
            </div>
          </motion.div>
        </div>
      </div>
  );
};

export default App;
