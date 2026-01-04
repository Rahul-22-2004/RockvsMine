import React from "react";
import { motion, AnimatePresence } from "framer-motion";

const LatestPrediction = ({ result }) => {
  if (!result) return null;

  const isMine = result.prediction === "Mine";

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: -20 }}
        className={`rounded-2xl shadow-2xl p-8 card-shadow ${
          isMine ? "result-mine" : "result-rock"
        }`}
      >
        {/* Header */}
        <h2 className="text-2xl font-bold mb-4 flex items-center">
          <span className="text-3xl mr-2">🎯</span>
          Latest Prediction
        </h2>

        {/* Content */}
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          {/* Left */}
          <div className="flex-1">
            <motion.p
              initial={{ scale: 0.8 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", stiffness: 200 }}
              className="text-6xl font-extrabold mb-4"
            >
              {isMine ? "💣 MINE" : "🪨 ROCK"}
            </motion.p>

            {/* Confidence */}
            <div className="mb-4">
              <p className="text-sm mb-1 opacity-90">Confidence Level</p>
              <div className="w-full bg-white bg-opacity-20 rounded-full h-3">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${result.confidence}%` }}
                  transition={{ duration: 0.8 }}
                  className="bg-white h-3 rounded-full"
                />
              </div>
              <p className="text-sm mt-1 font-semibold">
                {result.confidence}%
              </p>
            </div>

            {/* Timestamp */}
            <p className="text-sm opacity-90 flex items-center gap-2">
              <span>⏰</span>
              {new Date(result.timestamp).toLocaleString("en-IN", {
                timeZone: "Asia/Kolkata",
                hour12: true,
              })}
            </p>
          </div>

          {/* Right Emoji */}
          <motion.div
            animate={{ rotateY: [0, 360] }}
            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
            className="text-9xl opacity-20"
          >
            {isMine ? "💣" : "🪨"}
          </motion.div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
};

export default LatestPrediction;
