import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FaTrash, FaDownload, FaEye, FaEyeSlash } from "react-icons/fa";
// import ConfidenceMeter from "./ConfidenceMeter";
import { exportToCSV } from "../utils/exportCSV";

const ResultHistory = ({ latestResult, history, onClearHistory }) => {
  const [showAllHistory, setShowAllHistory] = useState(false);
  const displayHistory = showAllHistory ? history : history.slice(-5);

  // ✅ FIX: Safe input preview (array or string)
  const formatInputPreview = (input) => {
    if (Array.isArray(input)) {
      return input.join(", ").substring(0, 40);
    }
    return String(input).substring(0, 40);
  };

  // ✅ FIX: Confidence formatter (always %)
  const formatConfidence = (value) => {
    if (typeof value === "number") {
      return `${Math.round(value)}%`;
    }
    return `${value}%`;
  };

  // ✅ FIX: Date formatter (IST)
  const formatISTDateTime = (timestamp) => {
    return new Date(timestamp).toLocaleString("en-IN", {
      timeZone: "Asia/Kolkata",
      day: "2-digit",
      month: "2-digit",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: true,
    });
  };

  return (
    <div className="space-y-6">
      {/* History Table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-dark-card rounded-2xl shadow-lg p-6 card-shadow"
      >
        <div className="flex flex-wrap justify-between items-center mb-4 gap-3">
          <h3 className="text-xl font-bold text-gray-800 dark:text-white flex items-center">
            <span className="text-2xl mr-2">📊</span>
            Prediction History ({history.length})
          </h3>

          <div className="flex gap-2">
            {history.length > 5 && (
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setShowAllHistory(!showAllHistory)}
                className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition text-sm font-medium flex items-center gap-2"
              >
                {showAllHistory ? <FaEyeSlash /> : <FaEye />}
                {showAllHistory ? "Show Less" : `Show All (${history.length})`}
              </motion.button>
            )}

            {history.length > 0 && (
              <>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => exportToCSV(history)}
                  className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition text-sm font-medium flex items-center gap-2"
                >
                  <FaDownload /> Export CSV
                </motion.button>

                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={onClearHistory}
                  className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition text-sm font-medium flex items-center gap-2"
                >
                  <FaTrash /> Clear
                </motion.button>
              </>
            )}
          </div>
        </div>

        {history.length === 0 ? (
          <div className="text-center py-16">
            <p className="text-gray-500 text-lg">No predictions yet.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className="border-b-2 border-gray-200 bg-gray-50">
                  <th className="p-3 text-sm font-semibold">#</th>
                  <th className="p-3 text-sm font-semibold">Input</th>
                  <th className="p-3 text-sm font-semibold">Prediction</th>
                  <th className="p-3 text-sm font-semibold">Confidence</th>
                  <th className="p-3 text-sm font-semibold">
                    Date / Time (IST)
                  </th>
                </tr>
              </thead>
              <tbody>
                <AnimatePresence>
                  {displayHistory
                    .slice()
                    .reverse()
                    .map((item, index) => (
                      <motion.tr
                        key={`${item.timestamp}-${index}`}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 20 }}
                        transition={{ delay: index * 0.05 }}
                        className="border-b hover:bg-gray-50 transition"
                      >
                        <td className="p-3 font-semibold">
                          {history.length - index}
                        </td>

                        <td className="p-3 text-xs font-mono max-w-xs truncate">
                          {formatInputPreview(item.input)}…
                        </td>

                        <td className="p-3">
                          {item.prediction === "Rock" ? "🪨 Rock" : "💣 Mine"}
                        </td>

                        <td className="p-3 font-medium">
                          {formatConfidence(item.confidence)}
                        </td>

                        <td className="p-3 text-xs">
                          {formatISTDateTime(item.timestamp)}
                        </td>
                      </motion.tr>
                    ))}
                </AnimatePresence>
              </tbody>
            </table>
          </div>
        )}
      </motion.div>
    </div>
  );
};

export default ResultHistory;
