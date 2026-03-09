import React, { useState } from "react";
import { motion } from "framer-motion";
import api from "../services/api";
import toast from "react-hot-toast";
import { FaEnvelope, FaLock, FaEye, FaEyeSlash } from "react-icons/fa";
import { BsGraphUp, BsCpu } from "react-icons/bs";

const getStrength = (pwd) => {
  if (pwd.length < 6) return 25;
  if (pwd.length < 10) return 50;
  if (!/[A-Z]/.test(pwd) || !/[0-9]/.test(pwd)) return 75;
  return 100;
};

const Register = ({ onSuccess, switchToLogin }) => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);

  const strength = getStrength(password);

  const handleRegister = async (e) => {
    e.preventDefault();

    if (!email.includes("@")) {
      toast.error("Enter a valid email");
      return;
    }

    if (password.length < 6) {
      toast.error("Password must be at least 6 characters");
      return;
    }

    try {
      setLoading(true);
      await api.post("/auth/register", { email, password });

      toast.success("Account created! Please login.");
      onSuccess();
    } catch (err) {
      console.log("REGISTER ERROR:", err.response);

      const message =
        err.response?.data?.detail ??
        err.response?.data?.message ??
        "Registration failed";

      toast.error(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative w-screen min-h-screen flex items-center justify-center overflow-hidden px-4 bg-gray-50 dark:bg-gray-950">

      {/* AI GRID BACKGROUND */}
      <div className="absolute inset-0 w-screen h-screen opacity-30 pointer-events-none">
        <motion.div
          className="absolute inset-0 
          bg-[linear-gradient(to_right,#6366f1_1px,transparent_1px),
          linear-gradient(to_bottom,#6366f1_1px,transparent_1px)]
          bg-[size:40px_40px]"
          animate={{ backgroundPosition: ["0px 0px", "40px 40px"] }}
          transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
        />
      </div>

      {/* SONAR WAVES */}
      <motion.div
        animate={{ scale: [1, 1.6, 1], opacity: [0.4, 0.1, 0.4] }}
        transition={{ duration: 6, repeat: Infinity }}
        className="absolute w-[500px] h-[500px] rounded-full border border-indigo-400"
      />

      <motion.div
        animate={{ scale: [1, 2, 1], opacity: [0.3, 0.05, 0.3] }}
        transition={{ duration: 8, repeat: Infinity }}
        className="absolute w-[650px] h-[650px] rounded-full border border-blue-400"
      />

      {/* FLOATING ML ICONS */}
      <motion.div
        animate={{ y: [0, -25, 0] }}
        transition={{ duration: 6, repeat: Infinity }}
        className="absolute left-[10%] top-[20%] text-indigo-400 text-2xl"
      >
        <BsGraphUp />
      </motion.div>

      <motion.div
        animate={{ y: [0, 20, 0] }}
        transition={{ duration: 7, repeat: Infinity }}
        className="absolute right-[12%] top-[25%] text-blue-400 text-2xl"
      >
        <BsCpu />
      </motion.div>

      {/* REGISTER CARD */}
      <motion.div
        initial={{ opacity: 0, y: 40, scale: 0.96 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.45 }}
        className="relative w-full max-w-md backdrop-blur-xl bg-white/70 dark:bg-gray-900/70 border border-white/30 dark:border-gray-700 rounded-3xl shadow-[0_20px_80px_rgba(0,0,0,0.25)] p-8"
      >
        <h2 className="text-3xl font-bold text-center text-gray-800 dark:text-white">
          Create Account
        </h2>

        <p className="text-center text-gray-500 dark:text-gray-400 mt-1">
          Save prediction history securely
        </p>

        <form onSubmit={handleRegister} className="mt-7 space-y-5">

          {/* EMAIL */}
          <div className="relative group">
            <FaEnvelope className="absolute top-1/2 left-4 -translate-y-1/2 text-gray-400 group-focus-within:text-indigo-500 transition" />

            <input
              type="email"
              autoComplete="username"
              placeholder="Email address"
              className="w-full pl-11 pr-4 py-3 rounded-xl border border-gray-300 bg-white/80 dark:bg-gray-800 dark:border-gray-700 dark:text-white focus:ring-2 focus:ring-indigo-500 outline-none transition"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>

          {/* PASSWORD */}
          <div className="relative group">
            <FaLock className="absolute top-1/2 left-4 -translate-y-1/2 text-gray-400 group-focus-within:text-indigo-500 transition" />

            <input
              type={showPassword ? "text" : "password"}
              autoComplete="new-password"
              placeholder="Password (min 6 chars)"
              className="w-full pl-11 pr-12 py-3 rounded-xl border border-gray-300 bg-white/80 dark:bg-gray-800 dark:border-gray-700 dark:text-white focus:ring-2 focus:ring-indigo-500 outline-none transition"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />

            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="absolute top-1/2 right-4 -translate-y-1/2 text-gray-500 hover:text-indigo-600 transition"
            >
              {showPassword ? <FaEyeSlash /> : <FaEye />}
            </button>
          </div>

          {/* PASSWORD STRENGTH */}
          <div>
            <div className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full">
              <motion.div
                className={`h-2 rounded-full ${
                  strength <= 25
                    ? "bg-red-500"
                    : strength <= 50
                    ? "bg-yellow-500"
                    : strength <= 75
                    ? "bg-blue-500"
                    : "bg-green-500"
                }`}
                animate={{ width: `${strength}%` }}
                transition={{ duration: 0.4 }}
              />
            </div>

            <p className="text-xs text-gray-500 mt-1">
              Password strength
            </p>
          </div>

          {/* REGISTER BUTTON */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.96 }}
            disabled={loading}
            className="w-full py-3 rounded-xl font-semibold text-white
            bg-gradient-to-r from-indigo-600 via-blue-600 to-indigo-600
            hover:from-indigo-700 hover:via-blue-700 hover:to-indigo-700
            shadow-lg shadow-indigo-500/30
            transition disabled:opacity-70"
          >
            {loading ? "Creating account..." : "Register"}
          </motion.button>
        </form>

        {/* LOGIN */}
        <p className="text-center text-sm text-gray-600 dark:text-gray-400 mt-7">
          Already have an account?{" "}
          <button
            onClick={switchToLogin}
            className="text-indigo-600 font-semibold hover:underline"
          >
            Login
          </button>
        </p>
      </motion.div>
    </div>
  );
};

export default Register;