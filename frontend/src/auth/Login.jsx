import React, { useState } from "react";
import { motion } from "framer-motion";
import api from "../services/api";
import toast from "react-hot-toast";
import { FaEnvelope, FaLock, FaEye, FaEyeSlash } from "react-icons/fa";
import { BsGraphUp, BsCpu } from "react-icons/bs";

const Login = ({ onLogin, switchToRegister }) => {
  const [email, setEmail] = useState(
    localStorage.getItem("remember_email") || ""
  );
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleLogin = async (e) => {
    e.preventDefault();

    try {
      setLoading(true);
      const res = await api.post("/auth/login", { email, password });

      localStorage.setItem("access_token", res.data.access_token);

      toast.success("Welcome back!");
      onLogin();
    } catch (err) {
      console.log("LOGIN ERROR:", err.response);

      const message =
        err.response?.data?.detail ??
        err.response?.data?.message ??
        "User does not exist or password is incorrect";

      toast.error(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative w-screen min-h-screen flex items-center justify-center overflow-hidden px-4 bg-gray-50 dark:bg-gray-950">

      {/* AI GRID BACKGROUND (FULL SCREEN + ANIMATED) */}
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

      {/* LOGIN CARD */}
      <motion.div
        initial={{ opacity: 0, y: 40, scale: 0.96 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.45 }}
        className="relative w-full max-w-md backdrop-blur-xl bg-white/70 dark:bg-gray-900/70 border border-white/30 dark:border-gray-700 rounded-3xl shadow-[0_20px_80px_rgba(0,0,0,0.25)] p-8"
      >
        <h2 className="text-3xl font-bold text-center text-gray-800 dark:text-white">
          Rock vs Mine
        </h2>

        <p className="text-center text-gray-500 dark:text-gray-400 mt-1">
          Login to access sonar prediction system
        </p>

        <form onSubmit={handleLogin} className="mt-7 space-y-5">

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
              autoComplete="current-password"
              placeholder="Password"
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

          {/* LOGIN BUTTON */}
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
            {loading ? "Logging in..." : "Login"}
          </motion.button>
        </form>

        {/* REGISTER */}
        <p className="text-center text-sm text-gray-600 dark:text-gray-400 mt-7">
          New user?{" "}
          <button
            onClick={switchToRegister}
            className="text-indigo-600 font-semibold hover:underline"
          >
            Register
          </button>
        </p>
      </motion.div>
    </div>
  );
};

export default Login;