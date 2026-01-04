import React, { useState } from "react";
import { motion } from "framer-motion";
import api from "../services/api";
import toast from "react-hot-toast";
import { FaEnvelope, FaLock, FaEye, FaEyeSlash } from "react-icons/fa";

const Login = ({ onLogin, switchToRegister }) => {
  const [email, setEmail] = useState(
    localStorage.getItem("remember_email") || ""
  );
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  // const [remember, setRemember] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleLogin = async (e) => {
    e.preventDefault();

    try {
      setLoading(true);
      const res = await api.post("/auth/login", { email, password });

      localStorage.setItem("access_token", res.data.access_token);

      // if (remember) {
      //   localStorage.setItem("remember_email", email);
      // } else {
      //   localStorage.removeItem("remember_email");
      // }

      toast.success("Welcome back!");
      onLogin();
    } catch (err) {
      console.log("LOGIN ERROR:", err.response); // 🔍 DEBUG

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
    <div className="min-h-screen flex items-center justify-center px-4">
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md bg-white dark:bg-gray-900 rounded-3xl shadow-2xl p-6 sm:p-8"
      >
        <h2 className="text-2xl sm:text-3xl font-extrabold text-center dark:text-white">
          Welcome Back
        </h2>
        <p className="text-center text-gray-500 dark:text-gray-400 mt-1">
          Login to continue
        </p>

        <form onSubmit={handleLogin} className="mt-6 space-y-4">
          {/* Email */}
          <div className="relative">
            <FaEnvelope className="absolute top-1/2 left-4 -translate-y-1/2 text-gray-400" />
            <input
              type="email"
              autoComplete="username"
              placeholder="Email address"
              className="w-full pl-11 pr-4 py-3 rounded-xl border bg-transparent dark:text-white dark:border-gray-700 focus:ring-2 focus:ring-indigo-500 outline-none"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>

          {/* Password */}
          <div className="relative">
            <FaLock className="absolute top-1/2 left-4 -translate-y-1/2 text-gray-400" />
            <input
              type={showPassword ? "text" : "password"}
              autoComplete="current-password"
              placeholder="Password (min 6 chars)"
              className="w-full pl-11 pr-12 py-3 rounded-xl border bg-transparent dark:text-white dark:border-gray-700 focus:ring-2 focus:ring-indigo-500 outline-none"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="absolute top-1/2 right-4 -translate-y-1/2 text-gray-500"
            >
              {showPassword ? <FaEyeSlash /> : <FaEye />}
            </button>
          </div>

          {/* Remember me */}
          {/* <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
            <input
              type="checkbox"
              checked={remember}
              onChange={(e) => setRemember(e.target.checked)}
            />
            Remember me
          </label> */}

          <button
            disabled={loading}
            className="w-full py-3 rounded-xl font-semibold text-white bg-indigo-600 hover:bg-indigo-700 transition disabled:opacity-70"
          >
            {loading ? "Logging in..." : "Login"}
          </button>
        </form>

        <p className="text-center text-sm text-gray-600 dark:text-gray-400 mt-6">
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
