import React, { useState } from "react";
import { motion } from "framer-motion";
import api from "../services/api";
import toast from "react-hot-toast";
import { FaEnvelope, FaLock, FaEye, FaEyeSlash } from "react-icons/fa";

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
      console.log("REGISTER ERROR:", err.response); // 🔍 DEBUG

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
    <div className="min-h-screen flex items-center justify-center px-4">
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md bg-white dark:bg-gray-900 rounded-3xl shadow-2xl p-6 sm:p-8"
      >
        <h2 className="text-2xl sm:text-3xl font-extrabold text-center dark:text-white">
          Create Account
        </h2>
        <p className="text-center text-gray-500 dark:text-gray-400 mt-1">
          Save prediction history securely
        </p>

        <form onSubmit={handleRegister} className="mt-6 space-y-4">
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
              autoComplete="new-password"
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

          {/* Strength meter */}
          <div>
            <div className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full">
              <div
                className={`h-2 rounded-full transition-all ${
                  strength <= 25
                    ? "bg-red-500"
                    : strength <= 50
                    ? "bg-yellow-500"
                    : strength <= 75
                    ? "bg-blue-500"
                    : "bg-green-500"
                }`}
                style={{ width: `${strength}%` }}
              />
            </div>
            <p className="text-xs text-gray-500 mt-1">Password strength</p>
          </div>

          <button
            disabled={loading}
            className="w-full py-3 rounded-xl font-semibold text-white bg-indigo-600 hover:bg-indigo-700 transition disabled:opacity-70"
          >
            {loading ? "Creating account..." : "Register"}
          </button>
        </form>

        <p className="text-center text-sm text-gray-600 dark:text-gray-400 mt-6">
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
