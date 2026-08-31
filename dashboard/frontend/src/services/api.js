import axios from "axios";

const API_BASE_URL =
  import.meta.env.VITE_API_URL ||
  "https://wildlife-movement-analysis.up.railway.app";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

export const getHealth = async () => {
  const response = await api.get("/api/health");
  return response.data;
};

export const getDashboardSummary = async () => {
  const response = await api.get("/api/dashboard/summary");
  return response.data;
};

export default api;

