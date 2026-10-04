import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#eef7ff",
          100: "#d9edff",
          500: "#2f7de1",
          600: "#2563c9",
          700: "#1e4f9f",
        },
      },
    },
  },
  plugins: [],
};
export default config;
