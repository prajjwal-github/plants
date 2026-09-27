/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'agri-green': '#2e7d32',
        'agri-dark': '#1b5e20',
        'agri-light': '#a5d6a7',
      }
    },
  },
  plugins: [],
}
