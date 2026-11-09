/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        panel: '#1b2a33',
        panel2: '#16232b',
        edge: '#30414b',
        accent: '#81d2bd',
        ink: '#edf5f3',
        muted: '#b1c0c3',
      },
    },
  },
  plugins: [],
}