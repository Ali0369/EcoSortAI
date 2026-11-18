/** Colors/typography sampled from the supplied Figma design. */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        cream: "#F6F7F2", station: "#EDF1E8", forest: "#173D33", deep: "#123C31", lime: "#D4EF82",
        muted: "#69776F", sage: "#EBF0E7", line: "#E0E6DD", leaf: "#D6E4BF", mist: "#CFDCC8", ink: "#354A3D",
        amber: { soft: "#FBEBD3", ink: "#9A6A1F", bar: "#C9A15C" },
      },
      fontFamily: { sans: ["Inter", "ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"] },
      borderRadius: { card: "20px", xl2: "28px" },
      boxShadow: { card: "0 1px 2px rgba(23,61,51,.04), 0 8px 24px rgba(23,61,51,.05)" },
    },
  },
  plugins: [],
};
