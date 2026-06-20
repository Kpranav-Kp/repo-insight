import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "@fontsource/abril-fatface";
import "@fontsource/cormorant-garamond/400-italic.css";
import "./index.css";
import App from "./App.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
