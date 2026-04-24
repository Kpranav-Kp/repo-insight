import js from "@eslint/js";
import globals from "globals";
import react from "eslint-plugin-react";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import jsxA11y from "eslint-plugin-jsx-a11y";
import importPlugin from "eslint-plugin-import";
import prettierConfig from "eslint-config-prettier";
import prettierPlugin from "eslint-plugin-prettier";

export default [
  {
    ignores: [
      "dist",
      "node_modules",
      "coverage",
      "build",
      "*.config.js",
      "eslint.config.js",
      "vite.config.js",
    ],
  },

  js.configs.recommended,

  {
    files: ["**/*.{js,jsx}"],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      globals: {
        ...globals.browser,
        ...globals.node,
      },
      parserOptions: {
        ecmaFeatures: { jsx: true },
      },
    },

    plugins: {
      react,
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
      "jsx-a11y": jsxA11y,
      import: importPlugin,
      prettier: prettierPlugin,
    },

    rules: {
      ...reactHooks.configs.recommended.rules,
      ...react.configs.recommended.rules,
      ...react.configs["jsx-runtime"].rules,
      ...jsxA11y.configs.recommended.rules,

      "react/prop-types": "off",
      "react-refresh/only-export-components": ["warn", { allowConstantExport: true }],

      "prettier/prettier": ["error", { semi: true }],

      "no-console": ["error", { allow: ["warn", "error"] }],
      eqeqeq: ["error", "always"],
      "prefer-const": "error",
      "no-var": "error",
      "object-shorthand": "error",
      "prefer-template": "error",
      "prefer-arrow-callback": "error",
      "no-nested-ternary": "warn",
      "no-unneeded-ternary": "error",
      "no-implied-eval": "error",
      "no-script-url": "error",
      "no-return-assign": ["error", "always"],

      "no-unused-vars": [
        "error",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
          caughtErrorsIgnorePattern: "^_",
        },
      ],

      "import/order": [
        "warn",
        {
          groups: ["builtin", "external", "internal", "parent", "sibling", "index"],
          "newlines-between": "always",
          alphabetize: { order: "asc", caseInsensitive: true },
        },
      ],
    },

    settings: {
      react: {
        version: "detect",
      },
      "import/resolver": {
        alias: {
          map: [["@", "./src"]],
          extensions: [".js", ".jsx"],
        },
      },
    },
  },

  prettierConfig,
];
