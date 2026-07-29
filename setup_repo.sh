#!/usr/bin/env bash
# Initialise the local git repository and push to GitHub.
# Run once, from inside the project folder.

set -e

# ---- 1. Set your git identity for THIS repository ----
read -rp "Your name (as it should appear on commits): " GIT_NAME
read -rp "Your GitHub email: " GIT_EMAIL
read -rp "Your GitHub username: " GH_USER
read -rp "Repository name [credit-fairness-research]: " REPO_NAME
REPO_NAME=${REPO_NAME:-credit-fairness-research}

git init
git config user.name  "$GIT_NAME"
git config user.email "$GIT_EMAIL"

# ---- 2. First commit ----
git add .
git commit -m "Initial commit: project structure, module skeletons and documentation"
git branch -M main

# ---- 3. Push ----
echo
echo "Now create an EMPTY repository named '$REPO_NAME' at:"
echo "   https://github.com/new"
echo "Do NOT initialise it with a README, .gitignore or licence."
echo
read -rp "Press Enter once the empty repository exists..."

git remote add origin "https://github.com/$GH_USER/$REPO_NAME.git"
git push -u origin main

echo
echo "Done. Repository: https://github.com/$GH_USER/$REPO_NAME"
