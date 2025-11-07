# CMU Data Science Club — Poker AI Agent

A Python-based agent developed for the **AI Poker Tournament** hosted by the **Carnegie Mellon University Data Science Club**.  
This project explores decision-making under uncertainty, game theory, and reinforcement learning in a competitive heads-up poker setting.

---

## 🧠 Overview
This agent was designed to play **heads-up no-limit poker** autonomously under tournament constraints.  
It evaluates hand strength, simulates possible outcomes, and adapts its strategy to maximize expected value over a 1000-hand match.

Originally developed by **Dev Sharma** in **2024**, and published to GitHub in **2025** for documentation and archival.

---

## 🎮 Tournament Details
| Parameter | Specification |
|-----------|---------------|
| **Format** | Heads-up (1v1) |
| **Hands per match** | 1000 |
| **Runtime limit** | 7 minutes per agent per match |
| **Resources** | 1 vCPU, 2 GB RAM |
| **Libraries** | Only approved Python libraries |
| **Scoring** | Winner determined by total bankroll |

---

## ⚙️ Technical Overview
- **Language:** Python  
- **Approach:** *(e.g., Monte Carlo simulation, heuristic evaluation, or reinforcement learning — replace with your method)*  
- **Core Components:**
  - `agent.py` – Main agent loop and interface
  - `strategy.py` – Policy and bet-sizing heuristics
  - `evaluator.py` – Hand strength or Monte Carlo evaluator
  - `match.py` – Local head-to-head match runner
  - `utils.py` – Helper functions

---

## ⚙️ Author
Dev Sharma
Developed for the CMU Data Science Club Poker AI Tournament (2024).
Published to GitHub in 2025.
