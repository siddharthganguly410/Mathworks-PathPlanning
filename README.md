# ABU Robocon 2026 – "Kung Fu Quest"
## Robot 2 (R2) – Meihua Forest Q-Learning Solver

---

## 📁 Project Structure

```
robocon_r2/
│
├── robocon_r2_qlearning.py   ← Main RL implementation (train + evaluate)
├── requirements.txt           ← Python dependencies
├── r2_q_table.pkl             ← Pre-trained Q-table (auto-generated after training)
├── r2_summary.json            ← Training summary (auto-generated after training)
├── r2_training_results.png    ← Training plots (auto-generated after training)
└── README.md                  ← This file
```

---

## ⚙️ Setup in VS Code

### Step 1 — Prerequisites
Make sure you have **Python 3.10+** installed.
Check with:
```bash
python --version
# or
python3 --version
```

---

### Step 2 — Open the project folder in VS Code
```
File → Open Folder → select the robocon_r2 folder
```

---

### Step 3 — Create a virtual environment (recommended)

Open the **VS Code Terminal** (`Ctrl + `` ` `` ` or `Terminal → New Terminal`) and run:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 4 — Install dependencies
```bash
pip install -r requirements.txt
```

---

### Step 5 — Select Python Interpreter in VS Code
- Press `Ctrl + Shift + P`
- Type: `Python: Select Interpreter`
- Choose the one inside your `venv` folder (e.g. `./venv/Scripts/python.exe`)

---

## ▶️ Running the Project

### Option A — Run from Terminal
```bash
python robocon_r2_qlearning.py
```

### Option B — Run from VS Code
- Open `robocon_r2_qlearning.py`
- Click the **▶ Run** button (top-right) or press `F5`

---

## 🔧 Changing the Scroll Configuration

Open `robocon_r2_qlearning.py` and scroll to the bottom (`if __name__ == "__main__":`) to change inputs:

```python
r2_real_scrolls = [2, 5, 8, 11]   # ← change real scroll block numbers
fake_scroll     = 6                # ← change fake scroll block number
r1_scrolls      = [1, 3, 10]      # ← change R1 scroll block numbers
```

Rules:
- `r2_real_scrolls` must have exactly **4** blocks
- All block numbers must be between **1 and 12**
- No overlaps between real scrolls and fake scroll

---

## 📊 Output Files (auto-generated after running)

| File | Description |
|------|-------------|
| `r2_training_results.png` | Training reward, steps, success-rate plots + forest map + best path |
| `r2_q_table.pkl` | Serialised Q-table (can be reloaded to skip re-training) |
| `r2_summary.json` | JSON summary of training & evaluation metrics |

---

## 🗺️ Forest Grid Layout

```
 Block 1  |  Block 2  |  Block 3    ← Entry row  (R2 enters here)
 Block 4  |  Block 5  |  Block 6
 Block 7  |  Block 8  |  Block 9
 Block 10 |  Block 11 |  Block 12   ← Exit row   (R2 exits here)
```

---

## 🧠 Algorithm Summary

| Component | Detail |
|-----------|--------|
| Algorithm | Tabular Q-Learning |
| State | `(block, collected, remaining, fake_loc, r1_locs)` |
| Actions | UP, DOWN, LEFT, RIGHT, COLLECT, EXIT |
| Exploration | ε-greedy with exponential decay (1.0 → 0.01) |
| Learning rate | Harmonic schedule: `α / (1 + n/200)` per state-action |
| Init | Optimistic Q₀ = 10 (encourages early exploration) |
| Early stop | Stops when success rate ≥ 98% over 3,000 episodes |

---

## 📋 Reward Table

| Event | Reward |
|-------|--------|
| Collect real scroll | +100 |
| Collect all real scrolls | +500 |
| Successful exit | +300 |
| Move | −1 |
| Extra movement | −2 |
| Step onto fake scroll | −500 |
| Collect fake scroll | −1000 |
| Attempt collect R1 scroll | −1000 |
| Invalid move | −500 |
| Unnecessary wandering | −5 |

---

## ✅ Expected Output (default config)

```
Optimal path  →  Block 2 → (collect 2,5) → Block 5 → (collect 8) → Block 8 → (collect 11) → Block 11 → EXIT
Steps         →  7
Reward        →  1197.0
Success rate  →  100%
```
