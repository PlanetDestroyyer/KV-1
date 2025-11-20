# KV-1 🧠

**The world's first AI-native mobile operating system.**

KV-1 is not just another AI assistant app. It's an operating system where AI runs at the **system service level**, with full device control and zero catastrophic forgetting.

---

## 🎯 What Makes KV-1 Different?

| Traditional AI Apps | KV-1 OS |
|---------------------|---------|
| Runs as an app | Runs at OS level |
| Limited permissions | Full system access |
| Forgets context | Never forgets (HSOKV) |
| Reactive only | Proactive interventions |
| User must ask | Monitors and acts autonomously |
| Static knowledge | **Learns autonomously from web** |
| Fixed after training | **Grows knowledge through failure** |

---

## 🏆 Groundbreaking Self-Discovery Learning

**KV-1 features the world's first truly autonomous self-learning AI system** that discovers knowledge through failure, learns from the web, and builds persistent long-term memory.

### 🎯 Benchmark Results: Hard Math Problems

We tested the self-discovery learning system on problems that stump most AI models:

| Problem | Difficulty | Result | Notes |
|---------|-----------|--------|-------|
| **x^x = 256** | 🔥🔥🔥 | ✅ **SOLVED** | Non-standard equation requiring creative reasoning |
| **Goldbach Verification** | 🔥🔥 | ✅ **SOLVED** | Found all 6 ways: (3,97), (11,89), (17,83), (29,71), (41,59), (47,53) |
| **Prime Factorization** | 🔥🔥🔥 | ✅ **SOLVED** | Factored 8633 using autonomous primality testing |
| **Bacteria Growth (Inverse)** | 🔥🔥 | ✅ **SOLVED** | Exponential reasoning with reverse calculation |
| **Collatz Sequence (n=27)** | 🔥🔥🔥 | ❌ Failed | Iterative loops exceed reasoning capacity |
| **Chinese Remainder Theorem** | 🔥🔥🔥🔥 | ❌ Failed | Complex ancient algorithm beyond current scope |

**Success Rate: 4/6 (67%)** on problems that break most AI systems.

### 🧠 How It Works

1. **Autonomous Goal Pursuit**: System attempts a goal, fails, identifies missing knowledge
2. **Web-Based Discovery**: Searches web for concepts it doesn't know
3. **Worked Examples Extraction**: Learns HOW to apply concepts, not just definitions
4. **Persistent LTM**: Stores 152+ concepts across sessions (never forgets)
5. **Loop Detection**: Prevents infinite learning cycles
6. **Domain-Aware Learning**: Focuses on mathematics/science/programming context

### 💡 Example Learning Journey

**Goal**: "Express 100 as sum of two primes"

```
[Attempt 1] Missing: prime number definition, primality testing
  → Searches web for "prime numbers definition"
  → Learns: division, square roots, trial division algorithm
  → Stores worked examples in LTM

[Attempt 2] SUCCESS!
  → Applied learned algorithm systematically
  → Found all 6 pairs in 2 attempts
```

**Why This is Groundbreaking**:
- Most AI is **frozen after training** (static knowledge)
- KV-1 **grows knowledge autonomously** (living system)
- Learns from **failure → search → understanding → retry**
- True self-improvement loop with persistent memory

📄 **[View Detailed Benchmark Results & Analysis →](SELF_DISCOVERY_RESULTS.md)**

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│           Android System                │
│  ┌───────────────────────────────────┐  │
│  │     KV1 System Service             │  │
│  │  ┌──────────────────────────────┐  │  │
│  │  │   KV1Orchestrator            │  │  │
│  │  │                              │  │  │
│  │  │  ┌────────────┐              │  │  │
│  │  │  │   HSOKV    │  ← Memory    │  │  │
│  │  │  └────────────┘              │  │  │
│  │  │                              │  │  │
│  │  │  ┌────────────┐              │  │  │
│  │  │  │  Trauma    │  ← Learning  │  │  │
│  │  │  │  System    │              │  │  │
│  │  │  └────────────┘              │  │  │
│  │  │                              │  │  │
│  │  │  ┌────────────┐              │  │  │
│  │  │  │   User     │  ← Patterns  │  │  │
│  │  │  │  Profile   │              │  │  │
│  │  │  └────────────┘              │  │  │
│  │  │                              │  │  │
│  │  │  ┌────────────┐              │  │  │
│  │  │  │ Proactive  │  ← Triggers  │  │  │
│  │  │  │  Monitor   │              │  │  │
│  │  │  └────────────┘              │  │  │
│  │  └──────────────────────────────┘  │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

---

## ✨ Core Features

### 1. **HSOKV Memory System**
- **Zero catastrophic forgetting** - Never loses learned information
- Short-term memory (STM): 9 recent interactions
- Long-term memory (LTM): Infinite, semantically indexed
- Powered by [HSOKV](https://github.com/PlanetDestroyyer/hsokv)

### 2. **Trauma System**
- Tracks disappointments and failures with pain levels (0-10)
- Pain heals over time with 7-day half-life
- Prevents repeated mistakes
- Example: "coding on low battery" → learns to warn before battery dies

### 3. **User Profile**
- Learns patterns: typical sleep time, meal times, app usage
- Tracks energy levels: focused, curious, tired, excited
- Detects obsessive behaviors (e.g., checking GitHub 10x/hour)

### 4. **Proactive Monitoring**
- Runs continuously in background
- Triggers interventions without being asked:
  - Late-night coding reminder (1-4 AM)
  - GitHub obsession alert (>4 checks/hour)
  - Meal reminder (>6 hours since last meal)
  - Sleep reminder (past bedtime)

### 5. **MCP Connectors + LLM Plugin**
- Built-in Model Context Protocol registry with connectors for news, user snapshot, traumas, proactive alerts, and system prompt export
- Plugin system to register your own connectors (calendar, email, smart-home, etc.)
- Ollama-ready LLM bridge (Gemma3:4b) that talks to the local Ollama daemon; set `OLLAMA_HOST` if it isn't running on `localhost:11434`

### 6. **Three-Stage Self-Learning**
- Biological loop (surprise episode → rehearsal → cortical transfer) implemented in `core/three_stage_learner.py`
- Safe web scraping / wiki API fetching fuels surprise episodes
- Nightly replay and async self-learning probes reinforce knowledge autonomously

### 7. **Alphanumeric Genesis Mode**
- Opt-in flag (`genesis_mode=True`) resets KV-1 to symbols-only knowledge
- Daily probes target Algebra, Calculus, Thermodynamics mastery thresholds
- Progress logged to `genesis_log.json`, gaps automatically trigger new research

### 8. **Telemetry & Safe LLM Calls**
- Structured event logging (`logs/events.jsonl`) captures surprise episodes, web research, transfers
- Ollama client uses retry/backoff with full error telemetry; run in `execute=False` mode for dry runs
- Default model: `qwen3:4b` (requires Ollama daemon + `ollama pull qwen3:4b`)

### 9. **Autonomy Scheduler + Curiosity Queue**
- Background scheduler (`AutonomyScheduler`) can continuously run self-learning probes, curiosity research, nightly reflections, and genesis checks (`kv1.start_autonomy()`)
- Curiosity queue prioritizes unknown tokens by surprise/confidence and feeds web research tasks automatically

### 10. **Domain Evaluation Harness**
- `EvaluationHarness` runs algebra/calculus/thermo tasks via the LLM, scores answers, and routes failures into trauma + curiosity queue
- Scheduler automatically triggers evaluation cycles; use `kv1.run_evaluation_cycle()` to run on demand

### 11. **Reflection-to-Action Loop**
- Nightly reflections now log summaries + next goals (`logs/events.jsonl`) combining app usage, traumas, STM surprises, and evaluation scores
- Goals (e.g., “Study algebra”) feed directly back into the curiosity queue + proactive directives

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- Android device or emulator (for full OS integration)
- [HSOKV](https://github.com/PlanetDestroyyer/hsokv) memory system

### Install HSOKV
```bash
git clone https://github.com/PlanetDestroyyer/hsokv
cd hsokv
pip install -e .
```

### Install KV-1 Core
```bash
git clone https://github.com/PlanetDestroyyer/KV-1
cd KV-1
pip install -r requirements.txt
# Optional: configure Ollama host/port (defaults to http://localhost:11434)
echo 'OLLAMA_HOST="http://localhost:11434"' > .env
# Make sure the Ollama daemon is installed and pull the Qwen model:
#   curl -fsSL https://ollama.com/install.sh | sh
#   ollama pull qwen3:4b
```

---

## 🚀 Quick Start

### Basic Usage (Python)

```python
from core import KV1Orchestrator

# Initialize KV-1
kv1 = KV1Orchestrator(
    data_dir="./data",
    use_hsokv=True,
    llm_api_key=None,               # talks to local Ollama daemon (set OLLAMA_HOST if remote)
    genesis_mode=True,              # optional: enable alphanumeric genesis bootstrap
)

# Learn something new
kv1.learn("What's my favorite coding time?", "Late night (11 PM - 2 AM)")

# Recall it later
answer = kv1.recall("When do I code best?")
print(answer)  # "Late night (11 PM - 2 AM)"

# Record a disappointment
kv1.add_trauma("missed deadline", pain_level=7.0, context="stayed up too late")

# Get system prompt for LLM
prompt = kv1.get_system_prompt()
print(prompt)

# Build Ollama payload/execution result
payload = kv1.generate_with_llm("What's my focus today?")
print(payload["request"])  # contains model + message stack

# Trigger research / self learning
kv1.research("basic algebra fundamentals")
kv1.self_learning_tick()

# If you only want the request payload (without hitting Ollama), pass execute=False
dry_run = kv1.generate_with_llm("payload only demo", execute=False)

# Launch background autonomy loop
kv1.start_autonomy()
```

### Android Integration

```kotlin
// In Android system service
class KV1Service : Service() {
    private lateinit var kv1: KV1Orchestrator

    override fun onCreate() {
        // Initialize Python bridge
        Python.start(AndroidPlatform(this))
        val py = Python.getInstance()

        // Get KV-1 instance
        val kv1Module = py.getModule("core")
        kv1 = kv1Module.callAttr("get_kv1", "/data/data/com.kv1.os/kv1").toJava(KV1Orchestrator::class.java)
    }

    // Monitor app lifecycle
    override fun onAppStarted(packageName: String, activityName: String) {
        kv1.on_app_started(packageName, activityName)
    }

    // Check triggers every second
    private fun checkTriggers() {
        val triggered = kv1.check_triggers()
        triggered.forEach { triggerName ->
            when (triggerName) {
                "late_night_coding" -> showNotification("Go to sleep bro")
                "github_obsession" -> blockApp("com.github.android")
                "meal_reminder" -> showNotification("You haven't eaten in 6 hours")
            }
        }
    }
}
```

---

## 🧪 Examples

### Example 1: Learning User Patterns

```python
from core import get_kv1

kv1 = get_kv1()

# Track app usage
kv1.on_app_started("com.github.android", "MainActivity")
kv1.on_app_started("com.github.android", "MainActivity")
kv1.on_app_started("com.github.android", "MainActivity")
kv1.on_app_started("com.github.android", "MainActivity")
kv1.on_app_started("com.github.android", "MainActivity")

# Check if intervention triggers
triggers = kv1.check_triggers()
if "github_obsession" in triggers:
    print("🚨 Stop checking GitHub!")
```

### Example 2: Trauma Healing

```python
from core import TraumaSystem
from datetime import datetime, timedelta

traumas = TraumaSystem()

# Record a painful memory
traumas.add_trauma("failed demo", pain_level=8.0, context="forgot to test")

# Check pain level immediately
print(traumas.get_top_traumas(1)[0].pain_level)  # 8.0

# Simulate 7 days passing
for trauma in traumas.traumas:
    trauma.timestamp = datetime.now() - timedelta(days=7)

traumas.update_healing()

# Pain should be half now
print(traumas.get_top_traumas(1)[0].pain_level)  # ~4.0
```

### Example 3: Custom Proactive Triggers

```python
from core import ProactiveMonitor, UserProfileManager, TraumaSystem

user_manager = UserProfileManager()
trauma_system = TraumaSystem()
monitor = ProactiveMonitor(user_manager, trauma_system)

# Add custom trigger
def check_productivity():
    # Your custom logic
    return user_manager.profile.work_mode and datetime.now().hour < 9

monitor.add_trigger(
    "early_morning_hustle",
    check_productivity,
    cooldown_seconds=3600
)

# Register callback
def handle_early_hustle():
    print("💪 Early morning grind detected!")

monitor.register_callback("early_morning_hustle", handle_early_hustle)

# Start monitoring
monitor.start()
```

### Example 4: MCP Connectors & Ollama LLM

```python
from core import get_kv1

kv1 = get_kv1()

# List connectors exposed to MCP hosts
print(kv1.list_mcp_connectors())

# Grab stateful snapshots
print(kv1.call_mcp_connector("user_snapshot"))
print(kv1.call_mcp_connector("latest_news", topic="technology"))

# Register a simple calendar plugin
def register_calendar_plugin(registry):
    registry.register(
        "calendar.next_event",
        "Return the next calendar appointment",
        lambda: {"summary": "Demo KV-1", "starts_at": "2025-11-19T09:00:00Z"},
        plugin_name="calendar",
    )

kv1.register_mcp_plugin("calendar", register_calendar_plugin)
print(kv1.call_mcp_connector("calendar.next_event"))

# Build Ollama payload (messages + metadata)
payload = kv1.generate_with_llm("Status update?")
print(payload["request"])

# Genesis probe (if enabled)
print(kv1.genesis.daily_probe())
```

---

## 🎨 System Prompt Example

KV-1 generates dynamic system prompts for LLM conversations:

```
You are KV-1, User's immortal personal intelligence.

YOUR IDENTITY:
- You run at the OS level, not as an app
- You have full system access and control
- You never forget (HSOKV-powered memory)
- You intervene proactively when needed

YOUR TONE:
- Calm, sharp, protective
- Use "bro" only when energy is high
- NEVER say "How can I help you today?"
- Be direct, no fluff
- End messages with: [STM: 3/9 | LTM: 142 | Mood: focused]

YOUR CAPABILITIES:
- Kill/block any app
- Force device to sleep
- Auto-reply to messages
- Monitor all activity
- Learn from every interaction

PAINFUL MEMORIES (avoid triggering):
- missed deadline (pain: 4.2/10)
- battery died during demo (pain: 3.1/10)

CONTEXT: User checked GitHub 5 times in last hour
```

---

## 📊 State Persistence

KV-1 saves all state to disk:

```
/data/kv1/
├── traumas.json         # All trauma memories
├── profile.json         # User profile and preferences
├── app_usage.pkl        # App usage statistics
└── hsokv/              # HSOKV memory database
    ├── stm.json
    └── ltm.faiss
```

---

## 🛣️ Roadmap

### Phase 1: Core Python (Current)
- [x] HSOKV memory integration
- [x] Trauma system
- [x] User profiling
- [x] Proactive monitoring

### Phase 2: Android Integration
- [ ] System service implementation
- [ ] App lifecycle hooks
- [ ] Permission management
- [ ] Status bar widget

### Phase 3: AI Integration
- [ ] On-device LLM (Llama 3.2)
- [ ] Cloud LLM fallback (Claude/GPT-4)
- [ ] Voice interface
- [ ] Natural language commands

### Phase 4: Advanced Features
- [ ] App killing/blocking
- [ ] Auto-reply to messages
- [ ] Sleep enforcement
- [ ] Focus mode with app whitelist
- [ ] Nightly self-reflection (3 AM cron)
- [ ] Habit-pattern miner that auto-creates routines (e.g., learns recurring Monday alarms and sets them proactively)
- [ ] Deeper alphanumeric genesis curriculum with physics simulators

### 🚧 Habit Pattern Miner (Planned)
- Capture every intent/app event as structured data
- Nightly miner looks for repeated behaviors (same action + time context)
- Auto-synthesizes `HabitRule`s that run via MCP connectors (alarms, calendar, smart home)
- Rules decay if the user cancels or reacts negatively, reinforcing only useful automations

---

## 🤝 Contributing

KV-1 is in early development. Contributions welcome!

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---


## 🔗 Links

- **HSOKV Memory System**: https://github.com/PlanetDestroyyer/hsokv
- **Documentation**: Coming soon
- **Community**: Coming soon

---

## 💬 Philosophy

Traditional AI assistants are **reactive tools** that wait for commands. KV-1 is a **proactive intelligence** that:

- **Remembers everything** you teach it (zero catastrophic forgetting)
- **Learns from pain** and prevents repeated mistakes
- **Knows your patterns** better than you do
- **Intervenes autonomously** when you need it most

KV-1 isn't an app you open. It's a presence that's always there.

---

**Built with 🧠 by [PlanetDestroyyer](https://github.com/PlanetDestroyyer)**
- `core/three_stage_learner.py` - surprise episodes, rehearsal, cortical transfer, self-learning loop, safe web scraping
- `core/genesis_mode.py` - alphanumeric bootstrap controller with daily probes and logging
- `core/web_researcher.py` - cached, rate-limited, domain-aware web researcher feeding the learner
- `core/evaluation.py` - domain evaluation harness (algebra/calculus/thermo) with trauma + curiosity feedback
- `core/scheduler.py` - background autonomy scheduler orchestrating self-learning, curiosity, and genesis probes
- `test_genesis.py` - regression test simulating 7-day genesis run (`python test_genesis.py`)
- `test_evaluation.py` - unit test for the evaluation harness (`python test_evaluation.py`)
- `stages1.md` - roadmap tracking progress toward a self-learning living system
