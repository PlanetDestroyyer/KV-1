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
- Gemini-ready LLM bridge that outputs the HTTP payload you can forward to Google's API (or swap out for your own provider)
- Configure with `GEMINI_API_KEY` or pass `llm_api_key` when constructing `KV1Orchestrator`

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
    llm_api_key="YOUR_GEMINI_KEY",  # or set GEMINI_API_KEY env var
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

# Build Gemini payload (forward this dict from your MCP / plugin host)
payload = kv1.generate_with_llm("What's my focus today?")
print(payload["endpoint"])
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

### Example 4: MCP Connectors & Gemini Plugin

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

# Build Gemini payload (forward request to Google APIs from host)
payload = kv1.generate_with_llm("Status update?")
print(payload["endpoint"])
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
