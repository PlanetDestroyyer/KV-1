"""
KV-1 Proactive Monitoring System

Runs continuously checking for intervention triggers.
When a trigger fires, KV-1 takes action without being asked.
"""

from datetime import datetime
from typing import List, Callable, Dict
import threading
import time


class ProactiveTrigger:
    """A single trigger condition"""

    def __init__(self, name: str, check_fn: Callable, cooldown_seconds: int = 300):
        self.name = name
        self.check_fn = check_fn  # Function that returns True if should trigger
        self.cooldown_seconds = cooldown_seconds
        self.last_triggered = None

    def should_trigger(self) -> bool:
        """Check if trigger should fire (respecting cooldown)"""
        # Check cooldown
        if self.last_triggered:
            elapsed = (datetime.now() - self.last_triggered).total_seconds()
            if elapsed < self.cooldown_seconds:
                return False

        # Check condition
        if self.check_fn():
            self.last_triggered = datetime.now()
            return True

        return False


class ProactiveMonitor:
    """Monitors system state and triggers interventions"""

    def __init__(self, user_profile, trauma_system):
        self.user = user_profile
        self.traumas = trauma_system
        self.triggers: List[ProactiveTrigger] = []
        self.is_running = False
        self.monitor_thread = None
        self.check_interval = 1.0  # seconds

        # Callbacks for when triggers fire
        self.trigger_callbacks: Dict[str, Callable] = {}

        # Initialize default triggers
        self._init_default_triggers()

    def _init_default_triggers(self):
        """Set up default intervention triggers"""

        # Late night coding (1-4 AM)
        self.add_trigger(
            "late_night_coding",
            lambda: self.user.profile.is_late_night(),
            cooldown_seconds=3600  # Once per hour
        )

        # GitHub obsession (>4 checks in last hour)
        self.add_trigger(
            "github_obsession",
            lambda: self.user.profile.github_checks_last_hour > 4,
            cooldown_seconds=1800  # Once per 30 min
        )

        # Meal reminder (>6 hours since last meal)
        self.add_trigger(
            "meal_reminder",
            lambda: self.user.profile.hours_since_meal() > 6,
            cooldown_seconds=7200  # Once per 2 hours
        )

        # Sleep time (past typical sleep time + still active)
        self.add_trigger(
            "sleep_reminder",
            lambda: self.user.profile.is_sleep_time(),
            cooldown_seconds=3600
        )

    def add_trigger(self, name: str, check_fn: Callable, cooldown_seconds: int = 300):
        """Add a custom trigger"""
        trigger = ProactiveTrigger(name, check_fn, cooldown_seconds)
        self.triggers.append(trigger)

    def register_callback(self, trigger_name: str, callback: Callable):
        """Register a callback for when trigger fires"""
        self.trigger_callbacks[trigger_name] = callback

    def start(self):
        """Start monitoring in background thread"""
        if self.is_running:
            return

        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop monitoring"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Check all triggers
                for trigger in self.triggers:
                    if trigger.should_trigger():
                        self._handle_trigger(trigger.name)

                time.sleep(self.check_interval)

            except Exception as e:
                print(f"[ProactiveMonitor] Error: {e}")
                time.sleep(1.0)

    def _handle_trigger(self, trigger_name: str):
        """Handle a triggered intervention"""
        print(f"[KV-1] 🔔 Trigger fired: {trigger_name}")

        # Call registered callback if exists
        if trigger_name in self.trigger_callbacks:
            try:
                self.trigger_callbacks[trigger_name]()
            except Exception as e:
                print(f"[KV-1] Error in callback for {trigger_name}: {e}")

    def check_triggers_sync(self) -> List[str]:
        """
        Synchronously check all triggers (for Android service)
        Returns list of trigger names that should fire
        """
        fired = []
        for trigger in self.triggers:
            if trigger.should_trigger():
                fired.append(trigger.name)
        return fired
