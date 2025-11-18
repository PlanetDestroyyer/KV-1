"""
KV-1 User Profile System

Tracks user patterns, preferences, and behavior over time.
Used for personalization and proactive interventions.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List
import json


@dataclass
class UserProfile:
    """User behavior and preferences"""
    name: str = "User"
    energy_level: str = "focused"  # focused, curious, tired, excited
    github_checks_last_hour: int = 0
    last_meal_time: datetime = None
    typical_sleep_time: int = 23  # Hour (24h format)
    typical_wake_time: int = 7
    work_mode: bool = False
    focus_apps: List[str] = None  # Apps to allow during focus
    blocked_apps: List[str] = None  # Temporarily blocked apps

    def __post_init__(self):
        if self.focus_apps is None:
            self.focus_apps = []
        if self.blocked_apps is None:
            self.blocked_apps = []
        if self.last_meal_time is None:
            self.last_meal_time = datetime.now()

    def increment_github_checks(self) -> None:
        """Track GitHub obsession"""
        self.github_checks_last_hour += 1

    def reset_github_checks(self) -> None:
        """Reset hourly counter"""
        self.github_checks_last_hour = 0

    def record_meal(self) -> None:
        """User just ate"""
        self.last_meal_time = datetime.now()

    def hours_since_meal(self) -> float:
        """How long since last meal"""
        if self.last_meal_time is None:
            return 999.0
        delta = datetime.now() - self.last_meal_time
        return delta.total_seconds() / 3600

    def is_sleep_time(self) -> bool:
        """Should user be sleeping?"""
        hour = datetime.now().hour
        if self.typical_sleep_time < self.typical_wake_time:
            # Normal case: sleep 23:00 - 07:00
            return hour >= self.typical_sleep_time or hour < self.typical_wake_time
        else:
            # Rare case: sleep crosses midnight
            return hour >= self.typical_sleep_time and hour < self.typical_wake_time

    def is_late_night(self) -> bool:
        """Is it late night? (1-4 AM)"""
        hour = datetime.now().hour
        return 1 <= hour < 4

    def set_energy(self, level: str) -> None:
        """Update energy level: focused, curious, tired, excited"""
        valid = ["focused", "curious", "tired", "excited", "calm"]
        if level in valid:
            self.energy_level = level

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "energy_level": self.energy_level,
            "github_checks_last_hour": self.github_checks_last_hour,
            "last_meal_time": self.last_meal_time.isoformat() if self.last_meal_time else None,
            "typical_sleep_time": self.typical_sleep_time,
            "typical_wake_time": self.typical_wake_time,
            "work_mode": self.work_mode,
            "focus_apps": self.focus_apps,
            "blocked_apps": self.blocked_apps
        }

    @staticmethod
    def from_dict(data: dict) -> 'UserProfile':
        profile = UserProfile(
            name=data.get("name", "User"),
            energy_level=data.get("energy_level", "focused"),
            github_checks_last_hour=data.get("github_checks_last_hour", 0),
            typical_sleep_time=data.get("typical_sleep_time", 23),
            typical_wake_time=data.get("typical_wake_time", 7),
            work_mode=data.get("work_mode", False),
            focus_apps=data.get("focus_apps", []),
            blocked_apps=data.get("blocked_apps", [])
        )

        meal_time = data.get("last_meal_time")
        if meal_time:
            profile.last_meal_time = datetime.fromisoformat(meal_time)

        return profile


class UserProfileManager:
    """Manages user profile persistence"""

    def __init__(self):
        self.profile = UserProfile()

    def save(self, filepath: str) -> None:
        """Save profile to disk"""
        with open(filepath, 'w') as f:
            json.dump(self.profile.to_dict(), f, indent=2)

    def load(self, filepath: str) -> None:
        """Load profile from disk"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.profile = UserProfile.from_dict(data)
        except FileNotFoundError:
            pass  # Use default profile
