from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from flask import Flask, redirect, render_template, request, send_file, session, url_for
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

from .agent import QLearningAgent, SARSAAgent
from .config import StudyPlannerConfig
from .db import (
    authenticate_user,
    create_notification,
    create_user,
    ensure_notification_prefs,
    get_unread_notification_count,
    get_user_analytics,
    get_user_by_id,
    get_user_plan,
    init_db,
    list_notifications,
    list_user_plans,
    mark_notification_read,
    record_progress_event,
    save_plan,
    update_notification_prefs,
)
from .environment import StudyPlannerEnvironment
from .openenv_env import OpenEnvStudyPlanner, state_to_dict, step_to_dict
from .persistence import load_q_table, save_q_table
from .trainer import (
    generate_greedy_schedule,
    generate_greedy_schedule_with_explanations,
    generate_random_schedule,
    generate_sarsa_greedy_schedule,
    train_agent,
    train_sarsa_agent,
)
from .visualization import plot_rewards_as_base64


TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "en": {
        "app_title": "Smart Study Planner",
        "nav_home": "Home",
        "nav_history": "History",
        "nav_analytics": "Analytics",
        "nav_login": "Login",
        "nav_register": "Register",
        "nav_logout": "Logout",
        "hello": "Hello",
        "login_title": "Login",
        "register_title": "Create Account",
        "username": "Username",
        "password": "Password",
        "login_button": "Sign In",
        "register_button": "Create Account",
        "history_title": "Your Saved Plans",
        "history_empty": "No plans saved yet. Generate a plan while logged in.",
        "download_pdf": "Download PDF",
        "home_eyebrow": "Student Planner",
        "home_title": "Smart Study Planner",
        "home_lede": "Fill your subjects, click Generate Plan, and the app will suggest a smart daily schedule with break balance and exam focus.",
        "home_time_slots": "Time slots",
        "home_actions": "Actions",
        "home_model": "Model",
        "home_actions_title": "5 types of study actions",
        "home_quick_start": "Quick Start",
        "home_quick_start_title": "How to use in 3 steps",
        "home_inputs": "Inputs",
        "home_configure": "Configure the planner",
        "home_add_subject": "Add subject",
        "home_training_preset": "Training preset",
        "home_load_model": "Load a saved model",
        "home_generate": "Generate study plan",
        "home_how_it_works": "How it works",
        "home_learns": "What the planner learns",
        "home_training": "Training",
        "home_generating": "Generating your plan...",
        "home_generate_note": "Initializing training...",
        "home_fix_inputs": "Please fix these inputs",
        "home_success": "Plan generated successfully",
        "home_results": "Results",
        "home_optimized_schedule": "Optimized schedule",
        "home_comparison": "Comparison",
        "home_total_reward": "Total reward",
        "home_final_epsilon": "Final epsilon",
        "home_random_schedule": "Random schedule",
        "home_model_loaded": "Model loaded",
        "home_saved_model": "Saved model",
        "home_real_progress": "Real Progress Tracking",
        "home_replan_title": "Mark each slot and auto-replan if needed",
        "home_completed": "Completed",
        "home_missed": "Missed",
        "home_adaptive_replan": "Adaptive Replan",
        "home_replan_tip": "Tip: If you miss a slot, click Adaptive Replan to rebalance remaining slots.",
        "home_notifications": "Smart Notifications",
        "home_notifications_title": "Personal alerts and reminders",
        "home_history_note": "Your generated plans are saved in your account history.",
        "home_view_saved": "View saved plans",
        "home_training_curve": "Training curve",
        "home_reward_vs_episodes": "Reward vs episodes",
        "home_qtable": "Q-table",
        "home_policy_snapshot": "Policy snapshot (easy view)",
        "home_policy_hint": "Instead of all 75 states, this shows slot-wise recommendations with confidence. The full numeric Q-table stays stored privately on the server.",
        "home_weekly": "Weekly Planner",
        "home_calendar_view": "7-day calendar view",
        "home_explainable": "Explainable AI",
        "home_why_slots": "Why each slot got each subject",
        "home_algo_compare": "Algorithm Comparison",
        "home_qlearning_vs_sarsa": "Q-Learning vs SARSA",
        "analytics_title": "Analytics - Study Planner",
        "analytics_eyebrow": "Dashboard",
        "analytics_heading": "Your Study Analytics",
        "analytics_lede": "Track your progress, streaks, subject mastery, and recovery patterns in one place.",
        "analytics_completion": "Completion",
        "analytics_streak": "Streak",
        "analytics_recent_14d": "Recent 14d",
        "analytics_total_sessions": "Total Sessions",
        "analytics_sessions_desc": "Study sessions recorded",
        "analytics_completion_rate": "Completion Rate",
        "analytics_completion_desc": "Sessions completed successfully",
        "analytics_current_streak": "Current Streak",
        "analytics_streak_desc": "Days with activity",
        "analytics_total_plans": "Total Plans",
        "analytics_plans_desc": "Study plans generated",
        "analytics_missed_slots": "Missed Slots",
        "analytics_missed_desc": "Slots marked as missed",
        "analytics_recent_completion": "Recent Completion",
        "analytics_recent_desc": "Last 14 days completion rate",
        "analytics_real_progress": "Real Progress",
        "analytics_last_14_days": "Last 14 days activity",
        "analytics_no_activity": "No activity logged yet",
        "analytics_no_activity_desc": "Complete or miss a study slot and this panel will show your 14-day rhythm here.",
        "analytics_subject_analysis": "Subject Analysis",
        "analytics_subject_heading": "Performance by subject",
        "analytics_no_subjects": "No subject data yet. Complete some study sessions to see analytics.",
        "analytics_insights": "Insights",
        "analytics_takeaways": "Key takeaways",
        "analytics_generate_new": "Generate New Plan",
        "analytics_view_saved": "View Saved Plans",
        "analytics_notifications": "Smart Notifications",
        "analytics_alerts": "Alerts and reminders",
        "analytics_notifications_note": "These update automatically based on progress and streak patterns.",
        "analytics_no_notifications": "No notifications yet.",
        "analytics_settings": "Notification Settings",
        "analytics_control_alerts": "Control alert types",
        "analytics_settings_note": "Turn reminders on or off without changing your study plan.",
        "analytics_in_app": "In-app notifications",
        "analytics_daily": "Daily reminder",
        "analytics_streak_alerts": "Streak alerts",
        "analytics_missed_alerts": "Missed slot alerts",
        "analytics_auto_save": "Changes save automatically.",
    },
    "hi": {
        "app_title": "Smart Study Planner",
        "analytics_title": "Analytics - Study Planner",
        "analytics_eyebrow": "Dashboard",
        "analytics_heading": "Your Study Analytics",
        "analytics_lede": "Track your progress, streaks, subject mastery, and recovery patterns in one place.",
        "analytics_completion": "Completion",
        "analytics_streak": "Streak",
        "analytics_recent_14d": "Recent 14d",
        "analytics_total_sessions": "Total Sessions",
        "analytics_sessions_desc": "Study sessions recorded",
        "analytics_completion_rate": "Completion Rate",
        "analytics_completion_desc": "Sessions completed successfully",
        "analytics_current_streak": "Current Streak",
        "analytics_streak_desc": "Days with activity",
        "analytics_total_plans": "Total Plans",
        "analytics_plans_desc": "Study plans generated",
        "analytics_missed_slots": "Missed Slots",
        "analytics_missed_desc": "Slots marked as missed",
        "analytics_recent_completion": "Recent Completion",
        "analytics_recent_desc": "Last 14 days completion rate",
        "analytics_real_progress": "Real Progress",
        "analytics_last_14_days": "Last 14 days activity",
        "analytics_no_activity": "No activity logged yet",
        "analytics_no_activity_desc": "Complete or miss a study slot and this panel will show your 14-day rhythm here.",
        "analytics_subject_analysis": "Subject Analysis",
        "analytics_subject_heading": "Performance by subject",
        "analytics_no_subjects": "No subject data yet. Complete some study sessions to see analytics.",
        "analytics_insights": "Insights",
        "analytics_takeaways": "Key takeaways",
        "analytics_generate_new": "Generate New Plan",
        "analytics_view_saved": "View Saved Plans",
        "analytics_notifications": "Smart Notifications",
        "analytics_alerts": "Alerts and reminders",
        "analytics_notifications_note": "These update automatically based on progress and streak patterns.",
        "analytics_no_notifications": "No notifications yet.",
        "analytics_settings": "Notification Settings",
        "analytics_control_alerts": "Control alert types",
        "analytics_settings_note": "Turn reminders on or off without changing your study plan.",
        "analytics_in_app": "In-app notifications",
        "analytics_daily": "Daily reminder",
        "analytics_streak_alerts": "Streak alerts",
        "analytics_missed_alerts": "Missed slot alerts",
        "analytics_auto_save": "Changes save automatically.",
        "nav_home": "Home",
        "nav_history": "History",
        "nav_analytics": "Analytics",
        "analytics_title": "Analytics - Study Planner",
        "analytics_eyebrow": "Dashboard",
        "analytics_heading": "Aapki Study Analytics",
        "analytics_lede": "Aapki progress, streaks, subject mastery aur recovery patterns ek jagah dekhein.",
        "analytics_completion": "Completion",
        "analytics_streak": "Streak",
        "analytics_recent_14d": "Recent 14d",
        "analytics_total_sessions": "Total Sessions",
        "analytics_sessions_desc": "Recorded study sessions",
        "analytics_completion_rate": "Completion Rate",
        "analytics_completion_desc": "Successfully complete hui sessions",
        "analytics_current_streak": "Current Streak",
        "analytics_streak_desc": "Activity wale din",
        "analytics_total_plans": "Total Plans",
        "analytics_plans_desc": "Generated study plans",
        "analytics_missed_slots": "Missed Slots",
        "analytics_missed_desc": "Miss kiye gaye slots",
        "analytics_recent_completion": "Recent Completion",
        "analytics_recent_desc": "Pichhle 14 din ki completion rate",
        "analytics_real_progress": "Real Progress",
        "analytics_last_14_days": "Last 14 days activity",
        "analytics_no_activity": "Abhi koi activity logged nahi hai",
        "analytics_no_activity_desc": "Koi study slot complete ya miss karte hi yahan 14-day rhythm dikhne lagega.",
        "analytics_subject_analysis": "Subject Analysis",
        "analytics_subject_heading": "Subject-wise performance",
        "analytics_no_subjects": "Abhi subject data nahi hai. Analytics dekhne ke liye kuch study sessions complete karein.",
        "analytics_insights": "Insights",
        "analytics_takeaways": "Key takeaways",
        "analytics_generate_new": "Naya Plan Generate Karein",
        "analytics_view_saved": "Saved Plans Dekhein",
        "analytics_notifications": "Smart Notifications",
        "analytics_alerts": "Alerts aur reminders",
        "analytics_notifications_note": "Ye progress aur streak patterns ke hisaab se automatically update hota hai.",
        "analytics_no_notifications": "Abhi koi notification nahi hai.",
        "analytics_settings": "Notification Settings",
        "analytics_control_alerts": "Alert types control karein",
        "analytics_settings_note": "Study plan badle bina reminders on/off karein.",
        "analytics_in_app": "In-app notifications",
        "analytics_daily": "Daily reminder",
        "analytics_streak_alerts": "Streak alerts",
        "analytics_missed_alerts": "Missed slot alerts",
        "analytics_auto_save": "Changes automatically save ho jate hain.",
        "nav_login": "Login",
        "nav_register": "Register",
        "nav_logout": "Logout",
        "hello": "Namaste",
        "login_title": "Login",
        "register_title": "Naya Account Banaye",
        "username": "Username",
        "password": "Password",
        "login_button": "Sign In",
        "register_button": "Account Banaye",
        "history_title": "Aapke Saved Plans",
        "history_empty": "Abhi koi saved plan nahi hai. Login karke plan generate karein.",
        "download_pdf": "PDF Download",
        "home_eyebrow": "Student Planner",
        "home_title": "Smart Study Planner",
        "home_lede": "Apne subjects bhariye, Generate Plan dabaiye, aur app aapke liye smart daily schedule suggest karega.",
        "home_time_slots": "Time slots",
        "home_actions": "Actions",
        "home_model": "Model",
        "home_actions_title": "Study actions ke 5 types",
        "home_quick_start": "Quick Start",
        "home_quick_start_title": "3 steps mein use kaise karein",
        "home_inputs": "Inputs",
        "home_configure": "Planner set karein",
        "home_add_subject": "Subject add karein",
        "home_training_preset": "Training preset",
        "home_load_model": "Saved model load karein",
        "home_generate": "Study plan generate karein",
        "home_how_it_works": "Kaise kaam karta hai",
        "home_learns": "Planner kya seekhta hai",
        "home_training": "Training",
        "home_generating": "Aapka plan ban raha hai...",
        "home_generate_note": "Training shuru ho rahi hai...",
        "home_fix_inputs": "Kripya inputs theek karein",
        "home_success": "Plan safalta se ban gaya",
        "home_results": "Results",
        "home_optimized_schedule": "Optimized schedule",
        "home_comparison": "Comparison",
        "home_total_reward": "Total reward",
        "home_final_epsilon": "Final epsilon",
        "home_random_schedule": "Random schedule",
        "home_model_loaded": "Model loaded",
        "home_saved_model": "Saved model",
        "home_real_progress": "Real Progress Tracking",
        "home_replan_title": "Har slot mark karein aur zarurat ho to auto-replan karein",
        "home_completed": "Completed",
        "home_missed": "Missed",
        "home_adaptive_replan": "Adaptive Replan",
        "home_replan_tip": "Tip: Agar aap koi slot miss karte hain, to remaining slots ko balance karne ke liye Adaptive Replan dabaiye.",
        "home_notifications": "Smart Notifications",
        "home_notifications_title": "Personal alerts aur reminders",
        "home_history_note": "Aapke generated plans account history mein save hote hain.",
        "home_view_saved": "Saved plans dekhein",
        "home_training_curve": "Training curve",
        "home_reward_vs_episodes": "Reward vs episodes",
        "home_qtable": "Q-table",
        "home_policy_snapshot": "Policy snapshot (easy view)",
        "home_policy_hint": "Puri 75 states ki jagah, yeh slot-wise recommendations aur confidence dikhata hai. Full numeric Q-table server par private rehti hai.",
        "home_weekly": "Weekly Planner",
        "home_calendar_view": "7-day calendar view",
        "home_explainable": "Explainable AI",
        "home_why_slots": "Har slot ko subject kyun mila",
        "home_algo_compare": "Algorithm Comparison",
        "home_qlearning_vs_sarsa": "Q-Learning vs SARSA",
    },
}


@dataclass
class TrainingResult:
    schedule: List[str]
    schedule_reward: float
    random_schedule: List[str]
    random_reward: float
    episode_rewards: List[float]
    q_table_rows: List[Dict[str, Any]]
    plot_base64: Optional[str]
    plot_path: Optional[str]
    final_epsilon: float
    comparison_winner: str
    q_table_path: str
    model_loaded: bool
    action_labels: List[str]
    user_summary: str
    weekly_plan: List[Dict[str, Any]]
    decision_reasons: List[Dict[str, str]] = None
    sarsa_schedule: Optional[List[str]] = None
    sarsa_reward: Optional[float] = None
    algorithm_comparison: Optional[Dict[str, Any]] = None
    policy_action_mix: Optional[List[Dict[str, Any]]] = None


def _default_subject_rows() -> List[Dict[str, str]]:
    return [
        {"name": "OS", "difficulty": "5", "strength": "3"},
        {"name": "Java", "difficulty": "4", "strength": "3"},
        {"name": "DSA", "difficulty": "5", "strength": "2"},
    ]


def _parse_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _collect_subject_rows(form_data: Dict[str, str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    row_count = _parse_int(form_data.get("subject_count", "3"), 3)
    row_count = max(1, row_count)

    for index in range(row_count):
        rows.append(
            {
                "name": form_data.get(f"subject_name_{index}", "").strip(),
                "difficulty": form_data.get(f"difficulty_{index}", "3").strip() or "3",
                "strength": form_data.get(f"strength_{index}", "3").strip() or "3",
            }
        )

    return rows if rows else _default_subject_rows()


def _build_config_from_form(form: Dict[str, str]) -> StudyPlannerConfig:
    subject_rows = _collect_subject_rows(form)
    subjects = [row["name"] or f"Subject {index + 1}" for index, row in enumerate(subject_rows)]
    difficulties = [_parse_int(row["difficulty"], 3) for row in subject_rows]
    strengths = [_parse_int(row["strength"], 3) for row in subject_rows]

    return StudyPlannerConfig.from_input(
        subjects=subjects,
        difficulties=difficulties,
        strengths=strengths,
        exam_date_text=form.get("exam_date", "").strip(),
        episodes=_parse_int(form.get("episodes", "1500"), 1500),
        alpha=_parse_float(form.get("alpha", "0.1"), 0.1),
        gamma=_parse_float(form.get("gamma", "0.9"), 0.9),
        epsilon=_parse_float(form.get("epsilon", "1.0"), 1.0),
        epsilon_decay=_parse_float(form.get("epsilon_decay", "0.995"), 0.995),
        min_epsilon=_parse_float(form.get("min_epsilon", "0.05"), 0.05),
    )


def _validate_config(config: StudyPlannerConfig, subject_rows: List[Dict[str, str]]) -> List[str]:
    errors: List[str] = []
    cleaned_names = [subject.strip() for subject in config.subjects]

    if len(cleaned_names) < 1:
        errors.append("Add at least one subject.")
    if any(not name for name in cleaned_names):
        errors.append("Subject names cannot be empty.")
    if len(set(name.lower() for name in cleaned_names if name)) != len([name for name in cleaned_names if name]):
        errors.append("Subject names should be unique.")

    if not (100 <= config.episodes <= 10000):
        errors.append("Training episodes must be between 100 and 10000.")
    if not (0.01 <= config.alpha <= 1.0):
        errors.append("Learning rate (alpha) must be between 0.01 and 1.0.")
    if not (0.10 <= config.gamma <= 0.99):
        errors.append("Discount factor (gamma) must be between 0.10 and 0.99.")
    if not (0.01 <= config.epsilon <= 1.0):
        errors.append("Initial epsilon must be between 0.01 and 1.0.")
    if not (0.90 <= config.epsilon_decay <= 0.999):
        errors.append("Epsilon decay must be between 0.90 and 0.999.")
    if not (0.01 <= config.min_epsilon <= 0.20):
        errors.append("Minimum epsilon must be between 0.01 and 0.20.")
    if config.min_epsilon > config.epsilon:
        errors.append("Minimum epsilon cannot be greater than initial epsilon.")

    for index, row in enumerate(subject_rows):
        difficulty = _parse_int(row.get("difficulty", "3"), 3)
        strength = _parse_int(row.get("strength", "3"), 3)
        if not (1 <= difficulty <= 5):
            errors.append(f"Difficulty for subject {index + 1} must be between 1 and 5.")
        if not (1 <= strength <= 5):
            errors.append(f"Confidence for subject {index + 1} must be between 1 and 5.")

    return errors


def _load_uploaded_model(upload_file, expected_shape: tuple[int, int]) -> Optional[np.ndarray]:
    if upload_file is None or not upload_file.filename:
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as temp_file:
        upload_file.save(temp_file.name)
        temp_path = temp_file.name

    try:
        loaded = load_q_table(temp_path)
        q_table = np.asarray(loaded["q_table"], dtype=float)
        if q_table.shape != expected_shape:
            return None
        return q_table
    finally:
        Path(temp_path).unlink(missing_ok=True)


def _build_weekly_plan(time_slots: List[str], subjects: List[str], daily_schedule: List[str]) -> List[Dict[str, Any]]:
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    subject_count = max(1, len(subjects))
    weekly_plan: List[Dict[str, Any]] = []

    for day_index, day_name in enumerate(day_names):
        slot_map: Dict[str, str] = {slot_name: "Free" for slot_name in time_slots}
        for slot_index, action in enumerate(daily_schedule):
            if slot_index >= len(time_slots):
                break

            if action == "Break":
                slot_map[time_slots[slot_index]] = "Break"
                continue

            if action in subjects:
                action_index = subjects.index(action)
            else:
                action_index = slot_index % subject_count

            rotated_subject = subjects[(action_index + day_index) % subject_count]
            slot_map[time_slots[slot_index]] = rotated_subject

        weekly_plan.append({"day": day_name, "slots": slot_map})

    return weekly_plan


def _pad_schedule(schedule: List[str], time_slots: List[str], fill_value: str = "Free") -> List[str]:
    padded = list(schedule[: len(time_slots)])
    while len(padded) < len(time_slots):
        padded.append(fill_value)
    return padded


def _extract_subject(action_name: str) -> str:
    if action_name in {"Break", "Free", "Meditation", "Snack", "Water_Break"}:
        return action_name
    if "_" in action_name:
        return action_name.split("_", 1)[0]
    return action_name


def _adaptive_replan_schedule(
    schedule: List[str],
    missed_slot_index: int,
    subjects: List[str],
    difficulties: Dict[str, int],
    strengths: Dict[str, int],
    exam_date_text: str = "",
) -> Dict[str, Any]:
    if missed_slot_index < 0 or missed_slot_index >= len(schedule):
        return {"schedule": schedule, "note": "Invalid missed slot index."}

    replanned = list(schedule)
    days_left = None
    if exam_date_text:
        try:
            exam_date = datetime.strptime(exam_date_text, "%Y-%m-%d").date()
            days_left = (exam_date - datetime.utcnow().date()).days
        except ValueError:
            days_left = None

    if not subjects:
        return {"schedule": replanned, "note": "No subjects provided for replanning."}

    action_catalog = set(subjects)
    action_catalog.update(f"{subject}_Deep" for subject in subjects)
    action_catalog.update(f"{subject}_Rev" for subject in subjects)
    action_catalog.update(f"{subject}_Mock" for subject in subjects)

    notes: List[str] = []
    missed_action = schedule[missed_slot_index]
    missed_subject = _extract_subject(missed_action)

    for slot_index in range(missed_slot_index, len(replanned)):
        best_subject = subjects[0]
        best_score = -10**9

        for subject in subjects:
            difficulty = difficulties.get(subject, 3)
            strength = strengths.get(subject, 3)
            score = float(difficulty + (5 - strength))

            if subject == missed_subject:
                score += 2.3

            if slot_index > 0 and _extract_subject(replanned[slot_index - 1]) == subject:
                score -= 1.2

            if days_left is not None and days_left <= 7 and difficulty >= 4:
                score += 1.0

            if score > best_score:
                best_score = score
                best_subject = subject

        difficulty = difficulties.get(best_subject, 3)
        strength = strengths.get(best_subject, 3)

        if slot_index == missed_slot_index:
            candidate = f"{best_subject}_Rev"
            reason = f"Recovered missed slot with quick revision for {best_subject}."
        elif days_left is not None and days_left <= 5:
            candidate = f"{best_subject}_Mock"
            reason = f"Exam near, shifted {best_subject} to mock practice."
        elif difficulty >= 4 and strength <= 2:
            candidate = f"{best_subject}_Deep"
            reason = f"Boosted difficult low-confidence subject {best_subject}."
        else:
            candidate = best_subject
            reason = f"Balanced slot toward {best_subject}."

        replanned[slot_index] = candidate if candidate in action_catalog else best_subject
        notes.append(reason)

    summary = " ".join(notes[:2]) if notes else "Adaptive replan generated."
    return {"schedule": replanned, "note": summary}


def _generate_smart_notifications(
    db_path: str,
    user_id: int,
    analytics_data: Dict[str, Any],
    prefs: Dict[str, Any],
) -> None:
    if not prefs.get("in_app_enabled", True):
        return

    if prefs.get("streak_alert_enabled", True) and analytics_data.get("current_streak", 0) >= 3:
        create_notification(
            db_path,
            user_id,
            "streak",
            f"Great momentum. Current streak is {analytics_data['current_streak']} days.",
            dedupe_daily=True,
        )

    if prefs.get("missed_slot_alert_enabled", True) and analytics_data.get("missed_count", 0) > 0:
        create_notification(
            db_path,
            user_id,
            "missed-slot",
            "You have missed slots recently. Use adaptive replan to recover quickly.",
            dedupe_daily=True,
        )

    if prefs.get("daily_reminder_enabled", True):
        recent_activity = analytics_data.get("recent_activity", [])
        if recent_activity:
            today_stats = recent_activity[-1]
            if (today_stats.get("completed", 0) + today_stats.get("missed", 0)) == 0:
                create_notification(
                    db_path,
                    user_id,
                    "daily-reminder",
                    "No progress logged today yet. Complete one slot to keep your momentum.",
                    dedupe_daily=True,
                )

    if analytics_data.get("recent_completion_rate", 0) < 60:
        create_notification(
            db_path,
            user_id,
            "consistency",
            "Recent completion rate dropped below 60%. Consider shorter focused sessions.",
            dedupe_daily=True,
        )


def run_planner(form: Dict[str, str], uploaded_model=None) -> TrainingResult:
    subject_rows = _collect_subject_rows(form)
    config = _build_config_from_form(form)
    validation_errors = _validate_config(config, subject_rows)
    if validation_errors:
        raise ValueError("\n".join(validation_errors))

    environment = StudyPlannerEnvironment(config)
    agent = QLearningAgent(
        num_states=environment.num_states,
        num_actions=environment.num_actions,
        alpha=config.alpha,
        gamma=config.gamma,
        epsilon=config.epsilon,
        epsilon_decay=config.epsilon_decay,
        min_epsilon=config.min_epsilon,
        seed=config.seed,
    )

    model_loaded = False
    if uploaded_model is not None and getattr(uploaded_model, "filename", ""):
        loaded_q_table = _load_uploaded_model(uploaded_model, agent.q_table.shape)
        if loaded_q_table is not None:
            agent.q_table = loaded_q_table
            model_loaded = True

    episode_rewards: List[float] = []
    if not model_loaded:
        episode_rewards = train_agent(environment, agent, config.episodes, log_interval=0)

    schedule, schedule_reward, decision_reasons = generate_greedy_schedule_with_explanations(environment, agent, config)
    random_schedule, random_reward = generate_random_schedule(environment)
    schedule = _pad_schedule(schedule, environment.time_slots)
    random_schedule = _pad_schedule(random_schedule, environment.time_slots)
    weekly_plan = _build_weekly_plan(environment.time_slots, environment.subjects, schedule)

    # Train SARSA agent for comparison
    sarsa_agent = SARSAAgent(
        num_states=environment.num_states,
        num_actions=environment.num_actions,
        alpha=config.alpha,
        gamma=config.gamma,
        epsilon=config.epsilon,
        epsilon_decay=config.epsilon_decay,
        min_epsilon=config.min_epsilon,
        seed=config.seed,
    )
    
    if not model_loaded:
        train_sarsa_agent(environment, sarsa_agent, config.episodes, log_interval=0)
    
    sarsa_schedule, sarsa_reward = generate_sarsa_greedy_schedule(environment, sarsa_agent)
    sarsa_schedule = _pad_schedule(sarsa_schedule, environment.time_slots)
    
    # Algorithm comparison
    algorithm_comparison = {
        "qlearning": {
            "reward": schedule_reward,
            "schedule": schedule,
        },
        "sarsa": {
            "reward": sarsa_reward,
            "schedule": sarsa_schedule,
        },
        "winner": "Q-Learning" if schedule_reward >= sarsa_reward else "SARSA",
        "reward_diff": abs(schedule_reward - sarsa_reward),
    }

    plot_base64 = plot_rewards_as_base64(episode_rewards) if episode_rewards else None
    plot_path = str(Path("reward_vs_episodes.png").resolve()) if episode_rewards else None

    q_table_path = save_q_table(
        "saved_q_table.npz",
        agent.q_table,
        [
            f"{environment.time_slots[slot_index]} / Energy {energy_level} / {last_action_type}"
            for slot_index, energy_level, last_action_type in environment.state_space
        ],
        environment.actions,
    )

    q_table_rows: List[Dict[str, Any]] = []
    policy_action_mix: List[Dict[str, Any]] = []

    # Summarize the learned policy by time slot so the UI stays readable.
    for slot_index, slot_name in enumerate(environment.time_slots):
        slot_state_indexes = [
            index
            for index, state in enumerate(environment.state_space)
            if state[0] == slot_index
        ]
        if not slot_state_indexes:
            continue

        avg_q_values = np.mean(agent.q_table[slot_state_indexes], axis=0)
        ranked_actions = sorted(
            zip(environment.actions, [float(value) for value in avg_q_values]),
            key=lambda item: item[1],
            reverse=True,
        )
        top_actions = [f"{action} ({value:.2f})" for action, value in ranked_actions[:3]]
        confidence_gap = 0.0
        if len(ranked_actions) > 1:
            confidence_gap = ranked_actions[0][1] - ranked_actions[1][1]

        if confidence_gap >= 1.0:
            confidence_label = "High"
        elif confidence_gap >= 0.4:
            confidence_label = "Medium"
        else:
            confidence_label = "Low"

        q_table_rows.append(
            {
                "state_label": slot_name,
                "best_action": ranked_actions[0][0] if ranked_actions else "-",
                "top_actions": top_actions,
                "confidence": confidence_label,
                "confidence_gap": confidence_gap,
            }
        )

    best_action_counts: Dict[str, int] = {}
    for state_values in agent.q_table:
        best_action_index = int(np.argmax(state_values))
        action_name = environment.actions[best_action_index]
        best_action_counts[action_name] = best_action_counts.get(action_name, 0) + 1

    total_states = len(environment.state_space)
    sorted_mix = sorted(best_action_counts.items(), key=lambda item: item[1], reverse=True)
    for action_name, count in sorted_mix[:6]:
        policy_action_mix.append(
            {
                "action": action_name,
                "states": count,
                "percent": (count / total_states) * 100 if total_states else 0.0,
            }
        )

    comparison_winner = "RL-based schedule" if schedule_reward >= random_reward else "Random schedule"
    summary_mode = "loaded a saved model" if model_loaded else "trained a new model"
    user_summary = (
        f"Planner {summary_mode}, selected {comparison_winner}, "
        f"and achieved reward {schedule_reward:.2f}."
    )

    return TrainingResult(
        schedule=schedule,
        schedule_reward=schedule_reward,
        random_schedule=random_schedule,
        random_reward=random_reward,
        episode_rewards=episode_rewards,
        q_table_rows=q_table_rows,
        plot_base64=plot_base64,
        plot_path=plot_path,
        final_epsilon=agent.epsilon,
        comparison_winner=comparison_winner,
        q_table_path=q_table_path,
        model_loaded=model_loaded,
        action_labels=environment.actions,
        user_summary=user_summary,
        weekly_plan=weekly_plan,
        decision_reasons=decision_reasons,
        sarsa_schedule=sarsa_schedule,
        sarsa_reward=sarsa_reward,
        algorithm_comparison=algorithm_comparison,
        policy_action_mix=policy_action_mix,
    )


def _serialize_plan_for_db(result: TrainingResult, config: StudyPlannerConfig, language: str) -> Dict[str, Any]:
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "language": language,
        "config": {
            "subjects": config.subjects,
            "difficulties": config.difficulties,
            "strengths": config.strengths,
            "episodes": config.episodes,
            "alpha": config.alpha,
            "gamma": config.gamma,
            "epsilon": config.epsilon,
        },
        "result": {
            "schedule": result.schedule,
            "schedule_reward": result.schedule_reward,
            "random_schedule": result.random_schedule,
            "random_reward": result.random_reward,
            "comparison_winner": result.comparison_winner,
            "user_summary": result.user_summary,
            "weekly_plan": result.weekly_plan,
            "action_labels": result.action_labels,
            "decision_reasons": result.decision_reasons or [],
            "sarsa_schedule": result.sarsa_schedule,
            "sarsa_reward": result.sarsa_reward,
            "algorithm_comparison": result.algorithm_comparison,
        },
    }


def _build_plan_pdf(plan: Dict[str, Any], username: str) -> BytesIO:
    payload = plan["payload"]
    result = payload["result"]
    config = payload["config"]

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 2 * cm

    def write_line(text: str, gap: float = 0.6 * cm, size: int = 11) -> None:
        nonlocal y
        pdf.setFont("Helvetica", size)
        pdf.drawString(2 * cm, y, text)
        y -= gap

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(2 * cm, y, "Reinforcement Learning Smart Study Planner")
    y -= 0.9 * cm
    write_line(f"User: {username}")
    write_line(f"Saved Plan ID: {plan['id']}")
    write_line(f"Saved At: {plan['created_at']}")

    y -= 0.3 * cm
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(2 * cm, y, "Daily Optimized Schedule")
    y -= 0.8 * cm

    for slot_name, action in zip(["Morning", "Afternoon", "Evening"], result["schedule"]):
        write_line(f"{slot_name}: {action}")

    write_line(f"Optimized Reward: {result['schedule_reward']:.2f}")
    write_line(f"Random Reward: {result['random_reward']:.2f}")
    write_line(f"Winner: {result['comparison_winner']}")

    y -= 0.3 * cm
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(2 * cm, y, "Weekly Planner")
    y -= 0.8 * cm

    for day_item in result["weekly_plan"]:
        if y < 3.5 * cm:
            pdf.showPage()
            y = height - 2 * cm
        slots = day_item["slots"]
        line = (
            f"{day_item['day']}: "
            f"Morning - {slots.get('Morning', '-')}, "
            f"Afternoon - {slots.get('Afternoon', '-')}, "
            f"Evening - {slots.get('Evening', '-')}"
        )
        write_line(line)

    y -= 0.3 * cm
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(2 * cm, y, "Training Settings")
    y -= 0.8 * cm
    write_line(f"Subjects: {', '.join(config['subjects'])}")
    write_line(f"Episodes: {config['episodes']} | Alpha: {config['alpha']} | Gamma: {config['gamma']}")
    write_line(f"Epsilon: {config['epsilon']}")

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer


def create_app() -> Flask:
    project_root = Path(__file__).resolve().parent.parent
    db_path = str(project_root / "data" / "study_planner.db")
    init_db(db_path)

    app = Flask(
        __name__,
        template_folder=str(project_root / "templates"),
        static_folder=str(project_root / "static"),
    )
    app.url_map.strict_slashes = False
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "rl-study-planner-secret-key")
    # Spaces renders apps in an iframe, so session cookies must allow cross-site usage.
    app.config["SESSION_COOKIE_SAMESITE"] = "None"
    app.config["SESSION_COOKIE_SECURE"] = True
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["DB_PATH"] = db_path
    openenv = OpenEnvStudyPlanner(seed=42)
    openenv.reset("easy-01")

    def current_language() -> str:
        selected = session.get("lang", "en")
        return selected if selected in TRANSLATIONS else "en"

    def current_user() -> Optional[Dict[str, Any]]:
        user_id = session.get("user_id")
        if not user_id:
            return None
        return get_user_by_id(app.config["DB_PATH"], int(user_id))

    @app.context_processor
    def inject_common_context():
        lang = current_language()
        user_data = current_user()
        unread_count = 0
        if user_data is not None:
            unread_count = get_unread_notification_count(app.config["DB_PATH"], int(user_data["id"]))

        def t(key: str) -> str:
            return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)

        return {"t": t, "current_lang": lang, "user": user_data, "unread_notification_count": unread_count}

    def template_globals() -> Dict[str, Any]:
        # Fallback globals passed explicitly to avoid any template context mismatch.
        lang = current_language()
        user_data = current_user()
        unread_count = 0
        if user_data is not None:
            unread_count = get_unread_notification_count(app.config["DB_PATH"], int(user_data["id"]))

        def t(key: str) -> str:
            return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)

        return {"t": t, "current_lang": lang, "user": user_data, "unread_notification_count": unread_count}

    @app.get("/set-language/<lang>")
    def set_language(lang: str):
        if lang in TRANSLATIONS:
            session["lang"] = lang
        return redirect(request.referrer or url_for("index"))

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            subject_rows=_default_subject_rows(),
            time_slots=list(StudyPlannerConfig(subjects=[], difficulties={}).time_slots),
            exam_date_value="",
            default_episodes=1500,
            default_alpha=0.1,
            default_gamma=0.9,
            default_epsilon=1.0,
            default_epsilon_decay=0.995,
            default_min_epsilon=0.05,
            latest_plan_id=session.get("last_plan_id"),
            result=None,
            errors=[],
            **template_globals(),
        )

    @app.post("/train")
    def train():
        subject_rows = _collect_subject_rows(request.form)
        errors: List[str] = []
        result: Optional[TrainingResult]
        config = _build_config_from_form(request.form)
        last_plan_id = session.get("last_plan_id")

        try:
            result = run_planner(request.form, request.files.get("model_file"))

            user_data = current_user()
            if user_data is not None:
                payload = _serialize_plan_for_db(result, config, current_language())
                last_plan_id = save_plan(app.config["DB_PATH"], int(user_data["id"]), payload)
                session["last_plan_id"] = last_plan_id
        except Exception as exc:  # pragma: no cover
            result = None
            errors = str(exc).splitlines() if str(exc).strip() else ["Something went wrong while training."]

        return render_template(
            "index.html",
            subject_rows=subject_rows,
            time_slots=list(StudyPlannerConfig(subjects=[], difficulties={}).time_slots),
            exam_date_value=request.form.get("exam_date", ""),
            default_episodes=request.form.get("episodes", "1500"),
            default_alpha=request.form.get("alpha", "0.1"),
            default_gamma=request.form.get("gamma", "0.9"),
            default_epsilon=request.form.get("epsilon", "1.0"),
            default_epsilon_decay=request.form.get("epsilon_decay", "0.995"),
            default_min_epsilon=request.form.get("min_epsilon", "0.05"),
            latest_plan_id=last_plan_id,
            result=result,
            errors=errors,
            **template_globals(),
        )

    @app.route("/register", methods=["GET", "POST"])
    def register():
        errors: List[str] = []
        success_message = ""
        if request.method == "POST":
            username = request.form.get("username", "")
            password = request.form.get("password", "")
            ok, message = create_user(app.config["DB_PATH"], username, password)
            if ok:
                success_message = message
            else:
                errors.append(message)
        return render_template(
            "register.html",
            errors=errors,
            success_message=success_message,
            **template_globals(),
        )

    @app.route("/login", methods=["GET", "POST"])
    def login():
        errors: List[str] = []
        if request.method == "POST":
            username = request.form.get("username", "")
            password = request.form.get("password", "")
            user_data = authenticate_user(app.config["DB_PATH"], username, password)
            if user_data is None:
                errors.append("Invalid username or password.")
            else:
                session["user_id"] = user_data["id"]
                return redirect(url_for("index"))
        return render_template("login.html", errors=errors, **template_globals())

    @app.get("/logout")
    def logout():
        session.pop("user_id", None)
        return redirect(url_for("index"))

    @app.get("/history")
    def history():
        user_data = current_user()
        if user_data is None:
            return redirect(url_for("login"))
        plans = list_user_plans(app.config["DB_PATH"], int(user_data["id"]))
        return render_template("history.html", plans=plans, **template_globals())

    @app.get("/download-plan/<int:plan_id>")
    def download_plan(plan_id: int):
        user_data = current_user()
        if user_data is None:
            return redirect(url_for("login"))

        plan = get_user_plan(app.config["DB_PATH"], int(user_data["id"]), plan_id)
        if plan is None:
            return redirect(url_for("history"))

        pdf_data = _build_plan_pdf(plan, user_data["username"])
        return send_file(
            pdf_data,
            as_attachment=True,
            download_name=f"study_plan_{plan_id}.pdf",
            mimetype="application/pdf",
        )

    @app.get("/analytics")
    def analytics():
        user_data = current_user()
        if user_data is None:
            return redirect(url_for("login"))
        analytics_data = get_user_analytics(app.config["DB_PATH"], int(user_data["id"]))
        prefs = ensure_notification_prefs(app.config["DB_PATH"], int(user_data["id"]))
        _generate_smart_notifications(app.config["DB_PATH"], int(user_data["id"]), analytics_data, prefs)
        notifications_data = list_notifications(app.config["DB_PATH"], int(user_data["id"]), unread_only=False, limit=8)

        return render_template(
            "analytics.html",
            analytics=analytics_data,
            notifications=notifications_data,
            notification_prefs=prefs,
            **template_globals(),
        )

    @app.post("/api/progress-event")
    def progress_event_api():
        user_data = current_user()
        if user_data is None:
            return {"error": "Not authenticated"}, 401

        data = request.get_json() or {}
        subject = str(data.get("subject", "")).strip()
        time_slot = str(data.get("time_slot", "")).strip()
        status = str(data.get("status", "completed")).strip().lower()
        plan_id = data.get("plan_id")
        note = str(data.get("note", "")).strip()
        scheduled_date = data.get("scheduled_date")

        if subject and time_slot:
            record_progress_event(
                app.config["DB_PATH"],
                int(user_data["id"]),
                plan_id,
                subject,
                time_slot,
                status=status,
                scheduled_date=scheduled_date,
                note=note,
            )

            analytics_data = get_user_analytics(app.config["DB_PATH"], int(user_data["id"]))
            prefs = ensure_notification_prefs(app.config["DB_PATH"], int(user_data["id"]))
            _generate_smart_notifications(app.config["DB_PATH"], int(user_data["id"]), analytics_data, prefs)
            return {"success": True, "analytics": analytics_data}

        return {"error": "Missing subject or time_slot"}, 400

    @app.post("/api/record-completion")
    def record_completion_api():
        # Backward compatibility wrapper for older frontend calls.
        return progress_event_api()

    @app.post("/api/adaptive-replan")
    def adaptive_replan_api():
        user_data = current_user()
        if user_data is None:
            return {"error": "Not authenticated"}, 401

        data = request.get_json() or {}
        schedule = data.get("schedule") or []
        missed_slot_index = int(data.get("missed_slot_index", -1))
        subjects = data.get("subjects") or []
        difficulties = data.get("difficulties") or {}
        strengths = data.get("strengths") or {}
        exam_date = str(data.get("exam_date", "")).strip()

        if not isinstance(schedule, list) or not isinstance(subjects, list):
            return {"error": "Invalid payload"}, 400

        replanned = _adaptive_replan_schedule(
            schedule=schedule,
            missed_slot_index=missed_slot_index,
            subjects=[str(item) for item in subjects],
            difficulties={str(key): int(value) for key, value in difficulties.items()},
            strengths={str(key): int(value) for key, value in strengths.items()},
            exam_date_text=exam_date,
        )
        return {"success": True, "schedule": replanned["schedule"], "note": replanned["note"]}

    @app.get("/api/notifications")
    def notifications_api():
        user_data = current_user()
        if user_data is None:
            return {"error": "Not authenticated"}, 401

        unread_only = request.args.get("unread", "0") == "1"
        notifications_data = list_notifications(
            app.config["DB_PATH"],
            int(user_data["id"]),
            unread_only=unread_only,
            limit=20,
        )
        unread_count = get_unread_notification_count(app.config["DB_PATH"], int(user_data["id"]))
        return {"notifications": notifications_data, "unread_count": unread_count}

    @app.post("/api/notifications/<int:notification_id>/read")
    def read_notification_api(notification_id: int):
        user_data = current_user()
        if user_data is None:
            return {"error": "Not authenticated"}, 401
        mark_notification_read(app.config["DB_PATH"], int(user_data["id"]), notification_id)
        unread_count = get_unread_notification_count(app.config["DB_PATH"], int(user_data["id"]))
        return {"success": True, "unread_count": unread_count}

    @app.post("/api/notification-settings")
    def notification_settings_api():
        user_data = current_user()
        if user_data is None:
            return {"error": "Not authenticated"}, 401

        data = request.get_json() or {}
        updated = update_notification_prefs(
            app.config["DB_PATH"],
            int(user_data["id"]),
            {
                "in_app_enabled": bool(data.get("in_app_enabled", True)),
                "daily_reminder_enabled": bool(data.get("daily_reminder_enabled", True)),
                "streak_alert_enabled": bool(data.get("streak_alert_enabled", True)),
                "missed_slot_alert_enabled": bool(data.get("missed_slot_alert_enabled", True)),
            },
        )
        return {"success": True, "settings": updated}

    @app.post("/reset")
    @app.post("/openenv/reset")
    def openenv_reset():
        payload = request.get_json(silent=True) or {}
        task_id = payload.get("task_id") if isinstance(payload, dict) else None
        state = openenv.reset(task_id=task_id)
        return {"ok": True, "observation": state_to_dict(state)}, 200

    @app.get("/state")
    @app.get("/openenv/state")
    def openenv_state():
        state = openenv.state()
        return {"ok": True, "observation": state_to_dict(state)}, 200

    @app.post("/step")
    @app.post("/openenv/step")
    def openenv_step():
        payload = request.get_json(silent=True) or {}
        action = ""
        if isinstance(payload, dict):
            action = str(payload.get("action", "")).strip()
        if not action:
            return {"ok": False, "error": "Missing action"}, 400

        result = openenv.step(action)
        return {"ok": True, **step_to_dict(result)}, 200

    @app.get("/validate")
    @app.get("/openenv/validate")
    def openenv_validate_get():
        return {
            "ok": True,
            "service": "smart-study-planner",
            "status": "ready",
            "endpoints": [
                "/reset",
                "/state",
                "/step",
                "/validate",
                "/openenv/reset",
                "/openenv/state",
                "/openenv/step",
                "/openenv/validate",
            ],
            "tasks": openenv.tasks(),
            "reward_range": [0.0, 1.0],
        }, 200

    @app.post("/validate")
    @app.post("/openenv/validate")
    def openenv_validate_post():
        payload = request.get_json(silent=True) or {}
        return {
            "ok": True,
            "service": "smart-study-planner",
            "status": "ready",
            "received": payload,
            "tasks": openenv.tasks(),
            "reward_range": [0.0, 1.0],
        }, 200

    return app


app = create_app()
