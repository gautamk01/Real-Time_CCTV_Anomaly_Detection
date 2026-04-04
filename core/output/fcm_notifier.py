"""Firebase Cloud Messaging Notification Service."""

import json
import firebase_admin
from firebase_admin import credentials, messaging
from pathlib import Path
from typing import Dict, List
from datetime import datetime


class FCMNotifier:
    """
    Firebase Cloud Messaging notification service for violence detection alerts.
    """
    
    def __init__(self, credentials_path: Path, topic: str = "violence_alerts",
                 enable_review_notifications: bool = False):
        """
        Initialize FCM service with Firebase credentials.
        
        Args:
            credentials_path: Path to Firebase service account JSON key
            topic: FCM topic name for broadcasting alerts
        """
        self.topic = topic
        self.initialized = False
        self.enable_review_notifications = enable_review_notifications
        
        try:
            # Initialize Firebase app if not already initialized
            if not firebase_admin._apps:
                cred = credentials.Certificate(str(credentials_path))
                firebase_admin.initialize_app(cred)
            
            self.initialized = True
            print(f"✅ [FCM] Initialized successfully (topic: {self.topic})")
            
        except Exception as e:
            print(f"❌ [FCM] Initialization failed: {e}")
            print(f"   Credentials path: {credentials_path}")
            self.initialized = False
    
    def _build_message(self, verdict: Dict, history: List[str],
                       timestamp: str, alert_dir: str = None,
                       high_priority: bool = True) -> messaging.Message:
        """Build an FCM message for ALERT or REVIEW incidents."""
        status = verdict.get('status', 'ALERT')
        confidence = verdict.get('confidence', 0)
        reason = verdict.get('reason', 'Unknown')

        if len(reason) > 200:
            reason = reason[:197] + "..."

        if status == "REVIEW":
            title = "Review Needed"
            body = f"Ambiguous incident at {timestamp}"
            android_priority = 'normal'
            apns_priority = '5'
            notification_priority = 'default'
            vibrate = None
            sound = None
            channel_id = 'review_incidents'
            badge = 0
        else:
            title = "🚨 VIOLENCE DETECTED!"
            body = f"Confidence: {confidence}% at {timestamp}"
            android_priority = 'high' if high_priority else 'normal'
            apns_priority = '10' if high_priority else '5'
            notification_priority = 'high' if high_priority else 'default'
            vibrate = [0, 500, 200, 500, 200, 500] if high_priority else None
            sound = 'default' if high_priority else None
            channel_id = 'violence_alerts'
            badge = 1 if high_priority else 0

        android_notification_kwargs = {
            'channel_id': channel_id,
            'priority': notification_priority,
            'default_vibrate_timings': False,
            'default_sound': bool(sound),
        }
        if vibrate is not None:
            android_notification_kwargs['vibrate_timings_millis'] = vibrate

        return messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            data={
                'status': status,
                'timestamp': timestamp,
                'confidence': str(confidence),
                'reason': reason,
                'alert_dir': alert_dir or '',
                'rounds': str(len([h for h in history if h.startswith('[')])),
                'sent_at': datetime.now().isoformat()
            },
            topic=self.topic,
            android=messaging.AndroidConfig(
                priority=android_priority,
                notification=messaging.AndroidNotification(
                    **android_notification_kwargs
                )
            ),
            apns=messaging.APNSConfig(
                headers={'apns-priority': apns_priority},
                payload=messaging.APNSPayload(
                    aps=messaging.Aps(
                        alert=messaging.ApsAlert(title=title, body=body),
                        sound=sound,
                        badge=badge
                    )
                )
            )
        )

    def send_alert(self, verdict: Dict, history: List[str],
                   timestamp: str, alert_dir: str = None) -> bool:
        """
        Send an urgent violence alert notification via FCM.
        
        Args:
            verdict: Investigation verdict dict with status, confidence, reason
            history: Investigation history log
            timestamp: Alert timestamp
            alert_dir: Directory where alert frames are saved (optional)
            
        Returns:
            True if notification sent successfully, False otherwise
        """
        if not self.initialized:
            print("⚠️  [FCM] Skipping notification - service not initialized")
            return False
        
        try:
            confidence = verdict.get('confidence', 0)
            message = self._build_message(
                verdict,
                history,
                timestamp,
                alert_dir=alert_dir,
                high_priority=True,
            )

            # Send message
            response = messaging.send(message)
            print(f"📱 [FCM] Alert notification sent successfully")
            print(f"   Message ID: {response}")
            print(f"   Confidence: {confidence}%")
            print(f"   Topic: {self.topic}")
            
            return True
            
        except Exception as e:
            print(f"❌ [FCM] Failed to send notification: {e}")
            return False

    def send_review(self, verdict: Dict, history: List[str],
                    timestamp: str, alert_dir: str = None) -> bool:
        """Send a lower-priority manual-review notification via FCM."""
        if not self.enable_review_notifications:
            return False
        if not self.initialized:
            print("⚠️  [FCM] Skipping review notification - service not initialized")
            return False

        try:
            message = self._build_message(
                verdict,
                history,
                timestamp,
                alert_dir=alert_dir,
                high_priority=False,
            )
            response = messaging.send(message)
            print(f"📱 [FCM] Review notification sent successfully")
            print(f"   Message ID: {response}")
            return True
        except Exception as e:
            print(f"❌ [FCM] Failed to send review notification: {e}")
            return False
    
    def send_test_notification(self) -> bool:
        """
        Send a test notification to verify FCM is working.
        
        Returns:
            True if test notification sent successfully
        """
        if not self.initialized:
            print("❌ [FCM] Cannot send test - service not initialized")
            return False
        
        try:
            message = messaging.Message(
                notification=messaging.Notification(
                    title="🧪 FCM Test",
                    body="Violence detection system is online!"
                ),
                data={
                    'test': 'true',
                    'timestamp': datetime.now().isoformat()
                },
                topic=self.topic
            )
            
            response = messaging.send(message)
            print(f"✅ [FCM] Test notification sent successfully")
            print(f"   Message ID: {response}")
            return True
            
        except Exception as e:
            print(f"❌ [FCM] Test notification failed: {e}")
            return False


def send_violence_alert(verdict: Dict, history: List[str], 
                       timestamp: str, alert_dir: str = None,
                       fcm_notifier: FCMNotifier = None) -> bool:
    """
    Convenience function to send violence alert notification.
    
    Args:
        verdict: Investigation verdict
        history: Investigation history
        timestamp: Alert timestamp
        alert_dir: Alert directory path
        fcm_notifier: Pre-initialized FCMNotifier instance (optional)
        
    Returns:
        True if notification sent successfully
    """
    if fcm_notifier is None:
        from config import Config
        if not Config.ENABLE_FCM_NOTIFICATIONS:
            return False
        fcm_notifier = FCMNotifier(Config.FIREBASE_CREDENTIALS_PATH, Config.FCM_TOPIC)
    
    return fcm_notifier.send_alert(verdict, history, timestamp, alert_dir)
