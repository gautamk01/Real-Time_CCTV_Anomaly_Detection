import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter/foundation.dart';
import '../models/violence_alert.dart';
import '../providers/alert_provider.dart';
import 'alert_notification_service.dart';

class FCMService {
  final FirebaseMessaging _firebaseMessaging = FirebaseMessaging.instance;
  final AlertProvider alertProvider;

  FCMService(this.alertProvider);

  Future<void> initialize() async {
    // Request permission for notifications
    NotificationSettings settings = await _firebaseMessaging.requestPermission(
      alert: true,
      badge: true,
      sound: true,
      provisional: false,
    );

    if (settings.authorizationStatus == AuthorizationStatus.authorized) {
      debugPrint('✅ User granted permission');
    } else {
      debugPrint('⚠️ User declined or has not accepted permission');
    }

    // Subscribe to violence_alerts topic
    await _firebaseMessaging.subscribeToTopic('violence_alerts');
    debugPrint('📱 Subscribed to violence_alerts topic');

    // Get FCM token (for debugging)
    String? token = await _firebaseMessaging.getToken();
    debugPrint('📱 FCM Token: $token');

    // Configure foreground notification presentation
    await FirebaseMessaging.instance
        .setForegroundNotificationPresentationOptions(
          alert: true,
          badge: true,
          sound: true,
        );

    // Handle foreground messages
    FirebaseMessaging.onMessage.listen(_handleForegroundMessage);

    // Handle background/terminated messages
    FirebaseMessaging.onMessageOpenedApp.listen(_handleBackgroundMessage);
  }

  void _handleForegroundMessage(RemoteMessage message) async {
    debugPrint('🚨 Foreground message received!');
    debugPrint('Title: ${message.notification?.title}');
    debugPrint('Body: ${message.notification?.body}');
    debugPrint('Data: ${message.data}');

    if (message.data.isNotEmpty) {
      final alert = ViolenceAlert.fromFCM(message.data);
      alertProvider.addAlert(alert);

      // Trigger vibration and sound
      await AlertNotificationService.triggerAlert(confidence: alert.confidence);
    }
  }

  void _handleBackgroundMessage(RemoteMessage message) async {
    debugPrint('🚨 Background message opened!');
    debugPrint('Data: ${message.data}');

    if (message.data.isNotEmpty) {
      final alert = ViolenceAlert.fromFCM(message.data);
      alertProvider.addAlert(alert);

      // Trigger vibration and sound
      await AlertNotificationService.triggerAlert(confidence: alert.confidence);
    }
  }
}

// Handle background messages (outside of app context)
@pragma('vm:entry-point')
Future<void> firebaseMessagingBackgroundHandler(RemoteMessage message) async {
  debugPrint('🚨 Background message received (handler)');
  debugPrint('Message data: ${message.data}');
  // Note: We can't update UI here, but we can log/process data
}
