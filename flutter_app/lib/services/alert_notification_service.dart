import 'package:flutter/services.dart';
import 'package:vibration/vibration.dart';
import 'package:audioplayers/audioplayers.dart';

class AlertNotificationService {
  static final AudioPlayer _audioPlayer = AudioPlayer();

  /// Trigger alert with vibration and sound
  static Future<void> triggerAlert({required int confidence}) async {
    // Trigger vibration pattern based on confidence
    await _vibrateForAlert(confidence);

    // Play alert sound
    await _playAlertSound(confidence);
  }

  /// Vibration pattern based on confidence level
  static Future<void> _vibrateForAlert(int confidence) async {
    try {
      // Check if device supports vibration
      bool? hasVibrator = await Vibration.hasVibrator();
      if (hasVibrator != true) return;

      // Different patterns based on confidence
      if (confidence >= 80) {
        // High confidence: 3 strong bursts
        await Vibration.vibrate(
          pattern: [0, 500, 200, 500, 200, 500],
          intensities: [0, 255, 0, 255, 0, 255],
        );
      } else if (confidence >= 60) {
        // Medium confidence: 2 moderate bursts
        await Vibration.vibrate(
          pattern: [0, 400, 300, 400],
          intensities: [0, 180, 0, 180],
        );
      } else {
        // Low confidence: 1 gentle burst
        await Vibration.vibrate(duration: 300, amplitude: 128);
      }
    } catch (e) {
      print('⚠️ [VIBRATION] Error: $e');
    }
  }

  /// Play alert sound based on confidence
  static Future<void> _playAlertSound(int confidence) async {
    try {
      // Use system notification sound temporarily
      // You can add custom sound files later in assets/sounds/

      if (confidence >= 80) {
        // High confidence: Play twice
        await SystemSound.play(SystemSoundType.alert);
        await Future.delayed(const Duration(milliseconds: 500));
        await SystemSound.play(SystemSoundType.alert);
      } else if (confidence >= 60) {
        // Medium confidence: Play once
        await SystemSound.play(SystemSoundType.alert);
      } else {
        // Low confidence: Gentle click
        await SystemSound.play(SystemSoundType.click);
      }
    } catch (e) {
      print('⚠️ [SOUND] Error: $e');
    }
  }

  /// Cancel any ongoing vibration
  static Future<void> cancelVibration() async {
    try {
      await Vibration.cancel();
    } catch (e) {
      print('⚠️ [VIBRATION] Cancel error: $e');
    }
  }

  /// Dispose resources
  static void dispose() {
    _audioPlayer.dispose();
  }
}
