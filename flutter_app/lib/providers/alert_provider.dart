import 'package:flutter/material.dart';
import '../models/violence_alert.dart';

class AlertProvider extends ChangeNotifier {
  final List<ViolenceAlert> _alerts = [];

  List<ViolenceAlert> get alerts => List.unmodifiable(_alerts);

  void addAlert(ViolenceAlert alert) {
    _alerts.insert(0, alert); // Add to beginning for newest first
    notifyListeners();
  }

  void clearAll() {
    _alerts.clear();
    notifyListeners();
  }

  int get alertCount => _alerts.length;
}
