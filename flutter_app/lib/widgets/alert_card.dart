import 'package:flutter/material.dart';
import '../models/violence_alert.dart';

class AlertCard extends StatelessWidget {
  final ViolenceAlert alert;

  const AlertCard({super.key, required this.alert});

  bool get _isReview => alert.status == 'REVIEW';

  Color _getConfidenceColor() {
    if (_isReview) {
      return Colors.blue.shade50;
    }
    if (alert.confidence >= 80) {
      return Colors.red.shade100;
    } else if (alert.confidence >= 60) {
      return Colors.orange.shade100;
    } else {
      return Colors.yellow.shade100;
    }
  }

  Color _getConfidenceIconColor() {
    if (_isReview) {
      return Colors.blue.shade700;
    }
    if (alert.confidence >= 80) {
      return Colors.red.shade700;
    } else if (alert.confidence >= 60) {
      return Colors.orange.shade700;
    } else {
      return Colors.yellow.shade700;
    }
  }

  IconData _getIcon() {
    return _isReview ? Icons.rate_review_outlined : Icons.warning_amber_rounded;
  }

  String _getTitle() {
    return _isReview ? 'Manual Review' : 'Violence Alert';
  }

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      color: _getConfidenceColor(),
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header Row
            Row(
              children: [
                Icon(
                  _getIcon(),
                  color: _getConfidenceIconColor(),
                  size: 28,
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        _getTitle(),
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: _getConfidenceIconColor(),
                        ),
                      ),
                      Text(
                        alert.timeAgo,
                        style: TextStyle(
                          fontSize: 12,
                          color: Colors.grey.shade700,
                        ),
                      ),
                    ],
                  ),
                ),
                // Confidence Badge
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 6,
                  ),
                  decoration: BoxDecoration(
                    color: _getConfidenceIconColor(),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    '${alert.confidence}%',
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                      fontSize: 14,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            // Reason
            Text(
              alert.reason,
              style: const TextStyle(fontSize: 14, height: 1.4),
            ),
            const SizedBox(height: 12),
            // Metadata Row
            Row(
              children: [
                Icon(Icons.access_time, size: 16, color: Colors.grey.shade600),
                const SizedBox(width: 4),
                Text(
                  alert.timestamp,
                  style: TextStyle(fontSize: 12, color: Colors.grey.shade700),
                ),
                const SizedBox(width: 16),
                Icon(Icons.layers, size: 16, color: Colors.grey.shade600),
                const SizedBox(width: 4),
                Text(
                  '${alert.rounds} rounds',
                  style: TextStyle(fontSize: 12, color: Colors.grey.shade700),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
