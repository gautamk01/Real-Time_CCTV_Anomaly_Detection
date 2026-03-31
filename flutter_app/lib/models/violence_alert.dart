// Violence Alert Model
class ViolenceAlert {
  final String status;
  final String timestamp;
  final int confidence;
  final String reason;
  final String alertDir;
  final String rounds;
  final DateTime sentAt;
  final DateTime receivedAt;

  ViolenceAlert({
    required this.status,
    required this.timestamp,
    required this.confidence,
    required this.reason,
    required this.alertDir,
    required this.rounds,
    required this.sentAt,
    required this.receivedAt,
  });

  factory ViolenceAlert.fromFCM(Map<String, dynamic> data) {
    return ViolenceAlert(
      status: data['status'] ?? 'ALERT',
      timestamp: data['timestamp'] ?? 'Unknown',
      confidence: int.tryParse(data['confidence'] ?? '0') ?? 0,
      reason: data['reason'] ?? 'No details provided',
      alertDir: data['alert_dir'] ?? '',
      rounds: data['rounds'] ?? '0',
      sentAt: DateTime.tryParse(data['sent_at'] ?? '') ?? DateTime.now(),
      receivedAt: DateTime.now(),
    );
  }

  String get timeAgo {
    final difference = DateTime.now().difference(receivedAt);
    if (difference.inSeconds < 60) {
      return '${difference.inSeconds}s ago';
    } else if (difference.inMinutes < 60) {
      return '${difference.inMinutes}m ago';
    } else if (difference.inHours < 24) {
      return '${difference.inHours}h ago';
    } else {
      return '${difference.inDays}d ago';
    }
  }

  bool get isReview => status == 'REVIEW';
}
