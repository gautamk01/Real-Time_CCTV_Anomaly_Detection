import 'package:flutter/material.dart';
import '../providers/alert_provider.dart';
import '../widgets/alert_card.dart';

class HomeScreen extends StatelessWidget {
  final AlertProvider alertProvider;

  const HomeScreen({super.key, required this.alertProvider});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Violence Alerts',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          // Alert count badge
          ListenableBuilder(
            listenable: alertProvider,
            builder: (context, child) {
              return Padding(
                padding: const EdgeInsets.only(right: 8),
                child: Center(
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 6,
                    ),
                    decoration: BoxDecoration(
                      color: alertProvider.alertCount > 0
                          ? Colors.red.shade700
                          : Colors.grey.shade400,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Text(
                      '${alertProvider.alertCount}',
                      style: const TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                        fontSize: 14,
                      ),
                    ),
                  ),
                ),
              );
            },
          ),
          // Clear button
          ListenableBuilder(
            listenable: alertProvider,
            builder: (context, child) {
              if (alertProvider.alertCount == 0) return const SizedBox.shrink();
              return IconButton(
                icon: const Icon(Icons.clear_all),
                onPressed: () {
                  showDialog(
                    context: context,
                    builder: (context) => AlertDialog(
                      title: const Text('Clear All Alerts'),
                      content: const Text(
                        'Are you sure you want to clear all alerts?',
                      ),
                      actions: [
                        TextButton(
                          onPressed: () => Navigator.pop(context),
                          child: const Text('Cancel'),
                        ),
                        TextButton(
                          onPressed: () {
                            alertProvider.clearAll();
                            Navigator.pop(context);
                          },
                          child: const Text('Clear'),
                        ),
                      ],
                    ),
                  );
                },
              );
            },
          ),
        ],
      ),
      body: ListenableBuilder(
        listenable: alertProvider,
        builder: (context, child) {
          if (alertProvider.alertCount == 0) {
            return _buildEmptyState();
          }

          return ListView.builder(
            itemCount: alertProvider.alertCount,
            padding: const EdgeInsets.symmetric(vertical: 8),
            itemBuilder: (context, index) {
              final alert = alertProvider.alerts[index];
              return AlertCard(alert: alert);
            },
          );
        },
      ),
      // Status indicator
      bottomNavigationBar: Container(
        padding: const EdgeInsets.all(12),
        color: Colors.green.shade100,
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.check_circle, color: Colors.green.shade700, size: 20),
            const SizedBox(width: 8),
            Text(
              'Connected to violence_alerts topic',
              style: TextStyle(
                color: Colors.green.shade700,
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.shield_outlined, size: 80, color: Colors.grey.shade400),
          const SizedBox(height: 16),
          Text(
            'No Alerts Yet',
            style: TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.bold,
              color: Colors.grey.shade700,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'System is monitoring for threats',
            style: TextStyle(fontSize: 14, color: Colors.grey.shade600),
          ),
        ],
      ),
    );
  }
}
