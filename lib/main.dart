import 'package:flutter/material.dart';
import 'api.dart';

void main() => runApp(StockApp());

class StockApp extends StatelessWidget {
  const StockApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(home: StockPage());
  }
}

class StockPage extends StatefulWidget {
  const StockPage({super.key});

  @override
  _StockPageState createState() => _StockPageState();
}

class _StockPageState extends State<StockPage> {
  final stoidController = TextEditingController();
  String result = '';

  void getPrediction() async {
    final stoid = stoidController.text;
    final prediction = await predictStock(stoid);
    setState(() {
      result = '預測分數：$prediction';
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('股票分析')),
      body: Padding(
        padding: EdgeInsets.all(16),
        child: Column(children: [
          TextField(controller: stoidController, decoration: InputDecoration(labelText: 'Stock_ID')),
          SizedBox(height: 20),
          ElevatedButton(onPressed: getPrediction, child: Text('預測GO')),
          SizedBox(height: 20),
          Text(result),
        ]),
      ),
    );
  }
}