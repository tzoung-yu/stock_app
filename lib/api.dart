import 'dart:convert';
import 'package:http/http.dart' as http;

Future<double> predictStock(String stoid) async {
  final url = Uri.parse('http://127.0.0.1:8000/predict');
  final response = await http.post(
    url,
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({'stoid': stoid}),
  );

  if (response.statusCode == 200) {
    final data = jsonDecode(response.body);
    return data['prediction'];
  } else {
    throw Exception('Failed to get prediction');
  }
}