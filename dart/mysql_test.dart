/*
 * @lanhuage: dart
 * @Descripttion: 
 * @version: beta
 * @Author: xiaoshuyui
 * @Date: 2020-05-25 15:47:00
 * @LastEditors: xiaoshuyui
 * @LastEditTime: 2020-05-25 16:32:58
 */ 

import 'package:sqljocky5/sqljocky.dart';

Future main() async {
  var settings = ConnectionSettings(
    user: "othertest",
    password: "othertest123!",
    host: "rm-bp1oct2xit34b58pwlo.mysql.rds.aliyuncs.com",
    port: 3306,
    db: "dart_test",
  );

  var conn = await MySqlConnection.connect(settings);

  StreamedResults results = await conn.execute('select * from userinfo');

  results.forEach((element) {
    print(element);
    print(">>>>>>>>");
  });

  conn.close();
}