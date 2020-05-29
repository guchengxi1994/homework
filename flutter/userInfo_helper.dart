/*
 * @lanhuage: dart
 * @Descripttion: 
 * @version: beta
 * @Author: xiaoshuyui
 * @Date: 2020-05-26 08:58:50
 * @LastEditors: xiaoshuyui
 * @LastEditTime: 2020-05-26 09:36:40
 */ 

import 'dart:io' as io;
import 'userInfo.dart';

import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'package:sqflite/sqflite.dart';

class DatabaseHelper{
  static final DatabaseHelper _databaseHelper = new DatabaseHelper.internal();
  factory DatabaseHelper() => _databaseHelper;

  static Database _db;
  Future<Database> get db async{
    if (_db != null) return _db;
    _db = await initDb();
    return _db;
  }

  DatabaseHelper.internal();

  initDb() async{
    io.Directory documentsDirectory = await getApplicationDocumentsDirectory();
    String path = join(documentsDirectory.path,'main.db');
    var theDb = await openDatabase(path, version: 1, onCreate: _onCreate);
    return theDb;
  }

  void _onCreate(Database db,int version) async{
    await db.execute();
  }
  
}

