/*
 * @lanhuage: dart
 * @Descripttion: 
 * @version: beta
 * @Author: xiaoshuyui
 * @Date: 2020-05-25 13:28:07
 * @LastEditors: xiaoshuyui
 * @LastEditTime: 2020-05-25 15:31:27
 */ 
import 'dart:math';
import 'dart:convert';
import 'package:crypto/crypto.dart';

String encodeBase64(String data){
  var content = utf8.encode(data);
  var digest = base64Encode(content);
  return digest;
}

String decodeBase64(String data){
  return String.fromCharCodes(base64Decode(data));
}

String encodeSha(String data){
  var digest = sha1.convert(utf8.encode(data));
  return digest.toString();
}

String encodeMd5(String data){
  var digest = md5.convert(utf8.encode(data));
  return digest.toString();
}

String encodeSha224(String data){
  var digest = sha224.convert(utf8.encode(data));
  return digest.toString();
}

String encodeSha256(String data){
  var digest = sha256.convert(utf8.encode(data));
  return digest.toString();
}

String encodeSha384(String data){
  var digest = sha384.convert(utf8.encode(data));
  return digest.toString();
}

String encodeSha512(String data){
  var digest = sha512.convert(utf8.encode(data));
  return digest.toString();
}



void main() {
  String alphabet = 'qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM0123456789';
  int maxStrLength = 15;
  int minStrLength = 8;
  String left = '';
  for (var i = 0;i < maxStrLength;i++){
    if (i<=minStrLength){
      left = left + alphabet[Random().nextInt(alphabet.length)];
    }else{
      var flag = Random().nextInt(5);
      if (flag<=3){
        left = left + alphabet[Random().nextInt(alphabet.length)];
      }else{
        break;
      }
    }
    
  }
  print(left);
  print(encodeBase64(left));
  print(left.length);
  print(decodeBase64(encodeBase64(left)));
  print(".........................");
  print(encodeSha(left));
  print(encodeMd5(left));
}