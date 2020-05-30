/*
 * @lanhuage: dart
 * @Descripttion: 
 * @version: beta
 * @Author: xiaoshuyui
 * @Date: 2020-05-26 08:50:35
 * @LastEditors: xiaoshuyui
 * @LastEditTime: 2020-05-26 08:58:01
 */ 
class UserInfo{
  int id;
  String _appname;
  String _username;
  String _userpassword;

  UserInfo(this._appname,this._username,this._userpassword);

  UserInfo.map(dynamic obj){
    this._appname = obj["appName"];
    this._username = obj["userName"];
    this._userpassword = obj["userPassWord"];
  }

  String get appName => _appname;
  String get userName => _username;
  String get userPassWord  => _userpassword;

  Map<String, dynamic> toMap(){
    var map = new Map<String, dynamic>();

    map['appName'] = _appname;
    map['userName'] = _username;
    map['userPassWord'] = _userpassword;

    return map;
  }

  void setUserId(int id){
    this.id = id;
  }
}