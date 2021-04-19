/*
 * @Descripttion: 
 * @version: 
 * @Author: xiaoshuyui
 * @email: guchengxi1994@qq.com
 * @Date: 2021-04-19 19:23:24
 * @LastEditors: xiaoshuyui
 * @LastEditTime: 2021-04-19 19:59:28
 */
import 'dart:async';

import 'package:demo/simpleWIdget.dart';
import 'package:flutter/material.dart';
import 'package:flutter_easyrefresh/easy_refresh.dart';
import 'package:extended_nested_scroll_view/extended_nested_scroll_view.dart'
    as extended;

/// NestedScrollView示例页面
class NestedScrollViewPage extends StatefulWidget {
  @override
  NestedScrollViewPageState createState() {
    return NestedScrollViewPageState();
  }
}

class NestedScrollViewPageState extends State<NestedScrollViewPage>
    with SingleTickerProviderStateMixin {
  // Tab控制器
  late TabController _tabController;
  int _tabIndex = 0;
  // 列表
  int _listCount = 20;
  // 表格
  int _gridCount = 30;

  // 初始化
  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
  }

  @override
  void dispose() {
    super.dispose();
    _tabController.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: extended.NestedScrollView(
        pinnedHeaderSliverHeightBuilder: () {
          return MediaQuery.of(context).padding.top + kToolbarHeight;
        },
        innerScrollPositionKeyBuilder: () {
          if (_tabController.index == 0) {
            return Key('Tab0');
          } else {
            return Key('Tab1');
          }
        },
        headerSliverBuilder: (context, innerBoxIsScrolled) {
          return <Widget>[
            new SliverAppBar(
              title: Text("NestedScrollView"),
              centerTitle: true,
              expandedHeight: 190.0,
              flexibleSpace: SingleChildScrollView(
                physics: NeverScrollableScrollPhysics(),
                child: Container(),
              ),
              floating: false,
              pinned: true,
            ),
          ];
        },
        body: Column(
          children: <Widget>[
            PreferredSize(
              child: new Card(
                color: Theme.of(context).primaryColor,
                elevation: 0.0,
                margin: new EdgeInsets.all(0.0),
                shape: new RoundedRectangleBorder(
                  borderRadius: BorderRadius.all(Radius.circular(0.0)),
                ),
                child: new TabBar(
                  indicatorSize: TabBarIndicatorSize.label,
                  indicatorWeight: 3.0, // 指示器的高度/厚度
                  controller: _tabController,
                  isScrollable: true,
                  onTap: (index) {
                    setState(() {
                      if (index <= 1) {
                        _tabIndex = index;
                      }
                    });
                  },
                  tabs: <Widget>[
                    Container(
                      // width: 20,
                      child: new Tab(
                        text: 'List',
                      ),
                    ),
                    // new Tab(
                    //   text: 'Grid',
                    // ),
                    Container(
                      // width: 20,
                      child: new Tab(
                        text: 'Grid',
                      ),
                    ),
                    Container(
                      width: 300,
                      child: ElevatedButton(
                          onPressed: () {
                            print("这里是按钮");
                          },
                          child: Text("请点击我")),
                    )
                  ],
                ),
              ),
              preferredSize: new Size(double.infinity, 46.0),
            ),
            Expanded(
              child: IndexedStack(
                index: _tabIndex,
                children: <Widget>[
                  extended.NestedScrollViewInnerScrollPositionKeyWidget(
                    Key('Tab0'),
                    EasyRefresh(
                      child: ListView.builder(
                        padding: EdgeInsets.all(0.0),
                        itemBuilder: (context, index) {
                          return SampleListItem();
                        },
                        itemCount: _listCount,
                      ),
                      onRefresh: () async {
                        await Future.delayed(Duration(seconds: 2), () {
                          if (mounted) {
                            setState(() {
                              _listCount = 20;
                            });
                          }
                        });
                      },
                      onLoad: () async {
                        await Future.delayed(Duration(seconds: 2), () {
                          if (mounted) {
                            setState(() {
                              _listCount += 10;
                            });
                          }
                        });
                      },
                    ),
                  ),
                  extended.NestedScrollViewInnerScrollPositionKeyWidget(
                    Key('Tab1'),
                    EasyRefresh(
                      child: GridView.builder(
                        gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                          crossAxisCount: 2,
                          childAspectRatio: 6 / 7,
                        ),
                        itemBuilder: (context, index) {
                          return SampleListItem(
                            direction: Axis.horizontal,
                          );
                        },
                        itemCount: _gridCount,
                      ),
                      onRefresh: () async {
                        await Future.delayed(Duration(seconds: 2), () {
                          if (mounted) {
                            setState(() {
                              _gridCount = 30;
                            });
                          }
                        });
                      },
                      onLoad: () async {
                        await Future.delayed(Duration(seconds: 2), () {
                          if (mounted) {
                            setState(() {
                              _gridCount += 10;
                            });
                          }
                        });
                      },
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
