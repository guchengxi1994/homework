package com.xiaoshuyui.FormOCR;

import com.baidu.aip.ocr.AipOcr;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.HashMap;

public class Sample {

    //设置APPID/AK/SK
    public static final String APP_ID = "17160116";
    public static final String API_KEY = "4Gx0bItZyFX7YGt6mz7KZ3Sc";
    public static final String SECRET_KEY = "zOTzSqAm6x90i43MMYzDQXVwqyNOlxLw";

    public static void main(String[] args) {

        String request_id = "";
        // 初始化一个AipOcr
        AipOcr client = new AipOcr(APP_ID, API_KEY, SECRET_KEY);
        HashMap<String, String> options = new HashMap<String, String>();
        options.put("result_type", "json");


        // 可选：设置网络连接参数
        client.setConnectionTimeoutInMillis(2000);
        client.setSocketTimeoutInMillis(60000);

//        String requestId = "23454320-23255";

        // 可选：设置代理服务器地址, http和socket二选一，或者均不设置
//        client.setHttpProxy("proxy_host", proxy_port);  // 设置http代理
//        client.setSocketProxy("proxy_host", proxy_port);  // 设置socket代理

        // 可选：设置log4j日志输出格式，若不设置，则使用默认配置
        // 也可以直接通过jvm启动参数设置此环境变量
//        System.setProperty("aip.log4j.conf", "path/to/your/log4j.properties");

        // 调用接口
        String path = "d:\\Desktop\\微信图片_20191117101144.png";
        JSONObject res = client.tableRecognitionAsync(path, options);



//        System.out.println(res.toString(2));

        net.sf.json.JSONObject jsonObject = net.sf.json.JSONObject.fromObject(res.toString(2));

//        String result = res.getString("result");

        net.sf.json.JSONArray result = jsonObject.getJSONArray("result");



        if (null!=result && result.size()>0){

            request_id = result.getJSONObject(0).getString("request_id");

        }

        System.out.println(request_id);


        new Thread(new Runnable() {
            public void run() {

            }
        }).start();


        JSONObject res2 = client.tableResultGet(request_id,options);
        System.out.println(res2.toString(2));



    }
}
