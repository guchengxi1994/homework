package com.xiaoshuyui.FormOCR;

import com.baidu.aip.ocr.AipOcr;
import org.json.JSONObject;

import java.util.HashMap;

public class Test {
    //设置APPID/AK/SK
    public static final String APP_ID = "17160116";
    public static final String API_KEY = "4Gx0bItZyFX7YGt6mz7KZ3Sc";
    public static final String SECRET_KEY = "zOTzSqAm6x90i43MMYzDQXVwqyNOlxLw";

    public static void main(String[] args) {
        String requestid = "17160116_1220836";
        AipOcr client = new AipOcr(APP_ID, API_KEY, SECRET_KEY);
        HashMap<String, String> options = new HashMap<String, String>();
        options.put("result_type", "json");

        // 可选：设置网络连接参数
        client.setConnectionTimeoutInMillis(2000);
        client.setSocketTimeoutInMillis(60000);

        JSONObject res2 = client.tableResultGet(requestid,options);

        JSONObject result = res2.getJSONObject("result");

        System.out.println(result.get("ret_msg"));

        System.out.println(result.toString(2));
    }
}
