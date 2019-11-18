package com.xiaoshuyui.FormOCR;

import com.baidu.aip.ocr.AipOcr;
import org.json.JSONObject;

import java.util.HashMap;

public class GetMsg extends Thread {

    private AipOcr client;
    private String request_id;
    private HashMap options;

    private String jsonData;

    public AipOcr getClient() {
        return client;
    }

    public void setClient(AipOcr client) {
        this.client = client;
    }

    public String getRequest_id() {
        return request_id;
    }

    public void setRequest_id(String request_id) {
        this.request_id = request_id;
    }

    public HashMap getOptions() {
        return options;
    }

    public void setOptions(HashMap options) {
        this.options = options;
    }

    public GetMsg(AipOcr client, String request_id, HashMap options) {
        this.client = client;
        this.request_id = request_id;
        this.options = options;
    }

    public String getMsg(AipOcr client, String request_id, HashMap options) {

//        System.out.println(request_id);
        JSONObject res2 = client.tableResultGet(request_id, options);
//        System.out.println(res2.toString(2));
        JSONObject result = res2.getJSONObject("result");
        return (String) result.get("ret_msg");
    }

    public String getData(AipOcr client, String request_id, HashMap options) {
        JSONObject res2 = client.tableResultGet(request_id, options);
//        System.out.println(res2.toString(2));
        JSONObject result = res2.getJSONObject("result");
        return (String) result.get("result_data");
    }

    public String getJsonData() {
        return jsonData;
    }

    public void run() {
//        String e = getMsg(this.client,this.request_id,this.options);

        String e = "";

        while (true) {
            if ("已完成".equals(e)) {
                try {
                    Thread.sleep(1000);
                    jsonData = getData(this.client, this.request_id, this.options);
//                    System.out.println(jsonData);
                } catch (InterruptedException ex) {
                    ex.printStackTrace();
                }

                break;
            } else {
                try {
//                    System.out.println("not OK");
                    Thread.sleep(1000);
                    e = getMsg(this.client, this.request_id, this.options);

                } catch (InterruptedException ex) {
                    ex.printStackTrace();
                }
            }
        }

//        if ("已完成".equals(e)){
//            jsonData = getData(this.client,this.request_id,this.options);
//        }else{
//
//
//        }


//        System.out.println(e);

    }
}
