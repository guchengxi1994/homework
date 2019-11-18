package com.xiaoshuyui.FormOCR;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;

import java.io.*;

public class ReadJson  {

    public static String readFile(String filepath) {
        String jsonStr = "";

        try {
//            File jsonFile = new File("D:\\formOCR\\jsonData.txt");
            File jsonFile = new File(filepath);
            FileReader fileReader = new FileReader(jsonFile);
            Reader reader = new InputStreamReader(new FileInputStream(jsonFile),"utf-8");

            int ch = 0;
            StringBuffer sb = new StringBuffer();
            while ((ch = reader.read())!=-1){
                sb.append((char)ch);
            }
            fileReader.close();
            reader.close();
            jsonStr = sb.toString();
            return jsonStr;
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return null;
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
            return null;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }

    }

    public static void main(String[] args) {
        String filePath = "D:\\formOCR\\jsonData.txt";
        String s = readFile(filePath);
//        System.out.println(s);

        JSONObject jsonObject = JSON.parseObject(s);
        System.out.println(jsonObject.get("forms").toString());
    }
}
